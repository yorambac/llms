"""
Ladder training script — ~10M param GPT on FineWeb tokens.
Logs step metrics to results/ladder_results.csv.

Usage:
  python train.py --lr 3e-3 --run_name lr3e-3
"""

import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn,
)
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

# ---------------------------------------------------------------------------
# Model: ~10M param GPT (n_layer=6, n_head=6, n_embd=192, block_size=512)
# ---------------------------------------------------------------------------
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj  = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTConfig:
    def __init__(self, **kw):
        self.block_size = kw.get("block_size", 512)
        self.vocab_size = kw.get("vocab_size", 50257)
        self.n_layer    = kw.get("n_layer", 6)
        self.n_head     = kw.get("n_head", 6)
        self.n_embd     = kw.get("n_embd", 192)

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h   = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

class BinaryTokenDataset:
    def __init__(self, path, block_size):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.pos = 0

    def next_batch(self, batch_size, device):
        B, T = batch_size, self.block_size
        if self.pos + B * T + 1 > len(self.data):
            self.pos = 0
        chunk = torch.from_numpy(
            self.data[self.pos : self.pos + B * T + 1].astype(np.int64)
        )
        x = chunk[:-1].view(B, T).to(device)
        y = chunk[1:].view(B, T).to(device)
        self.pos += B * T
        return x, y

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def cosine_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def estimate_mfu(model, tokens_per_sec, dtype_flops=116.6e12):
    N = model.num_params()
    flops_per_token = 6 * N
    achieved = tokens_per_sec * flops_per_token
    return achieved / dtype_flops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",       type=float, default=3e-3)
    parser.add_argument("--run_name", type=str,   default="run")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--max_tokens", type=int, default=200_000_000)
    parser.add_argument("--data_dir",   type=str, default="data")
    parser.add_argument("--results_file", type=str, default="results/ladder_results.csv")
    parser.add_argument("--compile", action="store_true", default=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    cfg = GPTConfig(block_size=args.block_size)
    model = GPT(cfg).to(device)
    console.print(f"[bold cyan]Model params:[/] {model.num_params()/1e6:.2f}M")

    if args.compile:
        console.print("[bold yellow]Compiling model with torch.compile...[/]")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    data_dir = Path(args.data_dir)
    train_data = BinaryTokenDataset(data_dir / "train.bin", args.block_size)
    val_data   = BinaryTokenDataset(data_dir / "val.bin",   args.block_size)

    tokens_per_step = args.batch_size * args.block_size
    total_steps = args.max_tokens // tokens_per_step
    warmup_steps = max(1, total_steps // 100)

    results_path = Path(args.results_file)
    results_path.parent.mkdir(exist_ok=True)

    # Checkpointing
    ckpt_dir  = Path("checkpoints") / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "latest.pt"
    start_step   = 0
    tokens_seen  = 0
    if ckpt_path.exists():
        console.print(f"[yellow]Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        raw = model.module if hasattr(model, "module") else model
        raw.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step  = ckpt["step"] + 1
        tokens_seen = ckpt["tokens_seen"]
        console.print(f"  Resuming at step {start_step}/{total_steps}")

    write_header = not results_path.exists()

    console.rule(f"[bold green]Run: {args.run_name}  lr={args.lr}  steps={total_steps:,}  tok/step={tokens_per_step:,}")

    scaler = torch.amp.GradScaler(enabled=True)
    t_start = time.time()
    val_interval = 500
    ckpt_interval = 1000
    last_val_loss = float("nan")
    last_tps = 0.0
    last_mfu = 0.0

    with open(results_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "run_name", "lr", "step", "total_steps", "tokens_seen",
            "train_loss", "val_loss", "tokens_per_sec", "mfu_pct", "elapsed_s"
        ])
        if write_header:
            writer.writeheader()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[cyan]{task.fields[tps]} tok/s"),
            TextColumn("[yellow]loss={task.fields[loss]:.4f}"),
            TextColumn("[magenta]val={task.fields[val_loss]:.4f}"),
            TextColumn("[green]MFU={task.fields[mfu]:.1f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=4,
        ) as progress:
            task = progress.add_task(
                f"lr={args.lr}", total=total_steps, completed=start_step,
                tps="---", loss=float("nan"), val_loss=float("nan"), mfu=0.0,
            )

            for step in range(start_step, total_steps + 1):
                lr = cosine_lr(step, args.lr, args.lr * 0.1, warmup_steps, total_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                model.train()
                t0 = time.time()
                x, y = train_data.next_batch(args.batch_size, device)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                tokens_seen += tokens_per_step
                step_time = time.time() - t0
                tps = tokens_per_step / step_time
                last_tps = tps

                if step % val_interval == 0:
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for _ in range(20):
                            vx, vy = val_data.next_batch(args.batch_size, device)
                            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                                _, vl = model(vx, vy)
                            val_losses.append(vl.item())
                    last_val_loss = sum(val_losses) / len(val_losses)
                    raw_model = model.module if hasattr(model, "module") else model
                    last_mfu = estimate_mfu(raw_model, tps) * 100

                    elapsed = time.time() - t_start
                    row = dict(
                        run_name=args.run_name,
                        lr=args.lr,
                        step=step,
                        total_steps=total_steps,
                        tokens_seen=tokens_seen,
                        train_loss=f"{loss.item():.4f}",
                        val_loss=f"{last_val_loss:.4f}",
                        tokens_per_sec=f"{tps:.0f}",
                        mfu_pct=f"{last_mfu:.1f}",
                        elapsed_s=f"{elapsed:.0f}",
                    )
                    writer.writerow(row)
                    csvfile.flush()

                progress.update(
                    task, advance=1,
                    tps=f"{tps:,.0f}",
                    loss=loss.item(),
                    val_loss=last_val_loss,
                    mfu=last_mfu,
                )

                if step > 0 and step % ckpt_interval == 0:
                    raw = model.module if hasattr(model, "module") else model
                    torch.save({
                        "step": step,
                        "tokens_seen": tokens_seen,
                        "model": raw.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }, ckpt_path)

    total_time = time.time() - t_start
    console.print(f"[bold green]✓ {args.run_name} done in {total_time/60:.1f} min  "
                  f"val_loss={last_val_loss:.4f}  MFU={last_mfu:.1f}%")


if __name__ == "__main__":
    main()
