"""
0.5B GPT pretraining on FineWeb.
Config: n_embd=1408, n_head=22, n_layer=18, batch=6, ctx=1024, lr=1e-3

Usage:
  python -u train_500m.py
  python -u train_500m.py --resume   # auto-resumes from latest checkpoint

Checkpoints: checkpoints/run_500m/step_XXXXXX.pt  (every 1000 steps)
Metrics:     results/run_500m.csv
"""

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn,
)

console = Console()

# ── Model ──────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc   = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=50257, n_embd=1408, n_head=22,
                 n_layer=18, block_size=1024):
        super().__init__()
        self.block_size = block_size
        self.wte  = nn.Embedding(vocab_size, n_embd)
        self.wpe  = nn.Embedding(block_size, n_embd)
        self.h    = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        if targets is not None:
            loss = F.cross_entropy(
                self.lm_head(x).view(-1, self.lm_head.out_features),
                targets.view(-1),
            )
            return loss
        return self.lm_head(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# ── Data ───────────────────────────────────────────────────────────────────

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
            self.data[self.pos: self.pos + B * T + 1].astype(np.int64)
        )
        x = chunk[:-1].view(B, T).to(device)
        y = chunk[1:].view(B, T).to(device)
        self.pos += B * T
        return x, y

# ── LR schedule ────────────────────────────────────────────────────────────

def cosine_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    t = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))

def mfu(model, tps, peak=116.6e12):
    return tps * 6 * model.num_params() / peak

# ── Checkpoint helpers ─────────────────────────────────────────────────────

def save_checkpoint(ckpt_dir, step, model, optimizer, metrics):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:07d}.pt"
    raw = model.module if hasattr(model, "module") else model
    raw = getattr(raw, "_orig_mod", raw)
    torch.save({
        "step":      step,
        "model":     raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_cpu":   torch.get_rng_state(),
        "rng_gpu":   torch.cuda.get_rng_state(),
        "metrics":   metrics,
    }, path)
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    for old in ckpts[:-3]:
        old.unlink()
    return path

def load_latest_checkpoint(ckpt_dir, model, optimizer, device):
    ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    if not ckpts:
        return 0, {}
    path = ckpts[-1]
    console.print(f"[yellow]Resuming from {path.name}")
    ckpt = torch.load(path, map_location=device)
    raw = model.module if hasattr(model, "module") else model
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    raw.load_state_dict(sd)
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["rng_cpu"].cpu())
    torch.cuda.set_rng_state(ckpt["rng_gpu"].cpu())
    step = ckpt["step"]
    console.print(f"  Resumed at step {step:,}")
    return step, ckpt.get("metrics", {})

# ── Config ─────────────────────────────────────────────────────────────────

BATCH_SIZE   = 6
BLOCK_SIZE   = 1024
MAX_LR       = 1e-3
MIN_LR       = 1e-4
MAX_TOKENS   = 11_520_000_000   # 23 TPP × ~0.5B params = ~11.5B tokens (~9 days)
CKPT_EVERY   = 1000
VAL_EVERY    = 500
VAL_STEPS    = 20
DATA_DIR     = Path("data")
RESULTS_FILE = Path("results/run_500m.csv")
CKPT_DIR     = Path("checkpoints/run_500m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    torch.set_float32_matmul_precision("high")

    model = GPT().to(device)
    console.print(f"[bold cyan]0.5B GPT — {model.num_params()/1e6:.1f}M params[/]")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MAX_LR,
        betas=(0.9, 0.95), weight_decay=0.1,
        fused=True,
    )

    start_step = 0
    saved_metrics = {}

    if args.resume:
        start_step, saved_metrics = load_latest_checkpoint(
            CKPT_DIR, model, optimizer, device
        )

    if not args.no_compile:
        console.print("[yellow]Compiling with torch.compile (this takes ~1 min)...[/]")
        model = torch.compile(model)

    tokens_per_step = BATCH_SIZE * BLOCK_SIZE
    total_steps     = MAX_TOKENS // tokens_per_step
    warmup_steps    = total_steps // 100

    console.rule(f"[bold green]Training  steps={total_steps:,}  tok/step={tokens_per_step:,}  "
                 f"total={MAX_TOKENS/1e9:.1f}B tokens")

    train_data = BinaryTokenDataset(DATA_DIR / "train.bin", BLOCK_SIZE)
    val_data   = BinaryTokenDataset(DATA_DIR / "val.bin",   BLOCK_SIZE)

    if start_step > 0:
        train_data.pos = (start_step * tokens_per_step) % len(train_data.data)

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    csv_mode = "w" if start_step == 0 else "a"

    last_val_loss = saved_metrics.get("val_loss", float("nan"))
    last_mfu_pct  = saved_metrics.get("mfu_pct", 0.0)
    last_tps      = saved_metrics.get("tps", 0.0)
    t_start       = time.time()
    tokens_seen   = start_step * tokens_per_step

    with open(RESULTS_FILE, csv_mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "step", "total_steps", "tokens_seen", "lr",
            "train_loss", "val_loss", "tokens_per_sec", "mfu_pct", "elapsed_s",
        ])
        if csv_mode == "w":
            writer.writeheader()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=35),
            MofNCompleteColumn(),
            TextColumn("[yellow]lr={task.fields[lr]:.2e}"),
            TextColumn("[white]loss={task.fields[loss]:.4f}"),
            TextColumn("[green]val={task.fields[val]:.4f}"),
            TextColumn("[magenta]MFU={task.fields[mfu]:.1f}%"),
            TextColumn("[cyan]{task.fields[tps]} tok/s"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=4,
        ) as progress:
            task = progress.add_task(
                "0.5B", total=total_steps, completed=start_step,
                lr=MAX_LR, loss=float("nan"),
                val=last_val_loss, mfu=last_mfu_pct, tps="---",
            )

            for step in range(start_step, total_steps + 1):
                lr = cosine_lr(step, MAX_LR, MIN_LR, warmup_steps, total_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                model.train()
                t0 = time.perf_counter()
                x, y = train_data.next_batch(BATCH_SIZE, device)

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()

                tokens_seen += tokens_per_step
                tps = tokens_per_step / (time.perf_counter() - t0)
                last_tps = tps

                if step % VAL_EVERY == 0:
                    model.eval()
                    vlosses = []
                    with torch.no_grad():
                        for _ in range(VAL_STEPS):
                            vx, vy = val_data.next_batch(BATCH_SIZE, device)
                            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                                vl = model(vx, vy)
                            vlosses.append(vl.item())
                    last_val_loss = sum(vlosses) / len(vlosses)
                    raw = model.module if hasattr(model, "module") else model
                    raw = getattr(raw, "_orig_mod", raw)
                    last_mfu_pct = mfu(raw, tps) * 100

                    writer.writerow(dict(
                        step=step, total_steps=total_steps,
                        tokens_seen=tokens_seen, lr=f"{lr:.6f}",
                        train_loss=f"{loss.item():.4f}",
                        val_loss=f"{last_val_loss:.4f}",
                        tokens_per_sec=f"{tps:.0f}",
                        mfu_pct=f"{last_mfu_pct:.1f}",
                        elapsed_s=f"{time.time()-t_start:.0f}",
                    ))
                    csvfile.flush()

                if step > 0 and step % CKPT_EVERY == 0:
                    save_checkpoint(CKPT_DIR, step, model, optimizer, {
                        "val_loss": last_val_loss,
                        "mfu_pct":  last_mfu_pct,
                        "tps":      last_tps,
                    })

                progress.update(task, advance=1,
                    lr=lr, loss=loss.item(),
                    val=last_val_loss, mfu=last_mfu_pct,
                    tps=f"{tps:,.0f}",
                )

    console.print(f"[bold green]✓ Training complete in "
                  f"{(time.time()-t_start)/3600:.1f}h  "
                  f"val_loss={last_val_loss:.4f}")


if __name__ == "__main__":
    main()
