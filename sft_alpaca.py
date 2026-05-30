"""
SFT fine-tuning on Alpaca.
Loads pretrained 0.25B checkpoint, fine-tunes on instruction-following.

Usage:
  python sft_alpaca.py
  python sft_alpaca.py --resume   # resume from latest SFT checkpoint

Prompt format:
  ### Instruction:
  {instruction}

  ### Input:         ← omitted when empty
  {input}

  ### Response:
  {output}

Loss is computed only on the response tokens.

Checkpoints: checkpoints/sft_alpaca/epoch{E}_step{N}.pt  (every CKPT_EVERY steps)
Metrics:     results/sft_alpaca.csv
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
import tiktoken
from datasets import load_dataset
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn,
)

console = Console()

# ── Model (identical to train_250m.py) ────────────────────────────────────────

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
    def __init__(self, vocab_size=50257, n_embd=1024, n_head=16,
                 n_layer=16, block_size=1024):
        super().__init__()
        self.block_size = block_size
        self.wte     = nn.Embedding(vocab_size, n_embd)
        self.wpe     = nn.Embedding(block_size, n_embd)
        self.h       = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, labels=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if labels is not None:
            # -100 labels are ignored (instruction / padding tokens)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return loss
        return logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# ── Dataset ───────────────────────────────────────────────────────────────────

def build_prompt(instruction, inp):
    inp = inp.strip()
    if inp and inp.lower() != "<noinput>":
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"

class PretrainReplay:
    """Streams random chunks from train.bin for replay during SFT."""
    def __init__(self, path, block_size):
        if not Path(path).exists():
            self.data = None
            console.print("[yellow]Warning: data/train.bin not found — replay disabled")
            return
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        console.print(f"  Replay buffer: {len(self.data)/1e9:.1f}B tokens from {path}")

    def sample_batch(self, n, device):
        if self.data is None or n == 0:
            return None, None
        idxs = np.random.randint(0, len(self.data) - self.block_size - 1, size=n)
        xs, ys = [], []
        for i in idxs:
            chunk = torch.from_numpy(self.data[i: i + self.block_size + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return torch.stack(xs).to(device), torch.stack(ys).to(device)


class AlpacaDataset(torch.utils.data.Dataset):
    def __init__(self, enc, block_size, split="train", val_frac=0.05):
        console.print(f"[yellow]Loading Alpaca dataset ({split})…")
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        rows = list(ds)
        n = len(rows)
        cut = int(n * (1 - val_frac))
        rows = rows[:cut] if split == "train" else rows[cut:]

        self.examples = []
        skipped = 0
        for ex in rows:
            prompt = build_prompt(ex["instruction"], ex["input"])
            full   = prompt + ex["output"].strip()
            prompt_ids = enc.encode(prompt)
            full_ids   = enc.encode(full) + [enc.eot_token]

            if len(full_ids) > block_size + 1:
                skipped += 1
                full_ids = full_ids[:block_size + 1]

            # Shift: x=input tokens, y=target tokens (next-token prediction, matches pretraining)
            x_ids = full_ids[:-1]
            y_ids = full_ids[1:]

            n_prompt = min(len(prompt_ids), len(x_ids))
            # Mask instruction positions except the last (which predicts first response token)
            labels = list(y_ids)
            for i in range(n_prompt - 1):
                labels[i] = -100

            pad = block_size - len(x_ids)
            if pad > 0:
                x_ids  = list(x_ids) + [0] * pad
                labels = labels + [-100] * pad
            full_ids = x_ids  # rename for the append below

            self.examples.append((
                torch.tensor(full_ids, dtype=torch.long),
                torch.tensor(labels,   dtype=torch.long),
            ))

        console.print(f"  {len(self.examples):,} examples loaded  ({skipped} truncated)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# ── LR & MFU ─────────────────────────────────────────────────────────────────

def cosine_lr(step, max_lr, min_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    t = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))

def mfu(model, tps, peak=116.6e12):
    return tps * 6 * model.num_params() / peak

# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(ckpt_dir, epoch, step, model, optimizer, metrics):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"epoch{epoch:02d}_step{step:07d}.pt"
    raw = model.module if hasattr(model, "module") else model
    raw = getattr(raw, "_orig_mod", raw)
    torch.save({
        "epoch": epoch, "step": step,
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_gpu": torch.cuda.get_rng_state(),
        "metrics": metrics,
    }, path)
    ckpts = sorted(ckpt_dir.glob("epoch*_step*.pt"))
    for old in ckpts[:-3]:
        old.unlink()
    return path

def load_latest_checkpoint(ckpt_dir, model, optimizer, device):
    ckpts = sorted(ckpt_dir.glob("epoch*_step*.pt"))
    if not ckpts:
        return 0, 0, {}
    path = ckpts[-1]
    console.print(f"[yellow]Resuming from {path.name}")
    ckpt = torch.load(path, map_location=device)
    raw = model.module if hasattr(model, "module") else model
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    raw.load_state_dict(sd)
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["rng_cpu"].cpu())
    torch.cuda.set_rng_state(ckpt["rng_gpu"].cpu())
    console.print(f"  Resumed at epoch {ckpt['epoch']}, step {ckpt['step']:,}")
    return ckpt["epoch"], ckpt["step"], ckpt.get("metrics", {})

# ── Config ────────────────────────────────────────────────────────────────────

PRETRAINED_CKPT = Path("checkpoints/important/pretrained_250m_step610000.pt")
CKPT_DIR        = Path("checkpoints/sft_alpaca")
RESULTS_FILE    = Path("results/sft_alpaca.csv")

BLOCK_SIZE   = 1024
BATCH_SIZE   = 4
MAX_LR       = 2e-5
MIN_LR       = 2e-6
EPOCHS       = 4
WARMUP_FRAC  = 0.03   # 3% warmup
CKPT_EVERY        = 200
VAL_EVERY         = 100
VAL_BATCHES       = 20
PRETRAIN_VAL_EVERY   = 500
PRETRAIN_VAL_BATCHES = 40
GEN_EVAL_EVERY       = 500   # generation quality check (% empty responses)
GEN_EVAL_N           = 30    # number of Alpaca examples to generate on
REPLAY_FRAC  = 0.8    # 80% of each batch = random FineWeb chunks (anti-forgetting)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",     action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_true")
    args = parser.parse_args()

    device = "cuda"
    torch.set_float32_matmul_precision("high")
    enc = tiktoken.get_encoding("gpt2")

    # Build datasets (tokenises all examples in memory — takes ~30s first run)
    console.print("[yellow]Building datasets (tokenising Alpaca ~30s)…[/]")
    train_ds = AlpacaDataset(enc, BLOCK_SIZE, split="train")
    val_ds   = AlpacaDataset(enc, BLOCK_SIZE, split="val")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )
    replay = PretrainReplay("data/train.bin", BLOCK_SIZE)
    n_replay = max(1, int(BATCH_SIZE * REPLAY_FRAC))  # replay slots per batch
    pretrain_val = PretrainReplay("data/val.bin", BLOCK_SIZE)

    # Fixed set of examples for generation eval (same every time for comparability)
    console.print("[yellow]Loading gen-eval examples…[/]")
    ds_full = list(__import__("datasets").load_dataset("tatsu-lab/alpaca", split="train"))
    cut = int(len(ds_full) * 0.95)
    gen_eval_examples = ds_full[cut: cut + GEN_EVAL_N]

    # Fixed easy questions for sanity-checking generation quality in the log
    PROBE_QUESTIONS = [
        ("Name the capital of France.", ""),
        ("List three benefits of regular exercise.", ""),
        ("Write a one-sentence definition of gravity.", ""),
        ("Name the author of Romeo and Juliet.", ""),
        ("Translate the sentence to French.", "Hello, how are you?"),
    ]

    def quick_gen_eval(model, print_samples=False):
        # Use the uncompiled model — compiled model is traced with labels and
        # behaves unpredictably when called without labels for inference.
        raw = getattr(model, "_orig_mod", model)
        raw.eval()
        ok = 0
        with torch.no_grad():
            for ex in gen_eval_examples:
                prompt = build_prompt(ex["instruction"], ex["input"])
                ids = enc.encode(prompt)
                idx = torch.tensor([ids], dtype=torch.long, device=device)
                for _ in range(80):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = raw(idx[:, -BLOCK_SIZE:])
                    logits = logits[:, -1, :] / 0.7
                    v, _ = torch.topk(logits, 50)
                    logits[logits < v[:, [-1]]] = float("-inf")
                    next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
                    if next_tok.item() == enc.eot_token:
                        break
                    idx = torch.cat([idx, next_tok], dim=1)
                response = enc.decode(idx[0, len(ids):].tolist()).strip()
                if len(response) >= 10:
                    ok += 1

        if print_samples:
            console.print("  [dim]── sample responses ──")
            for inst, inp in PROBE_QUESTIONS:
                prompt = build_prompt(inst, inp)
                ids = enc.encode(prompt)
                idx = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    for _ in range(60):
                        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = raw(idx[:, -BLOCK_SIZE:])
                        logits = logits[:, -1, :] / 0.7
                        v, _ = torch.topk(logits, 50)
                        logits[logits < v[:, [-1]]] = float("-inf")
                        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
                        if nxt.item() == enc.eot_token:
                            break
                        idx = torch.cat([idx, nxt], dim=1)
                resp = enc.decode(idx[0, len(ids):].tolist()).strip()[:120]
                console.print(f"  ### Instruction: {inst}" + (f"  [Input: {inp}]" if inp else ""))
                console.print(f"  ### Response:    {resp[:150]}")

        return ok / len(gen_eval_examples) * 100

    # Model — load pretrained weights
    model = GPT().to(device)
    console.print(f"[bold cyan]0.25B SFT — {model.num_params()/1e6:.1f}M params[/]")

    if PRETRAINED_CKPT.exists():
        console.print(f"[yellow]Loading pretrained weights from {PRETRAINED_CKPT.name}")
        ckpt = torch.load(PRETRAINED_CKPT, map_location=device)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(sd)
    else:
        console.print("[red]Warning: pretrained checkpoint not found, training from scratch")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MAX_LR,
        betas=(0.9, 0.95), weight_decay=0.1,
    )

    start_epoch, start_step, saved_metrics = 0, 0, {}
    if args.resume:
        start_epoch, start_step, saved_metrics = load_latest_checkpoint(
            CKPT_DIR, model, optimizer, device
        )

    if not args.no_compile:
        console.print("[yellow]Compiling with torch.compile… (2-3 min, cache warm = ~10s)[/]")
        t_compile = time.time()
        model = torch.compile(model)
        console.print(f"[yellow]Compile done in {time.time()-t_compile:.0f}s — first step will be slow (kernel load), then full speed[/]")

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * EPOCHS
    warmup_steps    = max(1, int(total_steps * WARMUP_FRAC))

    console.rule(
        f"[bold green]SFT  epochs={EPOCHS}  steps/epoch={steps_per_epoch:,}  "
        f"total={total_steps:,}  warmup={warmup_steps:,}"
    )

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    csv_mode = "w" if start_step == 0 else "a"

    last_val_loss          = saved_metrics.get("val_loss", float("nan"))
    last_pretrain_val_loss = saved_metrics.get("pretrain_val_loss", float("nan"))
    last_gen_ok_pct        = saved_metrics.get("gen_ok_pct", float("nan"))
    last_mfu_pct           = saved_metrics.get("mfu_pct", 0.0)
    best_val_loss          = saved_metrics.get("best_val_loss", float("inf"))
    best_ckpt_path         = Path("checkpoints/important") / "sft_alpaca_best.pt"
    t_start = time.time()
    global_step = start_epoch * steps_per_epoch + start_step

    with open(RESULTS_FILE, csv_mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "epoch", "step", "global_step", "total_steps",
            "lr", "train_loss", "val_loss", "pretrain_val_loss",
            "gen_ok_pct", "tokens_per_sec", "mfu_pct", "elapsed_s",
        ])
        if csv_mode == "w":
            writer.writeheader()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[yellow]lr={task.fields[lr]:.1e}"),
            TextColumn("[white]loss={task.fields[loss]:.4f}"),
            TextColumn("[green]val={task.fields[val]:.4f}"),
            TextColumn("[magenta]MFU={task.fields[mfu]:.1f}%"),
            TextColumn("[cyan]{task.fields[tps]} tok/s"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=4,
        ) as progress:
            epoch_task = progress.add_task(
                "SFT", total=total_steps, completed=global_step,
                lr=MAX_LR, loss=float("nan"),
                val=last_val_loss, mfu=last_mfu_pct, tps="---",
            )

            for epoch in range(start_epoch, EPOCHS):
                # Skip batches already processed when resuming mid-epoch
                skip = start_step if epoch == start_epoch else 0

                for batch_idx, (x, labels) in enumerate(train_loader):
                    if batch_idx < skip:
                        continue

                    step = batch_idx
                    lr = cosine_lr(global_step, MAX_LR, MIN_LR, warmup_steps, total_steps)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                    x, labels = x.to(device), labels.to(device)

                    # Mix in pretraining replay (replaces last n_replay rows)
                    rx, ry = replay.sample_batch(n_replay, device)
                    if rx is not None:
                        x      = torch.cat([x[:-n_replay],      rx], dim=0)
                        labels = torch.cat([labels[:-n_replay],  ry], dim=0)

                    model.train()
                    t0 = time.perf_counter()
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(x, labels)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()

                    tps = (BATCH_SIZE * BLOCK_SIZE) / (time.perf_counter() - t0)

                    # SFT validation
                    if global_step % VAL_EVERY == 0:
                        model.eval()
                        vlosses = []
                        with torch.no_grad():
                            for i, (vx, vlabels) in enumerate(val_loader):
                                if i >= VAL_BATCHES:
                                    break
                                vx, vlabels = vx.to(device), vlabels.to(device)
                                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                                    vl = model(vx, vlabels)
                                vlosses.append(vl.item())
                        last_val_loss = sum(vlosses) / len(vlosses)
                        raw = model.module if hasattr(model, "module") else model
                        raw = getattr(raw, "_orig_mod", raw)
                        last_mfu_pct = mfu(raw, tps) * 100

                        # Generation quality check (skip step 0 — VRAM not yet warmed up)
                        if global_step > 0 and global_step % GEN_EVAL_EVERY == 0:
                            show = (global_step % 500 == 0)
                            last_gen_ok_pct = quick_gen_eval(model, print_samples=show)
                            torch.cuda.empty_cache()
                            console.print(f"  [bold {'green' if last_gen_ok_pct > 50 else 'red'}]"
                                          f"gen_ok={last_gen_ok_pct:.0f}%[/]  "
                                          f"({'OK' if last_gen_ok_pct > 50 else 'BROKEN — stop?'})")

                        # Pretraining perplexity (catastrophic forgetting check)
                        if global_step % PRETRAIN_VAL_EVERY == 0:
                            pt_losses = []
                            with torch.no_grad():
                                for _ in range(PRETRAIN_VAL_BATCHES):
                                    px, py = pretrain_val.sample_batch(BATCH_SIZE, device)
                                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                                        pl = model(px, py)
                                    pt_losses.append(pl.item())
                            last_pretrain_val_loss = sum(pt_losses) / len(pt_losses)

                        # Save best checkpoint (never rotated out)
                        if last_val_loss < best_val_loss:
                            best_val_loss = last_val_loss
                            raw = model.module if hasattr(model, "module") else model
                            raw = getattr(raw, "_orig_mod", raw)
                            torch.save({"epoch": epoch+1, "step": step,
                                        "global_step": global_step,
                                        "val_loss": best_val_loss,
                                        "model": raw.state_dict()}, best_ckpt_path)
                            console.print(f"  [bold green]★ New best val={best_val_loss:.4f} → saved to {best_ckpt_path.name}")

                        writer.writerow(dict(
                            epoch=epoch + 1, step=step, global_step=global_step,
                            total_steps=total_steps, lr=f"{lr:.8f}",
                            train_loss=f"{loss.item():.4f}",
                            val_loss=f"{last_val_loss:.4f}",
                            pretrain_val_loss=f"{last_pretrain_val_loss:.4f}",
                            gen_ok_pct=f"{last_gen_ok_pct:.1f}",
                            tokens_per_sec=f"{tps:.0f}",
                            mfu_pct=f"{last_mfu_pct:.1f}",
                            elapsed_s=f"{time.time()-t_start:.0f}",
                        ))
                        csvfile.flush()

                    if global_step > 0 and global_step % CKPT_EVERY == 0:
                        save_checkpoint(CKPT_DIR, epoch + 1, step, model, optimizer, {
                            "val_loss":          last_val_loss,
                            "pretrain_val_loss": last_pretrain_val_loss,
                            "gen_ok_pct":        last_gen_ok_pct,
                            "best_val_loss":     best_val_loss,
                            "mfu_pct":           last_mfu_pct,
                        })

                    progress.update(epoch_task, advance=1,
                        lr=lr, loss=loss.item(),
                        val=last_val_loss, mfu=last_mfu_pct,
                        tps=f"{tps:,.0f}",
                    )
                    global_step += 1

                # End of epoch checkpoint
                save_checkpoint(CKPT_DIR, epoch + 1, steps_per_epoch, model, optimizer, {
                    "val_loss": last_val_loss,
                    "mfu_pct":  last_mfu_pct,
                })
                console.print(f"[bold green]✓ Epoch {epoch+1} complete — val_loss={last_val_loss:.4f}")

                # Also save to important checkpoints at end of final epoch
                if epoch + 1 == EPOCHS:
                    important_path = Path("checkpoints/important") / f"sft_alpaca_epoch{EPOCHS}.pt"
                    raw = model.module if hasattr(model, "module") else model
                    raw = getattr(raw, "_orig_mod", raw)
                    torch.save({"epoch": EPOCHS, "model": raw.state_dict()}, important_path)
                    console.print(f"[bold cyan]Saved to {important_path}")

                start_step = 0  # reset skip after first epoch

    console.print(
        f"[bold green]✓ SFT complete in {(time.time()-t_start)/3600:.1f}h  "
        f"val_loss={last_val_loss:.4f}"
    )


if __name__ == "__main__":
    main()
