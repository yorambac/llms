"""
MFU profiling sweep for the 0.45B model on H100.
Sweeps batch size and context length to find optimal throughput.
Runs 50 warmup + 100 timed steps per config — no data needed.

Usage: python profile_mfu_h100.py
Results: results/mfu_profile_h200.csv
"""

import csv
import time
import itertools
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Model (same as train_500m.py) ──────────────────────────────────────────

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
        return self.c_proj(F.scaled_dot_product_attention(q, k, v, is_causal=True)
                           .transpose(1, 2).contiguous().view(B, T, C))

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
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.wte     = nn.Embedding(vocab_size, n_embd)
        self.wpe     = nn.Embedding(block_size, n_embd)
        self.h       = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
    def forward(self, idx, targets):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for block in self.h:
            x = block(x)
        return F.cross_entropy(self.lm_head(self.ln_f(x)).view(-1, self.wte.weight.shape[0]),
                               targets.view(-1))
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# ── Config ─────────────────────────────────────────────────────────────────

ARCH = dict(n_embd=1408, n_head=22, n_layer=16, label="1408-16L")  # 0.45B

BATCH_SIZES  = [8, 16, 24, 32]
BLOCK_SIZES  = [1024]

VOCAB_SIZE   = 50257
WARMUP_STEPS = 50
TIMED_STEPS  = 100
PEAK_BF16    = 989e12   # H100 SXM dense bf16 TFLOPS

# ── Profiler ───────────────────────────────────────────────────────────────

def profile_config(model, batch_size, block_size, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    model.train()

    def step():
        x = torch.randint(0, VOCAB_SIZE, (batch_size, block_size), device=device)
        y = torch.randint(0, VOCAB_SIZE, (batch_size, block_size), device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for _ in range(WARMUP_STEPS):
        step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(TIMED_STEPS):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tps  = batch_size * block_size * TIMED_STEPS / elapsed
    mfu  = tps * 6 * model.num_params() / PEAK_BF16
    vram = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    return tps, mfu, vram


def main():
    device = "cuda"
    torch.set_float32_matmul_precision("high")

    results_path = Path("results/mfu_profile_h200.csv")
    results_path.parent.mkdir(exist_ok=True)

    arch = ARCH
    print(f"\nProfiling {arch['label']} on H200")
    print(f"{'Batch':>6} {'Ctx':>5} {'Params':>8}  {'Tok/s':>10} {'MFU%':>6} {'VRAM':>8}")
    print("─" * 55)

    rows = []
    best_tps = 0
    best_cfg = None

    model = GPT(
        vocab_size=VOCAB_SIZE,
        n_embd=arch["n_embd"],
        n_head=arch["n_head"],
        n_layer=arch["n_layer"],
        block_size=max(BLOCK_SIZES),
    ).to(device)
    model = torch.compile(model)
    params = model.num_params() if hasattr(model, 'num_params') else sum(p.numel() for p in model.parameters())
    print(f"  Compiled {arch['label']} ({params/1e6:.1f}M params)")

    for batch_size, block_size in itertools.product(BATCH_SIZES, BLOCK_SIZES):
        torch.cuda.empty_cache()
        try:
            tps, mfu, vram_gb = profile_config(model, batch_size, block_size, device)
            marker = " ◀ best" if tps > best_tps else ""
            if tps > best_tps:
                best_tps = tps
                best_cfg = dict(batch=batch_size, ctx=block_size, tps=tps, mfu=mfu*100)
            print(f"  {batch_size:>6} {block_size:>5} {params/1e6:>8.1f}  {tps:>10,.0f} {mfu*100:>5.1f}% {vram_gb:>7.1f}G{marker}")
            rows.append(dict(
                arch=arch["label"], n_embd=arch["n_embd"], n_head=arch["n_head"],
                n_layer=arch["n_layer"], params_m=f"{params/1e6:.1f}",
                batch_size=batch_size, block_size=block_size,
                tokens_per_sec=f"{tps:.0f}", mfu_pct=f"{mfu*100:.1f}",
                vram_gb=f"{vram_gb:.1f}", status="ok",
            ))
        except torch.cuda.OutOfMemoryError:
            print(f"  {batch_size:>6} {block_size:>5} {'':>8}  {'OOM':>10}")
            rows.append(dict(
                arch=arch["label"], n_embd=arch["n_embd"], n_head=arch["n_head"],
                n_layer=arch["n_layer"], params_m=f"{params/1e6:.1f}",
                batch_size=batch_size, block_size=block_size,
                tokens_per_sec="OOM", mfu_pct="OOM", vram_gb="OOM", status="OOM",
            ))
            torch.cuda.empty_cache()

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'─'*55}")
    if best_cfg:
        print(f"  Best: batch={best_cfg['batch']}  ctx={best_cfg['ctx']}  "
              f"→  {best_cfg['tps']:,.0f} tok/s  MFU={best_cfg['mfu']:.1f}%")
    print(f"  Results saved to {results_path}\n")


if __name__ == "__main__":
    main()
