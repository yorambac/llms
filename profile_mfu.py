"""
MFU profiling sweep for the 0.25B model.
Sweeps batch size and block size to find optimal throughput config.
Runs 50 warmup + 100 timed steps per config — no data needed, uses random tokens.

Usage: python profile_mfu.py
Results: results/mfu_profile.csv
"""

import csv
import time
import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn

# ---------------------------------------------------------------------------
# Same model as train.py but parameterised for 0.25B
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
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
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.wte  = nn.Embedding(vocab_size, n_embd)
        self.wpe  = nn.Embedding(block_size, n_embd)
        self.h    = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, idx, targets):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

# ---------------------------------------------------------------------------
# Configs to sweep
# ---------------------------------------------------------------------------

# 0.25B architecture candidates
ARCH_CONFIGS = [
    dict(n_embd=1024, n_head=16, n_layer=16, label="1024-16L"),  # ~252M
    dict(n_embd=896,  n_head=14, n_layer=20, label="896-20L"),   # ~238M
]

BATCH_SIZES  = [4, 8, 16, 32]
BLOCK_SIZES  = [512, 1024]

VOCAB_SIZE   = 50257
WARMUP_STEPS = 30
TIMED_STEPS  = 80
PEAK_BF16    = 116.6e12  # RTX 4070

# ---------------------------------------------------------------------------

def profile_config(model, batch_size, block_size, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    def step():
        x = torch.randint(0, VOCAB_SIZE, (batch_size, block_size), device=device)
        y = torch.randint(0, VOCAB_SIZE, (batch_size, block_size), device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # warmup
    for _ in range(WARMUP_STEPS):
        step()
    torch.cuda.synchronize()

    # timed
    t0 = time.perf_counter()
    for _ in range(TIMED_STEPS):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens_per_step = batch_size * block_size
    tps = tokens_per_step * TIMED_STEPS / elapsed
    N   = model.num_params()
    mfu = tps * 6 * N / PEAK_BF16

    mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    torch.cuda.reset_peak_memory_stats(device)

    return tps, mfu, mem_mb


def main():
    device = "cuda"
    torch.set_float32_matmul_precision("high")

    results_path = Path("results/mfu_profile.csv")
    results_path.parent.mkdir(exist_ok=True)

    print(f"\n{'Arch':<12} {'BS':<5} {'CTX':<6} {'Params':<10} {'Tok/s':<12} {'MFU%':<8} {'VRAM MB'}")
    print("─" * 68)

    rows = []
    best_tps = 0
    best_cfg = None

    for arch in ARCH_CONFIGS:
        model = GPT(
            vocab_size=VOCAB_SIZE,
            n_embd=arch["n_embd"],
            n_head=arch["n_head"],
            n_layer=arch["n_layer"],
            block_size=max(BLOCK_SIZES),
        ).to(device)
        model = torch.compile(model)
        print(f"  Compiled {arch['label']} ({model.num_params()/1e6:.1f}M params)")

        for batch_size, block_size in itertools.product(BATCH_SIZES, BLOCK_SIZES):
            torch.cuda.empty_cache()
            try:
                tps, mfu, mem_mb = profile_config(model, batch_size, block_size, device)
                status = "OOM" if mem_mb > 11_000 else "ok"
                marker = " ◀ best" if tps > best_tps else ""
                if tps > best_tps:
                    best_tps = tps
                    best_cfg = dict(arch=arch["label"], batch_size=batch_size,
                                    block_size=block_size, tps=tps, mfu=mfu)
                print(f"  {arch['label']:<12} {batch_size:<5} {block_size:<6} "
                      f"{model.num_params()/1e6:<10.1f} {tps:<12,.0f} {mfu*100:<8.1f} "
                      f"{mem_mb:<8.0f}{marker}")
                rows.append(dict(
                    arch=arch["label"], n_embd=arch["n_embd"],
                    n_head=arch["n_head"], n_layer=arch["n_layer"],
                    params_m=f"{model.num_params()/1e6:.1f}",
                    batch_size=batch_size, block_size=block_size,
                    tokens_per_sec=f"{tps:.0f}", mfu_pct=f"{mfu*100:.1f}",
                    vram_mb=f"{mem_mb:.0f}", status=status,
                ))
            except torch.cuda.OutOfMemoryError:
                print(f"  {arch['label']:<12} {batch_size:<5} {block_size:<6} {'':10} OOM")
                rows.append(dict(
                    arch=arch["label"], n_embd=arch["n_embd"],
                    n_head=arch["n_head"], n_layer=arch["n_layer"],
                    params_m=f"{model.num_params()/1e6:.1f}",
                    batch_size=batch_size, block_size=block_size,
                    tokens_per_sec="OOM", mfu_pct="OOM",
                    vram_mb="OOM", status="OOM",
                ))
                torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'─'*68}")
    if best_cfg:
        print(f"  Best config: arch={best_cfg['arch']}  batch={best_cfg['batch_size']}  "
              f"ctx={best_cfg['block_size']}  →  {best_cfg['tps']:,.0f} tok/s  "
              f"MFU={best_cfg['mfu']*100:.1f}%")
    print(f"  Results saved to {results_path}\n")


if __name__ == "__main__":
    main()
