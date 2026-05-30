"""
Evaluate the SFT model on 200 Alpaca examples.
Generates responses and reports quality metrics.

Usage: python eval_sft.py [--ckpt path] [--n 200] [--max_tokens 150]
"""

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

# ── Model ─────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
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
    def __init__(self, vocab_size=50257, n_embd=1024, n_head=16, n_layer=16, block_size=1024):
        super().__init__()
        self.block_size = block_size
        self.wte     = nn.Embedding(vocab_size, n_embd)
        self.wpe     = nn.Embedding(block_size, n_embd)
        self.h       = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for block in self.h:
            x = block(x)
        return self.lm_head(self.ln_f(x))

# ── Generation ─────────────────────────────────────────────────────────────────

def build_prompt(instruction, inp):
    inp = inp.strip()
    if inp and inp.lower() != "<noinput>":
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:"
    return f"### Instruction:\n{instruction}\n\n### Response:"

def generate(model, enc, prompt, max_new_tokens=150, temperature=0.7, top_k=50, device="cuda"):
    ids = enc.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    eot = enc.eot_token
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1024:]
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            if next_tok.item() == eot:
                break
            idx = torch.cat([idx, next_tok], dim=1)
    return enc.decode(idx[0, len(ids):].tolist())

# ── Metrics ────────────────────────────────────────────────────────────────────

def score_response(response, reference):
    stripped = response.strip()
    metrics = {
        "empty":       len(stripped) == 0,
        "only_newlines": len(stripped) == 0 and len(response) > 0,
        "too_short":   0 < len(stripped) < 10,
        "ok":          len(stripped) >= 10,
        "len_chars":   len(stripped),
        "len_words":   len(stripped.split()),
        "ref_words":   len(reference.split()),
    }
    return metrics

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       default="checkpoints/important/sft_alpaca_best.pt")
    parser.add_argument("--n",          type=int, default=200)
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--temperature",type=float, default=0.7)
    parser.add_argument("--top_k",      type=int, default=50)
    parser.add_argument("--out",        default="results/eval_sft.jsonl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    console.print(f"[yellow]Loading {args.ckpt}…")
    ckpt = torch.load(args.ckpt, map_location=device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model = GPT().to(device)
    model.load_state_dict(sd)
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    console.print(f"  Loaded. Step={ckpt.get('global_step', ckpt.get('step','?'))}  val={ckpt.get('val_loss','?')}")

    # Load Alpaca val split (same 5% as training)
    console.print("[yellow]Loading Alpaca…")
    ds = list(load_dataset("tatsu-lab/alpaca", split="train"))
    cut = int(len(ds) * 0.95)
    val_examples = ds[cut:]   # same split as training
    eval_examples = val_examples[:args.n]
    console.print(f"  Evaluating on {len(eval_examples)} examples")

    # Run eval
    results = []
    counts = {"empty": 0, "too_short": 0, "ok": 0}

    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, "w") as f:
        for i, ex in enumerate(eval_examples):
            prompt   = build_prompt(ex["instruction"], ex["input"])
            response = generate(model, enc, prompt, args.max_tokens,
                                args.temperature, args.top_k, device)
            m = score_response(response, ex["output"])

            if m["empty"] or m["only_newlines"]:
                counts["empty"] += 1
            elif m["too_short"]:
                counts["too_short"] += 1
            else:
                counts["ok"] += 1

            record = {
                "i": i,
                "instruction": ex["instruction"],
                "input":       ex["input"],
                "reference":   ex["output"],
                "generated":   response.strip(),
                "metrics":     m,
            }
            results.append(record)
            f.write(json.dumps(record) + "\n")

            if (i + 1) % 20 == 0:
                console.print(f"  [{i+1}/{len(eval_examples)}]  "
                               f"ok={counts['ok']}  short={counts['too_short']}  empty={counts['empty']}")

    # Summary table
    total = len(results)
    console.print()
    t = Table(title="Eval Results", box=box.SIMPLE)
    t.add_column("Metric", style="cyan")
    t.add_column("Count", justify="right")
    t.add_column("Pct", justify="right", style="yellow")
    t.add_row("OK (≥10 chars)",   str(counts["ok"]),        f"{counts['ok']/total*100:.1f}%")
    t.add_row("Too short (<10c)", str(counts["too_short"]), f"{counts['too_short']/total*100:.1f}%")
    t.add_row("Empty/newlines",   str(counts["empty"]),     f"{counts['empty']/total*100:.1f}%")
    console.print(t)

    ok_results = [r for r in results if r["metrics"]["ok"]]
    if ok_results:
        avg_words = sum(r["metrics"]["len_words"] for r in ok_results) / len(ok_results)
        avg_ref   = sum(r["metrics"]["ref_words"] for r in ok_results) / len(ok_results)
        console.print(f"Avg generated words (ok only): {avg_words:.1f}  |  Avg reference words: {avg_ref:.1f}")

    # Show 5 sample outputs
    console.print("\n[bold]— Sample outputs —")
    for r in results[:5]:
        console.print(f"\n[cyan]Instruction:[/] {r['instruction'][:80]}")
        if r["input"]:
            console.print(f"[cyan]Input:[/]       {r['input'][:80]}")
        console.print(f"[green]Reference:[/]  {r['reference'][:120]}")
        console.print(f"[yellow]Generated:[/]  {repr(r['generated'][:120])}")

    console.print(f"\n[dim]Full results saved to {args.out}")


if __name__ == "__main__":
    main()
