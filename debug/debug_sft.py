"""
Minimal SFT debug: 200 steps, checks generation quality every 10 steps.
Goal: find exactly when/why responses go empty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import numpy as np
from pathlib import Path
from datasets import load_dataset

DEVICE = "cuda"
BLOCK_SIZE = 1024

# ── Minimal model (same as train_250m.py) ─────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, 2)
        hs = C // self.n_head
        q = q.view(B,T,self.n_head,hs).transpose(1,2)
        k = k.view(B,T,self.n_head,hs).transpose(1,2)
        v = v.view(B,T,self.n_head,hs).transpose(1,2)
        return self.c_proj(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).contiguous().view(B,T,C))

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4*n_embd, bias=False)
        self.proj = nn.Linear(4*n_embd, n_embd, bias=False)
    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate="tanh"))

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd); self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd); self.mlp  = MLP(n_embd)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(50257, 1024); self.wpe = nn.Embedding(1024, 1024)
        self.h = nn.ModuleList([Block(1024, 16) for _ in range(16)])
        self.ln_f = nn.LayerNorm(1024); self.lm_head = nn.Linear(1024, 50257, bias=False)
        self.wte.weight = self.lm_head.weight
    def forward(self, idx, labels=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for block in self.h: x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if labels is not None:
            return F.cross_entropy(logits.view(-1, 50257), labels.view(-1), ignore_index=-100)
        return logits

# ── Setup ─────────────────────────────────────────────────────────────────────

print("=== debug_sft.py starting ===", flush=True)
print("Step 1/6: loading tokenizer...", flush=True)
enc = tiktoken.get_encoding("gpt2")
print("  done.", flush=True)

print("Step 2/6: loading pretrained weights (~3s)...", flush=True)
ckpt = torch.load("checkpoints/important/pretrained_250m_step610000.pt", map_location=DEVICE)
sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
model = GPT().to(DEVICE)
model.load_state_dict(sd)
print("  done.", flush=True)

print("Step 3/6: loading Alpaca dataset (~10s)...", flush=True)
ds = list(load_dataset("tatsu-lab/alpaca", split="train"))
cut = int(len(ds) * 0.95)
train_ex = ds[:cut]
eval_ex  = ds[cut:cut+10]
print(f"  done. {len(train_ex)} train, {len(eval_ex)} eval examples.", flush=True)

def build_prompt(instruction, inp):
    inp = inp.strip()
    if inp and inp.lower() != "<noinput>":
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"

def make_example(ex):
    prompt = build_prompt(ex["instruction"], ex["input"])
    full   = prompt + ex["output"].strip()
    p_ids  = enc.encode(prompt)
    f_ids  = enc.encode(full) + [enc.eot_token]
    f_ids  = f_ids[:BLOCK_SIZE + 1]  # one extra for the shift

    x_ids = f_ids[:-1]   # input:   tokens 0..T-1
    y_ids = f_ids[1:]    # targets: tokens 1..T  (SHIFTED — matches pretraining convention)

    # Mask instruction positions except the last one (which predicts the first response token)
    n_p = min(len(p_ids), len(x_ids))
    labels = list(y_ids)
    for i in range(n_p - 1):   # mask 0..n_p-2; keep n_p-1 (":"->"Paris" — key signal)
        labels[i] = -100

    pad = BLOCK_SIZE - len(x_ids)
    if pad > 0:
        x_ids  = list(x_ids) + [0] * pad
        labels = labels + [-100] * pad

    return torch.tensor(x_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def gen_one(model, prompt, max_tokens=30):
    ids = enc.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    generated = []
    with torch.no_grad():
        for t in range(max_tokens):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(idx[:, -BLOCK_SIZE:])
            logits = logits[:, -1, :] / 0.8
            v, _ = torch.topk(logits, 50)
            logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            tok = nxt.item()
            if tok == enc.eot_token:
                print(f"    [EOT at token {t}]", flush=True)
                break
            generated.append(tok)
            idx = torch.cat([idx, nxt], dim=1)
    return enc.decode(generated)

PROBE = [
    ("Name the capital of France.", ""),
    ("List three benefits of regular exercise.", ""),
    ("Write a one-sentence definition of gravity.", ""),
]

def check_gen(model, step):
    print(f"\n--- gen check @ step {step} ---", flush=True)
    model.eval()
    ok = 0
    for inst, inp in PROBE:
        prompt = build_prompt(inst, inp)
        resp = gen_one(model, prompt).strip()
        words = resp.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        good = len(resp) >= 10 and unique_ratio >= 0.4
        status = "OK" if good else ("REPETITIVE" if len(resp) >= 10 else "EMPTY")
        if good: ok += 1
        print(f"  [{status}] Q: {inst}", flush=True)
        print(f"         A: {repr(resp[:100])}", flush=True)
    print(f"  => ok={ok}/{len(PROBE)}", flush=True)
    return ok

# ── Replay buffer (matches real SFT) ─────────────────────────────────────────

REPLAY_FRAC = 0.6   # keep in sync with sft_alpaca.py
BATCH_SIZE  = 4

class PretrainReplay:
    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint16, mode="r") if Path(path).exists() else None
        if self.data is not None:
            print(f"Replay buffer: {len(self.data)/1e9:.1f}B tokens", flush=True)
        else:
            print("WARNING: no replay buffer (data/train.bin not found)", flush=True)
    def sample(self, n):
        if self.data is None or n == 0:
            return None, None
        idxs = np.random.randint(0, len(self.data) - BLOCK_SIZE - 1, n)
        xs, ys = [], []
        for i in idxs:
            c = torch.from_numpy(self.data[i:i+BLOCK_SIZE+1].astype(np.int64))
            xs.append(c[:-1]); ys.append(c[1:])
        return torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)

print("Step 4/6: setting up replay buffer...", flush=True)
replay = PretrainReplay("data/train.bin")
n_replay = max(1, int(BATCH_SIZE * REPLAY_FRAC))
print(f"  {n_replay} replay slots per batch of {BATCH_SIZE}.", flush=True)

# ── DataLoader (matches real SFT) ─────────────────────────────────────────────

class AlpacaDS(torch.utils.data.Dataset):
    def __init__(self, exs):
        self.items = [make_example(e) for e in exs]
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

print("Step 5/6: tokenising 5000 examples (~20s)...", flush=True)
loader = torch.utils.data.DataLoader(AlpacaDS(train_ex[:5000]), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print("  done.", flush=True)

# ── Initial check ─────────────────────────────────────────────────────────────

print("\n--- Pretrained model (step 0) ---")
check_gen(model, 0)

# ── Train 600 steps with batch=4 + replay + compile, check every 100 ─────────

print(f"\nStep 6/6: compiling with torch.compile (~10s with warm cache)...", flush=True)
model = torch.compile(model)
print("  done. Starting training loop.", flush=True)
print(f"  Config: batch={BATCH_SIZE}, replay={REPLAY_FRAC:.0%}, 600 steps, gen check every 100.", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.95), weight_decay=0.1)
raw = getattr(model, "_orig_mod", model)

step = 0
for x_batch, l_batch in loader:
    step += 1
    x_batch, l_batch = x_batch.to(DEVICE), l_batch.to(DEVICE)
    # mix in replay
    rx, ry = replay.sample(n_replay)
    if rx is not None:
        x_batch = torch.cat([x_batch[:-n_replay], rx], dim=0)
        l_batch  = torch.cat([l_batch[:-n_replay], ry], dim=0)

    model.train()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x_batch, l_batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    if step % 20 == 0:
        print(f"step {step:4d}  loss={loss.item():.4f}", flush=True)

    if step % 100 == 0:
        ok = check_gen(raw, step)
        if ok == 0:
            raw.eval()
            prompt = build_prompt(eval_ex[0]["instruction"], eval_ex[0]["input"])
            ids = enc.encode(prompt)
            idx = torch.tensor([ids], device=DEVICE)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = raw(idx)
            probs = torch.softmax(logits[0, -1, :], -1)
            top5 = torch.topk(probs, 5)
            print("  *** BROKEN — top5 first token:")
            for p, t in zip(top5.values, top5.indices):
                print(f"    {repr(enc.decode([t.item()])):20s}  {p.item()*100:.1f}%")

    if step >= 600:
        break

print("\nDone.")
