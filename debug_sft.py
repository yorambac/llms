"""
Minimal SFT debug: 200 steps, checks generation quality every 10 steps.
Goal: find exactly when/why responses go empty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
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

enc = tiktoken.get_encoding("gpt2")

print("Loading pretrained weights...", flush=True)
ckpt = torch.load("checkpoints/important/pretrained_250m_step610000.pt", map_location=DEVICE)
sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
model = GPT().to(DEVICE)
model.load_state_dict(sd)
print("Model loaded.", flush=True)

print("Loading Alpaca...", flush=True)
ds = list(load_dataset("tatsu-lab/alpaca", split="train"))
cut = int(len(ds) * 0.95)
train_ex = ds[:cut]
eval_ex  = ds[cut:cut+10]  # fixed 10 eval examples

def build_prompt(instruction, inp):
    inp = inp.strip()
    if inp and inp.lower() != "<noinput>":
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:"
    return f"### Instruction:\n{instruction}\n\n### Response:"

def make_example(ex):
    prompt  = build_prompt(ex["instruction"], ex["input"])
    full    = prompt + ex["output"].strip()  # strip leading \n from outputs
    p_ids   = enc.encode(prompt)
    f_ids   = enc.encode(full) + [enc.eot_token]
    f_ids   = f_ids[:BLOCK_SIZE]
    n_p     = min(len(p_ids), len(f_ids))
    labels  = [-100]*n_p + f_ids[n_p:]
    labels  = labels[:BLOCK_SIZE]
    pad     = BLOCK_SIZE - len(f_ids)
    if pad > 0:
        f_ids  = f_ids  + [0]*pad
        labels = labels + [-100]*pad
    return torch.tensor(f_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

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

def check_gen(model, step):
    print(f"\n--- gen check @ step {step} ---", flush=True)
    model.eval()
    results = []
    for i, ex in enumerate(eval_ex[:3]):   # only 3 examples to keep it fast
        prompt = build_prompt(ex["instruction"], ex["input"])
        print(f"  ex{i}: inst='{ex['instruction'][:40]}' prompt_len={len(enc.encode(prompt))} tokens", flush=True)
        resp = gen_one(model, prompt).strip()
        results.append(resp)
        print(f"  ex{i}: resp={repr(resp[:80])}", flush=True)
    ok = sum(1 for r in results if len(r) >= 10)
    print(f"  => ok={ok}/3", flush=True)
    return ok

# ── Initial check (pretrained model) ─────────────────────────────────────────

print("\n--- Pretrained model ---")
check_gen(model, 0)

# ── Train for 200 steps, check every 10 ──────────────────────────────────────

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.95), weight_decay=0.1)

import random
random.shuffle(train_ex)

print("\n--- SFT training (no replay, no compile) ---", flush=True)
model.train()
for step in range(1, 201):
    ex = train_ex[(step-1) % len(train_ex)]
    x, labels = make_example(ex)
    x = x.unsqueeze(0).to(DEVICE)
    labels = labels.unsqueeze(0).to(DEVICE)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if step % 5 == 0:
        print(f"step {step:3d} loss={loss.item():.4f}", flush=True)

    if step % 10 == 0:
        ok = check_gen(model, step)
        model.train()
        if ok == 0 and step >= 20:
            print(f"\n*** BROKEN at step {step} — investigating ***")
            # Show logit distribution at first response token
            ex0 = eval_ex[0]
            prompt = build_prompt(ex0["instruction"], ex0["input"])
            ids = enc.encode(prompt)
            idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(idx)
            logits = logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            top5 = torch.topk(probs, 5)
            print("  Top 5 tokens after prompt:")
            for prob, tok in zip(top5.values, top5.indices):
                print(f"    {repr(enc.decode([tok.item()])):20s}  {prob.item()*100:.1f}%")
            model.train()

print("\nDone.")
