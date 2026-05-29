"""
Chat interface for the trained 0.25B GPT.
Run: streamlit run chat_app.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import streamlit as st
from pathlib import Path

CKPT_DIR   = Path("checkpoints/run_250m")
BLOCK_SIZE = 1024
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model (must match train_250m.py) ─────────────────────────────────────────

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

# ── Load model ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    ckpts = sorted(CKPT_DIR.glob("step_*.pt"))
    if not ckpts:
        st.error("No checkpoint found in checkpoints/run_250m/")
        st.stop()
    path = ckpts[-1]
    ckpt = torch.load(path, map_location=DEVICE)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model = GPT().to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    return model, enc, path.name

# ── Generation ────────────────────────────────────────────────────────────────

def generate(model, enc, prompt, max_new_tokens=200, temperature=0.8, top_k=50):
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
    return enc.decode(idx[0].tolist())

# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="llm_train chat", page_icon="🧠", layout="centered")
st.title("🧠 0.25B GPT — Chat")

model, enc, ckpt_name = load_model()
st.caption(f"Loaded `{ckpt_name}` · {sum(p.numel() for p in model.parameters())/1e6:.0f}M params · {DEVICE}")

with st.sidebar:
    st.header("Generation settings")
    max_tokens  = st.slider("Max new tokens",  50, 500, 200, 50)
    temperature = st.slider("Temperature",     0.1, 2.0, 0.8, 0.05)
    top_k       = st.slider("Top-k",           1,   200,  50,  1)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type a prompt…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating…"):
            full_text = generate(model, enc, prompt, max_tokens, temperature, top_k)
            response = full_text[len(prompt):]
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
