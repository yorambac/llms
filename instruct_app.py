"""
Instruction-tuned model serving interface.
Run: streamlit run instruct_app.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import streamlit as st
from pathlib import Path

CKPT_PATH  = Path("checkpoints/important/sft_alpaca_best.pt")
BLOCK_SIZE = 1024
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

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

@st.cache_resource
def load_model():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model = GPT().to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    step = ckpt.get("global_step", ckpt.get("step", "?"))
    val  = ckpt.get("val_loss", "?")
    return model, enc, step, val

def build_prompt(instruction, input_text):
    inp = input_text.strip()
    if inp and inp.lower() != "<noinput>":
        return (f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n")
    return f"### Instruction:\n{instruction}\n\n### Response:\n"

def generate(model, enc, prompt, max_new_tokens, temperature, top_k):
    ids = enc.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    eot = enc.eot_token
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            with torch.amp.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            if next_tok.item() == eot:
                break
            idx = torch.cat([idx, next_tok], dim=1)
    return enc.decode(idx[0, len(ids):].tolist())

# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="0.25B Instruct", page_icon="🤖", layout="wide")
st.title("🤖 0.25B Instruct · Alpaca SFT")

model, enc, step, val = load_model()
val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
st.caption(f"Checkpoint: `sft_alpaca_best.pt` · step {step} · val loss {val_str} · {DEVICE.upper()}")

with st.sidebar:
    st.header("Generation")
    max_tokens  = st.slider("Max new tokens",  50, 500, 200, 50)
    temperature = st.slider("Temperature",     0.1, 2.0, 0.7, 0.05)
    top_k       = st.slider("Top-k",           1, 200, 50, 1)
    st.divider()
    st.header("Examples")
    examples = [
        ("Explain what a neural network is in simple terms.", ""),
        ("Write a Python function that checks if a number is prime.", ""),
        ("Summarise the following text in one sentence.", "The transformer architecture, introduced in the paper 'Attention is All You Need', relies entirely on attention mechanisms and has become the dominant approach in NLP and beyond."),
        ("What are three tips for staying healthy?", ""),
        ("Translate the following sentence to French.", "The weather is beautiful today."),
        ("What is the capital of Japan?", ""),
    ]
    for inst, inp in examples:
        if st.button(inst[:50] + ("…" if len(inst) > 50 else ""), use_container_width=True):
            st.session_state["instruction"] = inst
            st.session_state["input_text"]  = inp
            st.rerun()

# ── Main form ─────────────────────────────────────────────────────────────────

if "instruction" not in st.session_state:
    st.session_state["instruction"] = ""
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    instruction = st.text_area(
        "Instruction",
        key="instruction",
        height=120,
        placeholder="What do you want the model to do?",
    )
    input_text = st.text_area(
        "Input  *(optional — extra context or data)*",
        key="input_text",
        height=80,
        placeholder="Leave blank if not needed.",
    )

    run = st.button("▶ Generate", type="primary", use_container_width=True)

    with st.expander("Raw prompt sent to model"):
        if instruction:
            st.code(build_prompt(instruction, input_text), language=None)

with col2:
    st.subheader("Response")
    if run and instruction.strip():
        prompt = build_prompt(instruction, input_text)
        with st.spinner("Generating…"):
            response = generate(model, enc, prompt, max_tokens, temperature, top_k)
        st.session_state["last_response"] = response

    if "last_response" in st.session_state:
        st.markdown(st.session_state["last_response"])
        st.caption(f"{len(enc.encode(st.session_state['last_response']))} tokens generated")
    elif not run:
        st.info("Fill in an instruction and click Generate.")
    else:
        st.warning("Please enter an instruction.")
