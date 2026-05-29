# llm_train

Training a 0.25B GPT from scratch on a single RTX 4070, then instruction-tuning it with SFT.

## Pipeline Overview

The full process to go from random weights to a chat-capable model:

| Step | Script | What it does | Time |
|------|--------|-------------|------|
| 1 | `prepare_data.py` | Download & tokenize FineWeb → `data/train.bin`, `data/val.bin` | ~20 min |
| 2 | `run_ladders.sh` | LR sweep at 0.01B scale to find best learning rate | ~50 min |
| 3 | `profile_mfu.py` | Sweep batch/ctx configs to find max MFU on this GPU | ~10 min |
| 4 | `train_250m.py` | Pretrain 0.25B GPT on 5B tokens (Chinchilla optimal) | ~6.8 days |
| 5 | `sft_alpaca.py` | SFT fine-tune on Alpaca 52k for instruction following | ~1.5 h |
| 6 | `chat_app.py` | Serve the model as a Streamlit chat interface | — |

Monitoring dashboards: `dashboard_app.py` (pretraining) · `sft_dashboard_app.py` (SFT)  
Dataset explorer: `tutorial/alpaca_explorer.py`

### Checkpoints

Important checkpoints are saved to `checkpoints/important/` — see [`checkpoints/important/README.md`](checkpoints/important/README.md).

---

## Hardware

| Component | Spec |
|-----------|------|
| CPU | Intel Core i7-12700K (12th Gen, 20 threads, 5.0 GHz boost) |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 4070 (12 GB VRAM) |
| Storage | ~730 GB free on NVMe |

## Target Scale

~0.25B parameters — fits comfortably in the RTX 4070's 12 GB VRAM for training with bf16/fp16 and a reasonable batch size.

## Training Strategy

### Scale Ladder (start here)

Before committing to a full 0.25B run, sweep hyperparameters at **0.01B (10M)** scale:
- Chinchilla optimal at 0.01B = 200M tokens ≈ **6 minutes per run**
- Use ladders to tune: learning rate, batch size, warmup schedule, architecture shape
- Transfer best config to 0.25B

#### Ladder Config

**Model:** ~12M param GPT (n_layer=6, n_head=6, n_embd=192, block_size=512, vocab=GPT-2 50257)
**Data:** FineWeb `sample-10BT` streamed from HuggingFace, GPT-2 tokenizer, 200M train + 20M val tokens
**Sweep:** learning rate × 4 — `[1e-3, 3e-3, 6e-3, 1e-2]`, cosine schedule, 1% warmup
**Config:** bf16, Flash Attention 2 (via `F.scaled_dot_product_attention`), torch.compile, batch=32
**Logging:** val loss every 500 steps + tokens/sec + MFU → `results/ladder_results.csv`

Scripts: [`prepare_data.py`](prepare_data.py) · [`train.py`](train.py) · [`run_ladders.sh`](run_ladders.sh)

#### Ladder Results

| Run | LR | Final Val Loss | Best Val Loss | Tok/s | MFU% | Time |
|-----|----|---------------|---------------|-------|------|------|
| lr1e-3 | 1e-3 | 4.7407 | **4.6878** ✓ | 273,905 | 17.5 | 11.7 min |
| lr3e-3 | 3e-3 | 4.7295 | 4.6959 | 283,621 | 18.1 | 12.2 min |
| lr6e-3 | 6e-3 | 4.8232 | 4.8016 | 283,837 | 18.1 | 12.2 min |
| lr1e-2 | 1e-2 | 4.8398 | 4.8241 | 283,392 | 18.1 | 12.2 min |

**Winner: lr=1e-3** — lr=3e-3 is close (Δ0.008), but lr=6e-3 and lr=1e-2 clearly overshoot (~0.13 worse). Higher LRs are too aggressive at this scale.

MFU settled at ~18% as expected for a 12M model (small matmuls, memory-bandwidth bound). This will improve significantly at 0.25B scale.

#### Recommended config for 0.25B run

Carry forward from ladders: **lr=1e-3**, cosine schedule, 1% warmup, bf16, torch.compile, Flash Attention. Scale batch size up to fill VRAM (~12 GB).

### MFU Profiling (0.25B scale)

Swept 2 architectures × 4 batch sizes × 2 context lengths = 16 configs at 0.25B scale.  
Script: [`profile_mfu.py`](profile_mfu.py) · Dashboard: [`profile_dashboard.py`](profile_dashboard.py)  
Results: [`results/mfu_profile.csv`](results/mfu_profile.csv)

Initial profiling ran in isolation (no desktop GPU processes), giving overly optimistic memory headroom. Re-tested with desktop running (~1.2 GB VRAM consumed by Xorg/gnome). Key findings:

- Batch=8, ctx=1024 fits with `torch.compile` (9.7 GB PyTorch + 1.2 GB desktop = 10.9 GB < 11.6 GB usable)
- Without compile the same config OOMs — compile saves ~1.3 GB of activation memory via kernel fusion
- Removed `GradScaler` (not needed for bf16; its fp32 gradient copies waste ~500 MB)
- Compute loss inside `forward()` to avoid retaining the `(B,T,vocab)` logits tensor during backward

| Arch | Params | Batch | Ctx | Tok/s | MFU% | VRAM |
|------|--------|-------|-----|-------|------|------|
| 1024-16L | 254M | 4 | 512 | 16,865 | 22.0% | 5.0 GB |
| 1024-16L | 254M | 4 | 1024 | 21,595 | 28.2% | 6.5 GB |
| 1024-16L | 254M | 8 | 512 | 22,279 | 29.1% | 6.5 GB |
| **1024-16L** | **254M** | **8** | **1024** | **24,942** | **32.6%** | **9.7 GB** |
| 1024-16L | 254M | 16 | 512 | 25,479 | 33.3% ★ | 9.7 GB |
| 1024-16L | 254M | 16 | 1024 | — | OOM | — |
| 896-20L | 239M | 4 | 512 | 17,460 | 21.4% | — |
| 896-20L | 239M | 4 | 1024 | 21,189 | 26.0% | 6.6 GB |
| 896-20L | 239M | 8 | 512 | 22,150 | 27.2% | 6.6 GB |
| 896-20L | 239M | ≥8 | 1024 | — | OOM | — |

**Selected config for 0.25B run:**

| Setting | Value | Reason |
|---------|-------|--------|
| Architecture | 1024-16L (n_embd=1024, n_head=16, n_layer=16) | Best MFU, ~254M params |
| Batch size | 8 | Allows ctx=1024; max that fits with desktop processes running |
| Context length | 1024 | 2× more context per update vs 512 at no throughput cost |
| Learning rate | 1e-3 | Ladder winner |
| MFU | **~31%** | 23,758 tok/s (measured in production with data I/O) |
| Estimated runtime | **~58 hours** (5B tokens @ 23.8k tok/s) | ≈ 2.4 days |

### Compute Budget (Chinchilla optimal: 20 tokens/param)

| Scale | Tokens | MFU 25% | MFU 35% | MFU 40% |
|-------|--------|---------|---------|---------|
| 0.01B | 200M | ~6 min | ~4 min | ~3 min |
| 0.25B | 5B | ~72 h | ~51 h | ~45 h |

The 0.25B full run is **2–3 days** depending on achieved MFU.

### MFU on the RTX 4070

Peak bf16 tensor throughput is **116.6 TFLOPS**. Realistic target is **35–40% MFU** (~40–46 effective TFLOPS). Key levers:

| Lever | Impact | Notes |
|-------|--------|-------|
| **bf16** | High | Mandatory — enables tensor cores |
| **Flash Attention 2** | High | Largest single win; cuts memory bandwidth pressure |
| **`torch.compile`** | Medium-High | Ada Lovelace benefits significantly from kernel fusion |
| **Batch size** | Medium | Larger batches → more compute-bound; tune up to VRAM limit |
| **Context length** | Medium | 1024 is a good sweep point; longer = more compute-bound attention |
| **Gradient checkpointing** | N/A | Not needed — 0.25B fits in 12 GB at bf16 |

The RTX 4070's memory bandwidth (504 GB/s) is the main bottleneck vs. data centre GPUs (A100: 2 TB/s). Flash Attention and large batch sizes are the primary mitigations.

## Selected Framework

**[litgpt](https://github.com/Lightning-AI/litgpt)** — Lightning AI's clean rewrite of popular architectures (Llama, Mistral, Phi, etc.) in plain PyTorch. Simple CLI, easy to read, supports pretraining and fine-tuning.

## Data Mix

Goal: **general knowledge**. Chinchilla optimal budget at 0.25B = 5B tokens (a small fraction of any of these datasets).

### Selected: FineWeb

**[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)** (HuggingFace, 15T tokens) — filtered and deduplicated CommonCrawl. Best breadth for general knowledge; state-of-the-art quality filtering.

### Options Evaluated

| Dataset | Tokens | Type | Notes |
|---------|--------|------|-------|
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | 15T | Web | **Selected.** Best general-purpose breadth |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 1.3T | Educational web | Higher quality per token; skews academic |
| [Dolma](https://huggingface.co/datasets/allenai/dolma) | 3T | Multi-source | Web + Wikipedia + books + code; good diversity |
| [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) | 627B | Multi-source | Cleaned RedPajama; smaller but curated |
| [C4](https://huggingface.co/datasets/allenai/c4) | 750B | Web | Classic filtered CommonCrawl; superseded by FineWeb |

FineWeb-Edu is worth considering if reasoning quality matters more than topic diversity.

## Frameworks Evaluated

| Framework | Notes |
|-----------|-------|
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Single-file, great for intuition, GPT-2-style only |
| [litgpt](https://github.com/Lightning-AI/litgpt) | **Selected.** Clean, multi-architecture, pretraining + fine-tuning |
| [torchtune](https://github.com/pytorch/torchtune) | Config-driven, best for fine-tuning only |
| [TRL](https://github.com/huggingface/trl) | Best for SFT/RLHF on HuggingFace checkpoints |
