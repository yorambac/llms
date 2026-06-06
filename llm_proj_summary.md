# LLM Training Project Summary

## What this project is

Training GPT-style language models from scratch on a single RTX 4070 (12 GB VRAM),
then instruction-tuning with SFT. Full pipeline: data prep → LR sweep → MFU profiling
→ pretraining → SFT fine-tune → instruction-following inference app.

---

## Completed runs

### 0.25B pretraining (train_250m.py)
- **Architecture:** n_embd=1024, n_head=16, n_layer=16
- **Config:** batch=8, ctx=1024, lr=1e-3 cosine (1e-4 min), 1% warmup
- **Data:** FineWeb `sample-10BT`, 5B tokens (Chinchilla optimal: 20 TPP)
- **Result:** val loss 3.87, MFU ~32%, ~6.8 days
- **Checkpoint:** `checkpoints/important/pretrained_250m_step610000.pt`

### SFT fine-tune (sft_alpaca.py)
Three runs needed to get right:

| Run | Config | Forgetting | Best SFT val |
|-----|--------|-----------|-------------|
| Buggy run 1 | 3 ep · 10% replay | +0.44 | invalid (label bug) |
| Buggy run 2 | 1 ep · 30% replay | +0.14 | invalid (label bug) |
| Post-fix run 1 | 1 ep · 60% replay | +0.18 | 2.994 |
| **Post-fix run 2** | **4 ep · 80% replay** | **−0.02** | **2.968 at step 46,700** |

Key bugs fixed:
- Off-by-one in label construction (`full_ids[t]` → `full_ids[t+1]`)
- Alpaca outputs have leading `\n` → use `.strip()`
- Prompt template trailing `\n` causes collapse → use `"### Response:"` not `"### Response:\n"`
- `<noinput>` sentinel not filtered from input field

Final SFT checkpoint: `checkpoints/important/sft_alpaca_best.pt` — served by `instruct_app.py`

---

## Current run: 0.453B pretraining (train_500m.py)

### Architecture search
Target was 0.5B without grad checkpointing. Extensive VRAM analysis on RTX 4070:
- torch.compile needs a ~3.65 GB temporary spike during first compilation
- This spike is model-size-dependent and doesn't show in steady-state OOM tests
- Browser + VSCode GPU processes (~700 MB) must be closed during first compile
- Final viable config: **n_embd=1408, n_layer=16, n_head=22, batch=4** = 452.9M params

Key finding: n_embd=1408 (wider matrices) gives better MFU than n_embd=1024 even
at smaller batch, because larger matmuls are more compute-bound.

MFU comparison (all on RTX 4070):

| Config | Params | Batch | Tok/s | MFU% |
|--------|--------|-------|-------|------|
| 0.25B (n_embd=1024, n_layer=16) | 254M | 8 | 27,944 | 36.5% |
| 0.45B (n_embd=1408, n_layer=16) | 453M | 4 | ~14,000 | ~33% |

### Training config
- **Tokens:** 10.4B (23 TPP — slightly above Chinchilla for better convergence)
- **LR:** 1e-3 cosine → 1e-4, 1% warmup
- **Steps:** 2,539,062 total
- **Launch:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_FX_GRAPH_CACHE=1 python -u train_500m.py`
- **Dashboard:** `streamlit run dashboard_500m_app.py --server.port 8502`

### Training status (as of ~step 330k, ~1.35B tokens)
- Val loss: declining slowly from 11.1 → ~4.7 (flat-looking due to small val set
  variance, but rolling minimum is improving)
- MFU: 32–33% consistently (dips every 1000 steps = checkpoint I/O, benign)
- LR: looks fine — no instability, smoothed train loss cleanly declining

### Practical notes
- **First compile takes 10+ min** from cold cache — normal, subsequent restarts are instant
- Close browser before first compile (browser GPU process uses ~240 MB, pushes compile spike over limit)
- Compile spike at batch=4: ~10.5 GB PyTorch — leaves ~0.5 GB headroom with browser closed
- Dashboard uses `streamlit-autorefresh` (not meta-refresh) so smoothing/y-axis controls persist

---

## Infrastructure

### Key GPU constraints (RTX 4070, 12 GB)
- Usable VRAM: ~11.56 GB (CUDA), minus ~650 MB desktop (Xorg + gnome-shell + VSCode GPU)
- Effective PyTorch budget: ~10.4 GB steady state
- torch.compile spike: ~3.65 GB extra on top of steady state (first run only)
- Without compile: activation memory too large for this model size (can't avoid compile)
- Compile saves ~1.3 GB of activation memory via kernel fusion

### Files
| File | Purpose |
|------|---------|
| `train_250m.py` | 0.25B pretraining (complete) |
| `train_500m.py` | 0.45B pretraining (in progress) |
| `sft_alpaca.py` | SFT fine-tune on Alpaca 52k |
| `instruct_app.py` | Inference UI for SFT model (Streamlit, port 8501) |
| `dashboard_app.py` | 0.25B training dashboard |
| `dashboard_500m_app.py` | 0.45B training dashboard (port 8502) |
| `results/run_500m.csv` | Live training metrics |
| `remote_run.md` | Guide for running on cloud GPU (RunPod H100 spot) |

---

## Cloud GPU (when needed)

**Best option: RunPod spot H100 SXM** — ~$1.30–1.60/hr, done in ~19 hours, ~$27 total.
With 80 GB VRAM: no compile OOM, can raise batch to 8 for better MFU (~38%).

Setup: VSCode Remote SSH → identical workflow to local. See [`remote_run.md`](remote_run.md).

Other options: H100 NVL on Vast.ai ($2.00/hr, ~$42), RTX 4090 on Vast.ai spot ($0.14–0.39/hr, ~$20–40 but 4.5 days).

---

## Next steps
1. Let 0.45B pretraining complete (~7 days local or ~19 hrs on RunPod H100 spot)
2. SFT fine-tune the 0.45B base model (same sft_alpaca.py, adjusted for new model architecture)
3. Serve with instruct_app.py (minor update to load new model)
