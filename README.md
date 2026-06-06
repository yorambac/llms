# llm_train

Training a 0.25B GPT from scratch on a single RTX 4070, then instruction-tuning it with SFT.

---

## New Session Info

**If the Claude session dies, here's everything needed to resume.**

### Active pod

| Field | Value |
|-------|-------|
| Pod name | middle_plum_parrotfish |
| GPU | NVIDIA H100 SXM, 80 GB VRAM |
| SSH | `ssh root@64.247.201.60 -p 12801 -i ~/.ssh/id_ed25519` |
| Repo on pod | `/llms` |
| Price | ~$3.29/hr spot |

### What we're doing

Training a 0.45B GPT (n_embd=1408, n_head=22, n_layer=16) on H100. All sweeps complete:

1. **MFU sweep** — done on H200 (b=48 optimal) and H100 (b=40 optimal — plateau shifts left due to lower HBM bandwidth).
2. **LR ladder** — done on H200 (lr=2.3e-3 @ b=48) and H100 (lr=1.9e-3 @ b=40 — linear scaling from H200 winner confirmed).
3. **0.5B pretraining** — running. Command: `python -u train_500m.py --batch_size 40 --lr 1.9e-3 --peak_tflops 989 --ckpt_dir checkpoints/run_h100 --results_file results/run_h100.csv`

### How to resume on new H100 pod

Sweeps already done — use known-good config (batch=40, lr=1.9e-3).

```bash
# 0. SSH key workaround (RunPod doesn't always apply keys post-creation — use web terminal):
mkdir -p ~/.ssh && echo "ssh-ed25519 AAAA...your-key..." >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys

# 1. Clone and install
git clone https://github.com/yorambac/llms.git /llms && cd /llms
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q --break-system-packages
pip install numpy tiktoken rich streamlit plotly pandas streamlit-autorefresh datasets -q --break-system-packages

# 2. Download data (~20 min)
python -u prepare_data.py > /tmp/prepare_data.log 2>&1 &
tail -f /tmp/prepare_data.log

# 3. Launch training (batch=40, lr=1.9e-3 validated on H100)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
nohup python -u train_500m.py --batch_size 40 --lr 1.9e-3 --peak_tflops 989 \
  --ckpt_dir checkpoints/run_h100 --results_file results/run_h100.csv \
  > /tmp/train_500m.log 2>&1 &
```

---

## Current Step

**Two hero runs active** (as of 2026-06-06)

### Remote — H100 SXM (middle_plum_parrotfish)

Step ~3,000 / 253,906 · val loss 4.52 · 158k tok/s · 43% MFU · ETA ~18h · ~$59 total

| Setting | Value |
|---------|-------|
| Batch / LR | 40 / 1.9e-3 (validated by H100 MFU + LR sweeps) |
| Total tokens | 10.4B — 253,906 steps |
| Checkpoint dir | `checkpoints/run_h100` |
| Results CSV | `results/run_h100.csv` (on pod) |

### Local — RTX 4070

Step ~635,000 / 2,539,062 · **25% done** · val loss ~4.7 · 13k tok/s · elapsed ~45h · ETA ~7 days

| Setting | Value |
|---------|-------|
| Batch / LR | 4 / 1e-3 (local defaults) |
| Total tokens | 10.4B — 2,539,062 steps |
| Checkpoint dir | `checkpoints/run_500m` |
| Results CSV | `results/run_500m.csv` |

### Dashboards

```bash
# Local run (RTX 4070) — http://localhost:8502
/home/yoram/miniconda3/envs/llm_train/bin/streamlit run dashboard_500m_app.py --server.port 8502

# Remote run (H100) — http://localhost:8503
/home/yoram/miniconda3/envs/llm_train/bin/streamlit run dashboard_remote_app.py --server.port 8503
```

Both dashboards can run simultaneously. The remote one fetches data via SSH every 15s — no action needed on the pod.

```bash
# Live log from H100 pod
ssh root@64.247.201.60 -p 12801 -i ~/.ssh/id_ed25519 "tail -f /tmp/train_500m.log"
```

---

## Completed Runs

### 0.25B pretraining (train_250m.py)

| Setting | Value |
|---------|-------|
| Architecture | n_embd=1024, n_head=16, n_layer=16, vocab=50257 |
| Config | batch=8, ctx=1024, lr=1e-3 cosine (min 1e-4), 1% warmup |
| Data | FineWeb sample-10BT, 5B tokens (20 TPP — Chinchilla optimal) |
| Runtime | ~6.8 days on RTX 4070 |
| Final val loss | 3.87 |
| MFU | ~32% (27,944 tok/s) |
| Checkpoint | `checkpoints/important/pretrained_250m_step610000.pt` |

### SFT fine-tune on 0.25B (sft_alpaca.py)

| Setting | Value |
|---------|-------|
| Dataset | Alpaca 52k |
| Replay | 80% FineWeb (prevents catastrophic forgetting) |
| Runtime | ~3.3 h on RTX 4070 |
| Best SFT val loss | 2.968 (step 46,700) |
| Forgetting (pretrain val Δ) | −0.02 (improved slightly) |
| Checkpoint | `checkpoints/important/sft_alpaca_best.pt` |

See [SFT Findings](#sft-findings) for full bug history and lessons.

---

## Pipeline Overview

The full process to go from random weights to a chat-capable model:

| Step | Script | What it does | Time |
|------|--------|-------------|------|
| 1 | `prepare_data.py` | Download & tokenize FineWeb → `data/train.bin`, `data/val.bin` | ~20 min |
| 2 | `run_ladders.sh` | LR sweep at 0.01B scale to find best learning rate | ~50 min |
| 3 | `profile_mfu.py` | Sweep batch/ctx configs to find max MFU on this GPU | ~10 min |
| 4a | `train_250m.py` | Pretrain 0.25B GPT on 5B tokens (Chinchilla optimal) | ~6.8 days |
| 4b | `train_500m.py` | Pretrain 0.5B GPT on 11.5B tokens (23 TPP) | ~9 days |
| 5 | `sft_alpaca.py` | SFT fine-tune on Alpaca 52k for instruction following | ~1.5 h |
| 6 | `instruct_app.py` | Serve the instruction-tuned model (task + input fields) | — |

Monitoring dashboards: `dashboard_app.py` (0.25B pretraining) · `dashboard_500m_app.py` (0.5B local run) · `sft_dashboard_app.py` (SFT) · `dashboard_remote_app.py` (H100 remote run — fetches live data from pod via SSH, run locally on port 8503)  
Dataset explorer: `tutorial/alpaca_explorer.py`

### Debug / utility scripts (`debug/`)

| Script | Purpose |
|--------|---------|
| `debug_sft.py` | Minimal 200-step SFT loop with per-step `gen_ok_pct` checks and top-5 token probability logging. Used to diagnose the Alpaca `\n` collapse bug. |
| `eval_sft.py` | Evaluates a checkpoint on N Alpaca examples: reports % empty responses, avg word count, shows sample outputs. |
| `profile_mfu.py` | Sweeps batch × context × architecture configs to find optimal MFU on the GPU. |
| `profile_dashboard.py` | Terminal dashboard for MFU profiling results. |
| `dashboard_250m.py` | Terminal (Rich) training dashboard — superseded by `dashboard_app.py` (Streamlit). |
| `dashboard.py` | Earlier terminal dashboard prototype — superseded. |
| `chat_app.py` | Base model (pretrained, not instruction-tuned) chat UI — superseded by `instruct_app.py`. |

### Checkpoints

Important checkpoints are saved to `checkpoints/important/` — see [`checkpoints/important/README.md`](checkpoints/important/README.md).

### 0.5B Pretraining

After completing the 0.25B run, a second pretraining run was planned at 0.5B scale. Key findings from the architecture/MFU search (results: [`results/mfu_profile_500m.csv`](results/mfu_profile_500m.csv)):

- With the desktop running, the true PyTorch VRAM budget is **~10.4 GB** (11.6 GB usable − 1.2 GB desktop)
- n_embd=1408 (wider matrices) gave **better MFU than n_embd=1024** even at smaller batch, because larger matmuls are more compute-bound
- The 0.5B config (n_embd=1408, n_layer=18) fit at batch=6 with **9.3 GB** — no gradient checkpointing needed
- MFU is **38.3%** vs 36.5% for the 0.25B run — wider matrices more than compensate for smaller batch

| Config | Params | Batch | Tok/s | MFU% | VRAM |
|--------|--------|-------|-------|------|------|
| 0.25B (n_embd=1024, n_layer=16) | 254M | 8 | 27,944 | 36.5% | 7.87 GB |
| 0.4B (n_embd=1408, n_layer=14) | 405M | 8 | 18,311 | 38.2% | 9.33 GB |
| **0.5B (n_embd=1408, n_layer=18)** | **501M** | **6** | **14,879** | **38.3%** | **9.30 GB** |

**Selected config for 0.5B run (local RTX 4070):**

| Setting | Value | Reason |
|---------|-------|--------|
| Architecture | n_embd=1408, n_head=22, n_layer=18 | Best MFU at ~0.5B, fits without grad checkpointing |
| Batch size | 6 | Max that fits at this model size with desktop running |
| Context length | 1024 | Same as 0.25B run |
| Learning rate | 1e-3 | Ladder winner, same as 0.25B |
| TPP | **23** | Slightly above Chinchilla (20) for better convergence |
| Tokens | **11.5B** | 23 × 501M params |
| MFU | **~38%** | 14,879 tok/s |
| Estimated runtime | **~9 days** | 11.5B tokens @ 14.9k tok/s |

### H200 MFU sweep (RunPod, 0.45B model)

MFU profiled on H200 SXM (140 GB VRAM) for the 0.45B config (n_embd=1408, n_head=22, n_layer=16). Coarse sweep + fine-sweep b=[40,48,56] to find the true peak. MFU% computed against H200 SXM sparse bf16 peak (1979 TFLOPS).

| Batch | Tok/s | MFU% | VRAM |
|-------|-------|------|------|
| 32 | 157,134 | 21.6% | 43.2 GB |
| 40 | 158,855 | 21.8% | 48.5 GB |
| **48** | **160,756** | **22.1%** | **61.6 GB** ← peak |
| 56 | 159,536 | 21.9% | 70.8 GB |
| 64 | 159,141 | 21.9% | 80.0 GB |

**Selected H200 config: batch=48, ctx=1024** — true throughput peak; b=56 and b=64 are marginally slower. See [`remote_log.md`](remote_log.md) for full session log.

### H100 MFU sweep (RunPod, 0.45B model)

H100 SXM has less HBM bandwidth than H200 (3.35 vs 4.8 TB/s), so the throughput plateau shifts left — smaller batches become optimal. Swept b=[40,48,56,64] ctx=1024. MFU% computed against H100 SXM dense bf16 peak (989 TFLOPS).

| Batch | Tok/s | MFU% | Notes |
|-------|-------|------|-------|
| **40** | **157,017** | **43.1%** | ← peak |
| 48 | slower | — | |
| 56 | slower | — | |
| 64 | slower | — | fits exactly at 80 GB |

**Selected H100 config: batch=40, ctx=1024.** Plateau shifts left vs H200 — larger batches don't help due to lower HBM bandwidth.

### LR ladder: H200 @ batch=48

Swept 200M tokens per LR, cosine schedule. Results from [`results/mfu_profile_h200.csv`](results/mfu_profile_h200.csv) (val loss at end of 200M tokens):

| LR | Val Loss |
|----|----------|
| 7e-4 | 5.0429 |
| 1e-3 | 4.8499 |
| 1.3e-3 | 4.7281 |
| 1.7e-3 | 4.6979 |
| **2.3e-3** | **4.6273** ← winner |
| 3e-3 | 4.8934 |
| 4e-3 | 4.9717 |

Sharp peak at 2.3e-3 — clear optimum, drops off quickly on both sides.

### LR ladder: H100 @ batch=40

H200 winner (2.3e-3) scales linearly to batch=40: 2.3e-3 × (40/48) = 1.92e-3. Swept [1.5e-3, 1.9e-3, 2.3e-3, 2.7e-3] to confirm. Results from [`results/ladder_h100_results.csv`](results/ladder_h100_results.csv) (val loss at end of 200M tokens):

| LR | Val Loss |
|----|----------|
| 1.5e-3 | 4.7214 |
| **1.9e-3** | **4.7206** ← winner (Δ0.0008 better) |
| 2.3e-3 | 4.9064 (+0.186) |
| 2.7e-3 | 4.8744 (+0.154) |

1.9e-3 and 1.5e-3 essentially tied; 2.3e-3 overshoots by ~0.19 — confirming the linear scaling prediction. Picked 1.9e-3 as it matches the scaled prediction and is marginally better.

### Cloud GPU pricing and cost breakdown

| GPU | Price/hr (spot) | Dense bf16 TFLOPS | HBM bandwidth |
|-----|----------------|------------------|---------------|
| H100 SXM | $3.29/hr | 989 TFLOPS | 3.35 TB/s |
| H200 SXM | $4.39/hr | 989 TFLOPS | 4.8 TB/s |
| B200 SXM | $5.89/hr | ~1,800 TFLOPS | 8.0 TB/s |

H100 and H200 have **identical compute** — H200's only advantage is higher memory bandwidth (+43%), which shifts the MFU-optimal batch size higher. For a 0.45B model at batch=40–48 the throughput difference is <3%; the H100 is ~25% cheaper per hour.

| Item | Duration (est.) | Cost (est.) |
|------|----------------|-------------|
| H200 MFU + LR sweeps (9 LR runs + fine MFU) | ~2 h | ~$9 |
| H200 aborted training (9,500 steps → stopped) | ~1 h | ~$4 |
| H100 MFU sweep (4 batch sizes) | <0.5 h | ~$2 |
| H100 LR ladder (4 runs × 200M tokens) | <0.5 h | ~$2 |
| H100 hero run (10.4B tokens, ~18 h) | ~18 h | ~$59 |
| **Total** | **~22 h** | **~$76** |

---

### SFT Findings

Four runs were needed to get SFT right — two pre-fix (label bug) and two post-fix:

| Run | Epochs | Replay | Forgetting | Best SFT val |
|-----|--------|--------|-----------|-------------|
| Buggy run 1 | 3 | 10% | +0.44 (severe) | — (off-by-one label bug; results invalid) |
| Buggy run 2 | 1 | 30% | +0.14 | — (off-by-one label bug; results invalid) |
| Post-fix run 1 | 1 | 60% | +0.18 | 2.994 at step ~1,500 |
| **Post-fix run 2** | **4** | **80%** | **−0.02 (zero)** | **2.968 at step 46,700** |

Key lessons and findings:

- **Root bug**: `make_example` had an off-by-one error — `labels[t] = full_ids[t]` (same position) instead of `full_ids[t+1]` (next token, matching pretraining). Every run before the fix was silently training on wrong targets.
- **Diagnosis**: 100% FineWeb replay kept the model healthy; 0% replay (pure SFT) also collapsed → proved the SFT training code itself was broken, not the replay fraction.
- **Format irrelevant**: All the `\n`/`:`/` ` loop issues were symptoms of the broken training corrupting the model, not the prompt format itself.
- **Post-fix run 1** (60% replay, 1 epoch): gen_ok=93–100%, SFT val=2.994, forgetting +0.18. Format works; some forgetting remains.
- **Post-fix run 2** (80% replay, 4 epochs): gen_ok=100% throughout all 4 epochs, SFT val=2.968, forgetting −0.02 (pretrain val *improved* slightly). Best checkpoint at step 46,700. Training time: 3.3 h.

### Critical bug in Alpaca data + prompt format

Two bugs caused all SFT runs to generate only empty/newline responses:

**Bug 1 — Alpaca outputs have leading `\n`:**  
Many entries in `tatsu-lab/alpaca` have outputs that begin with `\n` (e.g. `"\n1. First item\n2. Second item"`). After SFT the model learned that `\n` is the dominant first response token (~95–99% probability), then predicted `\n → \n → \n ...` forever. `.strip()` on the generated text yields empty.  
**Fix:** `ex["output"].strip()` when building training examples.

**Bug 2 — Prompt template ends with `\n`:**  
The original template `"### Response:\n"` ends with a newline. The model learned that `\n` (end of prompt) → `\n` (start of response) because `\n\n` patterns are extremely common throughout response bodies (lists, paragraphs). Even after fixing Bug 1, the model collapsed to predicting `\n` with 100% probability after `\n`.  
**Fix:** Remove the trailing newline — use `"### Response:"` (no `\n`). The model now predicts the first word directly after `:`, which has a diverse distribution.

**Bug 3 — `<noinput>` sentinel not filtered:**  
Some Alpaca examples use `"<noinput>"` as the input field value instead of `""`. The prompt builder treated this as real input, adding `### Input:\n<noinput>\n\n` to the prompt.  
**Fix:** `if inp and inp.lower() != "<noinput>"` in `build_prompt()`.

These bugs were diagnosed using `debug_sft.py` — a minimal 200-step SFT loop that checks `gen_ok_pct` (% of responses ≥10 chars) every 10 steps, with top-5 token probability logging when broken.

**Final working config:** 4 epochs · 80% FineWeb replay · `output.strip()` · `"### Response:"` (no trailing `\n`) · best-checkpoint tracking.

---

## Running on a Cloud GPU

RunPod spot H100 SXM is the best value for this model size. Setup takes ~30 min.

### GPU options

| GPU | $/hr (spot) | Dense bf16 TFLOPS | HBM bandwidth | Time (10.4B tok) | Total |
|-----|------------|------------------|---------------|-----------------|-------|
| RTX 4090 (Vast.ai) | $0.14–0.39 | ~330 | ~1 TB/s | ~4.5 days | ~$20–40 |
| **H100 SXM (RunPod)** | **$3.29** | **989** | **3.35 TB/s** | **~18 h** | **~$59** |
| H200 SXM (RunPod) | $4.39 | 989 | 4.8 TB/s | ~18 h | ~$79 |
| B200 SXM (RunPod) | $5.89 | ~1,800 | 8.0 TB/s | ~10 h | ~$59 |

H100 and H200 have **identical compute** — the only difference is HBM bandwidth (+43% for H200), which shifts the MFU-optimal batch size slightly. Throughput difference for 0.45B at batch=40–48 is <3%; H100 saves ~25%.

### Pod setup (per fresh pod)

#### 1. SSH key (RunPod workaround)

RunPod doesn't always apply SSH keys added after pod creation. Paste your key via the **web terminal** on the pod page:

```bash
mkdir -p ~/.ssh && echo "ssh-ed25519 AAAA...your-key... yorambac@gmail.com" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
```

Then from local: `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no`

#### 2. Clone repo and install deps

```bash
git clone https://github.com/yorambac/llms.git /llms && cd /llms
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q --break-system-packages
pip install numpy tiktoken rich streamlit plotly pandas streamlit-autorefresh datasets -q --break-system-packages
```

`--break-system-packages` is required because RunPod uses a system-managed Python.

#### 3. Download FineWeb data (~20 min)

```bash
python -u prepare_data.py > /tmp/prepare_data.log 2>&1 &
tail -f /tmp/prepare_data.log
```

#### 4. Launch training

Use the validated config for the GPU type (see MFU and LR ladder tables in [0.5B Pretraining](#05b-pretraining)):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
nohup python -u train_500m.py --batch_size 40 --lr 1.9e-3 --peak_tflops 989 \
  --ckpt_dir checkpoints/run_h100 --results_file results/run_h100.csv \
  > /tmp/train_500m.log 2>&1 &
tail -f /tmp/train_500m.log
```

First compile takes ~1–5 min. `TORCHINDUCTOR_FX_GRAPH_CACHE=1` caches compiled kernels — subsequent restarts skip compilation entirely.

#### 5. Monitor from local machine

```bash
# Live log
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519 "tail -f /tmp/train_500m.log"

# GPU stats
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519 \
  "nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader"

# Latest val loss
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519 "tail -3 /llms/results/run_h100.csv"
```

Or run the local Streamlit dashboard (fetches from pod via SSH):
```bash
/home/yoram/miniconda3/envs/llm_train/bin/streamlit run dashboard_remote_app.py --server.port 8503
```

### On spot interruption

1. Deploy new pod (same GPU type)
2. Repeat SSH key setup
3. Steps 2–3 above (data download can be skipped with a persistent volume)
4. Launch with same args — auto-resumes from last checkpoint
   Max data loss: ~1,000 steps × 40,960 tok/step ≈ 41M tokens (~4 min at 157k tok/s)

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
