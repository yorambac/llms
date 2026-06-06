# Remote Run Log

Chronological log of what we're doing on the RunPod H200 pod. Updated at every significant step so sessions can be resumed without context loss.

---

## Current pod

| Field | Value |
|-------|-------|
| Pod name | middle_plum_parrotfish |
| GPU | NVIDIA H100 SXM, 80 GB VRAM |
| SSH | `ssh root@64.247.201.60 -p 12801 -i ~/.ssh/id_ed25519` |
| Price | ~$3.29/hr spot |
| Status | **Active — setting up** |

---

## What we're doing: H200 MFU fine-sweep (batch 32–64)

### Background

A coarse MFU sweep was already run on this pod ([results/mfu_profile_h200.csv](results/mfu_profile_h200.csv)).
The 0.45B model (n_embd=1408, n_head=22, n_layer=16) was profiled at batch ∈ {8,16,32,64,128} × ctx ∈ {1024,2048}.

Key findings from the coarse sweep (ctx=1024 column, the winner):

| Batch | Tok/s | MFU% | VRAM |
|-------|-------|------|------|
| 8 | 146,101 | 21.9% | 14.9 GB |
| 16 | 151,190 | 20.8% | 24.9 GB |
| 32 | 157,134 | 21.6% | 43.2 GB |
| 64 | 159,141 | 21.9% | 80.0 GB |
| 128 | OOM | — | — |

Note: MFU% is computed against H200 SXM *sparse* bf16 peak (1979 TFLOPS); true dense-peak MFU is ~2×.

**Observation:** Throughput essentially plateaus between b=32 (157k tok/s) and b=64 (159k tok/s) — only +1.3% for doubling the batch. We want to find the exact inflection point so we can pick the smallest batch that captures most of the plateau gain (saves VRAM, allows larger context or grad accum headroom).

### Plan

b=32 and b=64 are already in the results CSV — no need to repeat them. Only the middle points are missing.

1. **Modify `profile_mfu_h200.py`** — set `BATCH_SIZES = [40, 48, 56]`, `BLOCK_SIZES = [1024]` only (2048 not needed).
2. **Run the fine sweep** on the H200 pod — ~5 min for 3 configs.
3. **Merge with existing data** — combine with b=32 and b=64 rows already in the CSV to get the full picture.
4. **Select optimal batch** — smallest batch within ~0.5% of peak tok/s.
5. **Update README and this log** with the winning config.
6. **Launch 0.5B pretraining** with the optimal batch.

### Fine-sweep results (combined with existing data)

| Batch | Tok/s | MFU% | VRAM |
|-------|-------|------|------|
| 32 | 157,134 | 21.6% | 43.2 GB |
| 40 | 158,855 | 21.8% | 48.5 GB |
| **48** | **160,756** | **22.1%** | **61.6 GB** ← peak |
| 56 | 159,536 | 21.9% | 70.8 GB |
| 64 | 159,141 | 21.9% | 80.0 GB |

**Winner: batch=48** — actual peak, not just a plateau. b=56 and b=64 are slightly slower (kernel scheduling overhead). 61.6 GB VRAM, comfortable headroom on H200.

### LR ladder results (batch=48, 200M tokens/run)

| LR | Val Loss |
|----|----------|
| 7e-4 | 5.0429 |
| 1e-3 | 4.8499 |
| 1.3e-3 | 4.7281 |
| 1.7e-3 | 4.6979 |
| **2.3e-3** | **4.6273** ← winner |
| 3e-3 | 4.8934 |
| 4e-3 | 4.9717 |

Sharp peak at 2.3e-3. Curve descends from 7e-4 to 2.3e-3 then shoots back up — clear optimum.

### Status

- [x] Coarse MFU sweep complete (b ∈ {8,16,32,64,128}, ctx ∈ {1024,2048})
- [x] Fine MFU sweep complete (b ∈ {40,48,56}): optimal **b=48**, 160,756 tok/s
- [x] LR ladder complete: optimal **lr=2.3e-3** at batch=48 (confirmed across multiple sweeps)
- [x] train_500m.py updated with CLI overrides (local defaults preserved for ongoing local run)
- [x] Remote dashboard created: `dashboard_remote_app.py` — run locally, fetches from pod via SSH
- [x] H200 terminated after 9,500 steps — not worth keeping (same compute as H100, 3× cheaper)
- [x] Spin up H100 SXM spot ($3.29/hr) — pod: middle_plum_parrotfish
- [x] 4-point MFU sweep on H100: batch=40 wins at 157,017 tok/s (plateau shifts left vs H200 due to less bandwidth)
- [x] 0.5B pretraining launched: --batch_size 40 --lr 2.3e-3 --peak_tflops 989 --ckpt_dir checkpoints/run_h100 --results_file results/run_h100.csv

---

## Session history

### 2026-06-06 — Initial session

- Confirmed pod `flexible_jade_boa` is live; GPU idle, no training running.
- Repo is clean and up to date with `origin/main`.
- Read coarse MFU sweep results — plateau observed at b=32→64.
- Ran fine MFU sweep b=[40,48,56] ctx=1024. **b=48 is the true peak** at 160,756 tok/s / 22.1% MFU / 61.6 GB VRAM.
- Ran LR ladder [7e-4, 1e-3, 1.3e-3, 1.7e-3, 2.3e-3, 3e-3, 4e-3] at batch=48, 200M tokens/run. **lr=2.3e-3 is the winner** — sharp peak, drops off quickly above and below.
