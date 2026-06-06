# Remote Run Log

Chronological log of what we're doing on the RunPod H200 pod. Updated at every significant step so sessions can be resumed without context loss.

---

## Current pod

| Field | Value |
|-------|-------|
| Pod name | flexible_jade_boa |
| GPU | NVIDIA H200 SXM, 140 GB VRAM |
| SSH | `ssh root@157.66.255.19 -p 17530 -i ~/.ssh/id_ed25519` |
| Status | **Active** |

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

### Status

- [x] Coarse sweep complete (b ∈ {8,16,32,64,128}, ctx ∈ {1024,2048})
- [x] Fine-sweep script updated (b ∈ {40,48,56}, ctx=1024 only)
- [x] Fine sweep run on H200
- [x] Optimal batch selected: **b=48**, 160,756 tok/s, 22.1% MFU
- [ ] README updated with final H200 config
- [ ] 0.5B pretraining launched

---

## Session history

### 2026-06-06 — Initial session

- Confirmed pod `flexible_jade_boa` is live; GPU idle, no training running.
- Repo is clean and up to date with `origin/main`.
- Read coarse MFU sweep results — plateau observed at b=32→64.
- Created this log and planned the fine-sweep.
- Ran fine-sweep b=[40,48,56] ctx=1024. **b=48 is the true peak** at 160,756 tok/s / 22.1% MFU / 61.6 GB VRAM.
