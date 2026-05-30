# Important Checkpoints

## pretrained_250m_step610000.pt

| Field | Value |
|-------|-------|
| **Type** | Base pretrained model |
| **Architecture** | GPT, n_embd=1024, n_head=16, n_layer=16, block_size=1024 |
| **Parameters** | 253.9M |
| **Vocabulary** | GPT-2 tokenizer, 50,257 tokens |
| **Training data** | FineWeb `sample-10BT` (web text) |
| **Tokens trained** | 5.0B (Chinchilla optimal: 20× params) |
| **Steps** | 610,000 / 610,351 |
| **Final val loss** | 3.87 |
| **LR schedule** | Cosine, max=1e-3, min=1e-4, 1% warmup |
| **Batch size** | 8 × 1024 = 8,192 tokens/step |
| **MFU** | ~32% (24,500 tok/s on RTX 4070) |
| **Training time** | ~6.8 days |
| **Hardware** | NVIDIA RTX 4070 (12 GB VRAM) |

### Notes

- bf16 training, no GradScaler (not needed for bf16)
- torch.compile with TORCHINDUCTOR_FX_GRAPH_CACHE
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- This is a **base model** — does text completion, not instruction following
- For instruction following, apply SFT (see `sft_alpaca.py`)

### Loading

```python
import torch
from train_250m import GPT

ckpt = torch.load("checkpoints/important/pretrained_250m_step610000.pt", map_location="cpu")
sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
model = GPT()
model.load_state_dict(sd)
model.eval()
```

---

## sft_alpaca_best.pt  (run 1 — superseded by run 2, see below)

| Field | Value |
|-------|-------|
| **Type** | SFT instruction-tuned model |
| **Base model** | pretrained_250m_step610000.pt |
| **Training data** | Alpaca 52k (tatsu-lab/alpaca) |
| **Best at** | Step 12,300 / 12,351 (end of epoch 1) |
| **Best SFT val loss** | 2.994 |
| **Pretrain val loss** | 3.87 → 4.07 (+0.18 forgetting) |
| **LR schedule** | Cosine, max=2e-5, min=2e-6, 3% warmup |
| **gen_ok** | 93–100% (follows instruction format) |
| **Factual accuracy** | Poor — format correct, facts often wrong (0.25B limit) |
| **Note** | First run with correct label fix. Use run 2 (80% replay) instead. |
| **Batch size** | 4 × 1024, 80% FineWeb replay per batch |
| **Epochs** | 1 |
| **Training time** | 0.8 hours |

### Key findings from SFT experiments

Three runs were needed to get this right:

| Run | Epochs | Replay | Result |
|-----|--------|--------|--------|
| Run 1 | 3 | 10% | Heavy forgetting (+0.44 pretrain val), severe overfitting (best val at step 1400, then rose to 0.57) |
| Run 2 | 1 | 30% | Still forgetting (+0.14), overfitting past step 1300 |
| **Run 3** | **1** | **80%** | **No forgetting (±noise), best val 0.314 at step 1500** |

**Lessons:**
- 1 epoch is enough — Alpaca format is learned quickly (~1500 steps); more epochs = overfitting
- 80% FineWeb replay per batch is needed to prevent catastrophic forgetting at this scale
- Best checkpoint must be tracked separately (val loss rises after step 1500 — never use final weights)
- Train loss >> val loss is expected: train loss averages 80% FineWeb (~3.8) + 20% Alpaca, val loss is Alpaca-only

### Prompt format

```
### Instruction:
{your instruction here}

### Input:
{optional extra context, or leave blank}

### Response:
```
Then let the model complete from `### Response:`.

### Loading

```python
import torch
from train_250m import GPT

ckpt = torch.load("checkpoints/important/sft_alpaca_best.pt", map_location="cpu")
sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
model = GPT()
model.load_state_dict(sd)
model.eval()
```

---

## sft_alpaca_epoch1.pt  (final epoch weights — overfit, not recommended)

Same as above but saved at the end of epoch 1 (step 12,351). Val loss 0.378 vs best 0.314. Use `sft_alpaca_best.pt` instead.
