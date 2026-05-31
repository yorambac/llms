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

## sft_alpaca_best.pt  ✓ recommended for inference (run 2)

| Field | Value |
|-------|-------|
| **Type** | SFT instruction-tuned model |
| **Base model** | pretrained_250m_step610000.pt |
| **Training data** | Alpaca 52k (tatsu-lab/alpaca) |
| **Run** | Run 2 (supersedes run 1) |
| **Best at** | Step 46,700 / 49,404 (epoch 4) |
| **Best SFT val loss** | 2.968 |
| **Pretrain val loss** | 3.968 → 3.947 (−0.02 — zero forgetting, slight improvement) |
| **LR schedule** | Cosine, max=2e-5, min=2e-6, 3% warmup |
| **Epochs** | 4 |
| **Replay** | 80% FineWeb per batch |
| **gen_ok** | 100% throughout all 4 epochs |
| **Factual accuracy** | Format correct, basic facts sometimes wrong (0.25B knowledge limit) |
| **Batch size** | 4 × 1024 |
| **Training time** | 3.3 hours |

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
