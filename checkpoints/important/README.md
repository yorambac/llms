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

## sft_alpaca_*.pt  *(to be added)*

Instruction-tuned version of the above, fine-tuned on the Alpaca dataset (52k examples, 3 epochs).
