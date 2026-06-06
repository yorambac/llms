# Running on a Cloud GPU (RunPod)

When a local GPU is too slow or unavailable, RunPod spot H100s are the best value:
~$1.30–1.60/hr, done in ~19 hours for ~$25–30 total (vs 7+ days locally).

## Why RunPod spot H100

| Option | $/hr | Time (0.45B, 9B tokens remaining) | Total |
|--------|------|------------------------------------|-------|
| RTX 4090 Vast.ai spot | $0.14–0.39 | ~4.5 days | ~$20–40 |
| **H100 SXM RunPod spot** | **$1.30–1.60** | **~19 hrs** | **~$27** |
| H100 NVL Vast.ai | $2.00 | ~21 hrs | ~$42 |
| A100 SXM4 Vast.ai spot | $1.00–1.80 | ~2.3 days | ~$55–77 |

RunPod spot can be interrupted, but checkpoints save every 1000 steps (~7 min), so
at most 7 min of work is lost on interruption. The training script auto-resumes.

On an H100 (80 GB VRAM) there are no memory constraints — batch can be raised to 8
for better MFU (~38% vs 32% locally).

## Setup (~30–45 min one-time)

### 1. Rent the instance

1. Go to [runpod.io](https://runpod.io) → **Secure Cloud** or **Community Cloud**
2. Filter for **H100 SXM** → select a spot instance
3. Choose template: **RunPod PyTorch** (has CUDA + conda pre-installed)
4. Set storage: ≥50 GB (for FineWeb data + checkpoint)
5. Deploy → wait for instance to start

### 2. Connect VSCode

1. Copy the SSH command from RunPod's instance page (e.g. `ssh root@<ip> -p <port>`)
2. Add your public key under RunPod **Settings → SSH Public Keys**
3. In VSCode: **Remote-SSH: Connect to Host** → paste the address
4. Open the remote folder in VSCode — Claude Code works identically from here

### 3. Set up the environment

```bash
# Clone repo
git clone https://github.com/yorambac/llms.git
cd llms

# Create conda env (or use the pre-installed PyTorch env)
conda create -n llm_train python=3.11 -y
conda activate llm_train
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy tiktoken rich streamlit plotly pandas streamlit-autorefresh datasets

# Download and tokenize FineWeb (~20 min, unattended)
python prepare_data.py
```

### 4. Upload checkpoint

From your **local machine**:

```bash
# Find latest checkpoint
ls checkpoints/run_500m/

# Upload to RunPod (get the scp address from RunPod's instance page)
scp -P <port> checkpoints/run_500m/step_XXXXXXX.pt root@<ip>:~/llms/checkpoints/run_500m/
```

### 5. Increase batch size (H100 has 80 GB)

Edit `train_500m.py`:
```python
BATCH_SIZE = 8   # up from 4 — H100 has plenty of VRAM
```

### 6. Launch training

```bash
cd ~/llms
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
python -u train_500m.py --resume > /tmp/train_500m.log 2>&1 &

# Dashboard
streamlit run dashboard_500m_app.py --server.port 8502 &
```

Compile takes ~5 min on H100 (80 GB VRAM — no OOM issues). After that it trains at
~27–30k tok/s (vs ~14k locally).

## On interruption (spot)

If the instance is interrupted, simply resume:
1. Rent a new spot instance (same or different host)
2. Repeat steps 2–4 (only the new checkpoint upload is needed, ~seconds)
3. `python train_500m.py --resume` — picks up from the last saved checkpoint

The worst case is losing ~7 min of training (1000 steps × 4 s/step).

## Compile cache note

The first run after a fresh instance always recompiles the model (~5–10 min).
After that, `TORCHINDUCTOR_FX_GRAPH_CACHE=1` caches the kernels and subsequent
restarts skip compilation entirely.
