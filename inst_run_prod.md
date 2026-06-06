# Running on RunPod: Instance Setup Reference

Quick reference for setting up a RunPod pod and resuming training.
For the full rationale and GPU comparison see [`remote_run.md`](remote_run.md).

---

## Pod details (first run, June 2026)

| Field | Value |
|-------|-------|
| Provider | RunPod |
| GPU | NVIDIA H200, 140 GB VRAM |
| Pod ID | zolf505rs1g2gx (terminated — spot) |
| SSH | `ssh -i ~/.ssh/id_ed25519 root@157.66.255.19 -p 17530` |
| Price | ~$4.39/hr spot |

---

## One-time SSH key setup

RunPod doesn't always apply SSH keys added after pod creation. Workaround:

1. Enable **web terminal** on the pod page
2. Run this (replacing the key with your `~/.ssh/id_ed25519.pub` contents):

```bash
mkdir -p ~/.ssh && echo "ssh-ed25519 AAAA...your-key... yorambac@gmail.com" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
```

Then SSH from local works:
```bash
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@<ip> -p <port>
```

---

## Setup sequence (run once per fresh pod)

### 1. Clone repo

```bash
git clone https://github.com/yorambac/llms.git
cd /llms
```

### 2. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q --break-system-packages
pip install numpy tiktoken rich streamlit plotly pandas streamlit-autorefresh datasets -q --break-system-packages
```

Note: `--break-system-packages` needed because RunPod uses a system-managed Python.

### 3. Download FineWeb data (~20 min, run in background)

```bash
cd /llms && python -u prepare_data.py > /tmp/prepare_data.log 2>&1 &
tail -f /tmp/prepare_data.log   # watch progress
```

### 4. Upload checkpoint from local machine

Find the latest checkpoint locally:
```bash
ls checkpoints/run_500m/
```

Upload (run on **local machine**):
```bash
scp -i ~/.ssh/id_ed25519 -P <port> checkpoints/run_500m/step_XXXXXXX.pt root@<ip>:/llms/checkpoints/run_500m/
```

### 5. Update batch size for H200

Edit `train_500m.py` — with 140 GB VRAM there are no constraints:
```python
BATCH_SIZE = 8   # up from 4 (local RTX 4070 limit)
```

Also update the dashboard:
```python
BATCH_SIZE = 8   # in dashboard_500m_app.py
```

### 6. Launch training

```bash
cd /llms
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
nohup python -u train_500m.py > /tmp/train_500m.log 2>&1 &
echo "PID: $!"
tail -f /tmp/train_500m.log
```

First run compiles torch kernels (~5 min on H200). Subsequent restarts load from cache instantly.

### 7. Optional: dashboard

```bash
nohup streamlit run dashboard_500m_app.py --server.port 8502 --server.address 0.0.0.0 > /tmp/dashboard.log 2>&1 &
```

Expose port 8502 via RunPod's HTTP services panel to view in browser.

---

## On spot interruption

1. Deploy new pod (same GPU type)
2. Repeat SSH key setup (step above)
3. Steps 1–4 above (data download can be skipped if using a persistent volume)
4. Upload the latest local checkpoint
5. `python train_500m.py` — auto-resumes from checkpoint

Max data loss: ~7 min (1000 steps × ~0.4 s/step at batch=8).

---

## Monitoring from local machine

```bash
# Live log
ssh -i ~/.ssh/id_ed25519 root@<ip> -p <port> "tail -f /tmp/train_500m.log"

# Latest val loss
ssh -i ~/.ssh/id_ed25519 root@<ip> -p <port> "tail -3 /llms/results/run_500m.csv"

# GPU stats
ssh -i ~/.ssh/id_ed25519 root@<ip> -p <port> "nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader"
```
