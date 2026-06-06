"""
REMOTE H200 Training Dashboard — tracks the 0.45B run on the RunPod H200 pod.
Fetches results/run_500m.csv and GPU stats from the pod via SSH every 15s.

Run locally:
    streamlit run dashboard_remote_app.py --server.port 8503

Pod SSH details are at the top of this file — update if the pod changes.
"""

import io
import math
import subprocess

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── Remote pod config ───────────────────────────────────────────────────────

SSH_HOST    = "64.247.201.60"
SSH_PORT    = "12801"
SSH_KEY     = "~/.ssh/id_ed25519"
SSH_USER    = "root"
REMOTE_DIR  = "/llms"

SSH_BASE = [
    "ssh", "-i", SSH_KEY,
    "-p", SSH_PORT,
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=5",
    f"{SSH_USER}@{SSH_HOST}",
]

# ── Training config (must match train_500m.py) ──────────────────────────────

MAX_LR          = 1.9e-3
MIN_LR          = 1.9e-4
MAX_TOKENS      = 10_400_000_000
BATCH_SIZE      = 40
BLOCK_SIZE      = 1024
TOKENS_PER_STEP = BATCH_SIZE * BLOCK_SIZE
TOTAL_STEPS     = MAX_TOKENS // TOKENS_PER_STEP
WARMUP_STEPS    = TOTAL_STEPS // 100
N_PARAMS        = 452.9e6

# ── Page setup ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="REMOTE · H200 Training Dashboard",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st_autorefresh(interval=15_000, key="autorefresh")

st.markdown("<style>.block-container { padding-top: 1rem; }</style>",
            unsafe_allow_html=True)

# ── Data fetching ───────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def fetch_csv():
    try:
        out = subprocess.check_output(
            SSH_BASE + [f"cat {REMOTE_DIR}/results/run_h100.csv"],
            text=True, timeout=8,
        )
        df = pd.read_csv(io.StringIO(out))
        if df.empty or "tokens_seen" not in df.columns:
            return pd.DataFrame()
        df["tokens_seen_b"] = df["tokens_seen"] / 1e9
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=10)
def fetch_gpu():
    try:
        out = subprocess.check_output(
            SSH_BASE + [
                "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,"
                "power.draw,memory.used,memory.total --format=csv,noheader,nounits"
            ],
            text=True, timeout=8,
        ).strip()
        temp, util, pwr, mu, mt = out.split(", ")
        return dict(temp=int(temp), util=int(util), power=float(pwr),
                    mem_used=int(mu), mem_total=int(mt))
    except Exception:
        return None

@st.cache_data(ttl=10)
def fetch_latest_ckpt():
    try:
        out = subprocess.check_output(
            SSH_BASE + [f"ls {REMOTE_DIR}/checkpoints/run_h100/ 2>/dev/null | tail -1"],
            text=True, timeout=8,
        ).strip()
        return out or "none"
    except Exception:
        return "unknown"

def cosine_lr_curve():
    steps = list(range(0, TOTAL_STEPS, max(1, TOTAL_STEPS // 500)))
    lrs = []
    for s in steps:
        if s < WARMUP_STEPS:
            lrs.append(MAX_LR * s / WARMUP_STEPS)
        else:
            t = (s - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
            lrs.append(MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * t)))
    return steps, lrs

# ── Layout ──────────────────────────────────────────────────────────────────

st.title("🛰 REMOTE · H100 SXM · 0.45B Training Dashboard")
st.caption(f"Pod: {SSH_USER}@{SSH_HOST}:{SSH_PORT}  ·  Auto-refreshes every 15s")

df = fetch_csv()
g  = fetch_gpu()

# ── GPU row ─────────────────────────────────────────────────────────────────

st.subheader("GPU · H100 SXM (80 GB)")
c1, c2, c3, c4, c5 = st.columns(5)
if g:
    c1.metric("Utilisation", f"{g['util']}%")
    c2.metric("VRAM Used",   f"{g['mem_used']:,} / {g['mem_total']:,} MiB")
    c3.metric("Temperature", f"{g['temp']}°C")
    c4.metric("Power",       f"{g['power']:.0f} W")
    c5.metric("VRAM Free",   f"{g['mem_total'] - g['mem_used']:,} MiB")
else:
    st.warning("GPU stats unavailable — pod may be unreachable")

st.divider()

# ── Progress row ─────────────────────────────────────────────────────────────

st.subheader("Training Progress · 0.45B  (23 TPP · 10.4B tokens · ~18 hrs on H200)")

if df.empty:
    st.info("Waiting for training to start… (results/run_500m.csv not found on pod)")
else:
    latest   = df.iloc[-1]
    step     = int(latest["step"])
    pct      = step / TOTAL_STEPS * 100
    tokens_b = float(latest["tokens_seen"]) / 1e9
    tps      = float(latest["tokens_per_sec"])
    tpp      = float(latest["tokens_seen"]) / N_PARAMS
    eta_h    = (MAX_TOKENS - float(latest["tokens_seen"])) / tps / 3600 if tps > 0 else 0

    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Progress",        f"{pct:.1f}%")
    p2.metric("Step",            f"{step:,} / {TOTAL_STEPS:,}")
    p3.metric("Tokens seen",     f"{tokens_b:.2f}B / 10.4B")
    p4.metric("Tok/param (TPP)", f"{tpp:.1f}x  (target=23)")
    p5.metric("Throughput",      f"{tps:,.0f} tok/s")
    p6.metric("ETA",             f"{eta_h:.1f} h")

    st.progress(pct / 100)
    st.caption(f"Last checkpoint: {fetch_latest_ckpt()}   |   "
               f"MFU: {latest['mfu_pct']}%   |   "
               f"Elapsed: {float(latest['elapsed_s']) / 3600:.1f}h")

st.divider()

# ── Charts ───────────────────────────────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Loss Curves")
    if not df.empty:
        val_df = df[df["val_loss"].notna()].copy()

        sc, mn, mx = st.columns([3, 1, 1])
        smooth   = sc.slider("Smoothing (EMA α)", 0.0, 0.99, 0.9, 0.01, key="loss_smooth")
        ymin_str = mn.text_input("Y min", key="ymin", placeholder="auto")
        ymax_str = mx.text_input("Y max", key="ymax", placeholder="auto")

        yrange = None
        try:
            ylo = float(ymin_str) if ymin_str else None
            yhi = float(ymax_str) if ymax_str else None
            if ylo is not None or yhi is not None:
                yrange = [ylo, yhi]
        except ValueError:
            pass

        def ema(series, alpha):
            return series.ewm(alpha=1 - alpha, adjust=False).mean() if alpha > 0 else series

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=val_df["tokens_seen_b"], y=val_df["train_loss"],
            mode="lines", name="Train (raw)",
            line=dict(color="#636EFA", width=1, dash="dot"), opacity=0.3,
        ))
        fig.add_trace(go.Scatter(
            x=val_df["tokens_seen_b"], y=val_df["val_loss"],
            mode="lines", name="Val (raw)",
            line=dict(color="#EF553B", width=1, dash="dot"), opacity=0.3,
        ))
        fig.add_trace(go.Scatter(
            x=val_df["tokens_seen_b"], y=ema(val_df["train_loss"], smooth),
            mode="lines", name="Train (smooth)",
            line=dict(color="#636EFA", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=val_df["tokens_seen_b"], y=ema(val_df["val_loss"], smooth),
            mode="lines", name="Val (smooth)",
            line=dict(color="#EF553B", width=2.5),
        ))
        fig.update_layout(
            xaxis_title="Tokens seen (B)",
            yaxis=dict(title="Cross-entropy loss", range=yrange),
            legend=dict(x=0.55, y=0.95),
            margin=dict(l=40, r=20, t=20, b=40),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        best_val   = val_df["val_loss"].min()
        latest_val = val_df["val_loss"].iloc[-1]
        st.caption(f"Best val loss: **{best_val:.4f}**   Latest: **{latest_val:.4f}**   "
                   f"Delta from start: {latest_val - val_df['val_loss'].iloc[0]:+.4f}")
    else:
        st.info("No data yet.")

with col_right:
    st.subheader("Learning Rate Schedule")
    steps, lrs = cosine_lr_curve()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=[s * TOKENS_PER_STEP / 1e9 for s in steps], y=lrs,
        mode="lines", name="LR schedule",
        line=dict(color="#00CC96", width=2),
        fill="tozeroy", fillcolor="rgba(0,204,150,0.1)",
    ))
    if not df.empty:
        cur_tokens = float(df.iloc[-1]["tokens_seen"]) / 1e9
        cur_lr     = float(df.iloc[-1]["lr"])
        fig2.add_trace(go.Scatter(
            x=[cur_tokens], y=[cur_lr],
            mode="markers", name="Current",
            marker=dict(color="#FF6692", size=12, symbol="circle"),
        ))
    fig2.update_layout(
        xaxis_title="Tokens seen (B)",
        yaxis_title="Learning rate",
        yaxis=dict(tickformat=".1e"),
        legend=dict(x=0.7, y=0.95),
        margin=dict(l=40, r=20, t=20, b=40),
        height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"Warmup: {WARMUP_STEPS:,} steps ({WARMUP_STEPS * TOKENS_PER_STEP / 1e6:.0f}M tokens)   "
               f"Max LR: {MAX_LR:.1e}   Min LR: {MIN_LR:.1e}")

st.divider()

# ── Throughput ───────────────────────────────────────────────────────────────

st.subheader("Throughput & MFU over time")
if not df.empty and len(df) > 1:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["tokens_seen_b"], y=df["tokens_per_sec"].astype(float),
        mode="lines", name="Tok/s",
        line=dict(color="#AB63FA", width=1.5), yaxis="y1",
    ))
    fig3.add_trace(go.Scatter(
        x=df["tokens_seen_b"], y=df["mfu_pct"].astype(float),
        mode="lines", name="MFU%",
        line=dict(color="#FFA15A", width=1.5), yaxis="y2",
    ))
    fig3.update_layout(
        xaxis_title="Tokens seen (B)",
        yaxis=dict(title="Tok/s", side="left"),
        yaxis2=dict(title="MFU%", side="right", overlaying="y"),
        legend=dict(x=0.85, y=0.95),
        margin=dict(l=40, r=60, t=20, b=40),
        height=250,
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No data yet.")

st.caption("Auto-refreshes every 15s · REMOTE H200 pod · llm_train 0.45B run")
