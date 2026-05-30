"""
SFT training dashboard — Streamlit + Plotly.
Run: streamlit run sft_dashboard_app.py
Opens in browser at http://localhost:8501, auto-refreshes every 10s.
"""

import math
import subprocess
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_FILE = Path("results/sft_alpaca.csv")
CKPT_DIR     = Path("checkpoints/sft_alpaca")

MAX_LR   = 2e-5
MIN_LR   = 2e-6
EPOCHS   = 1
N_TRAIN  = 49402   # ~95% of 52002
N_PARAMS = 254e6

BATCH_SIZE  = 4
BLOCK_SIZE  = 1024
STEPS_PER_EPOCH = math.ceil(N_TRAIN / BATCH_SIZE)
TOTAL_STEPS     = STEPS_PER_EPOCH * EPOCHS
WARMUP_STEPS    = max(1, int(TOTAL_STEPS * 0.03))

st.set_page_config(
    page_title="SFT Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<meta http-equiv="refresh" content="10">
<style>.block-container { padding-top: 1rem; }</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def load_sft():
    if not RESULTS_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(RESULTS_FILE)
    except Exception:
        return pd.DataFrame()
    df["train_loss"]        = pd.to_numeric(df["train_loss"],        errors="coerce")
    df["val_loss"]          = pd.to_numeric(df["val_loss"],          errors="coerce")
    df["pretrain_val_loss"] = pd.to_numeric(df.get("pretrain_val_loss", float("nan")), errors="coerce")
    return df

@st.cache_data(ttl=3)
def gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        ).strip()
        temp, util, pwr, mu, mt = out.split(", ")
        return dict(temp=int(temp), util=int(util), power=float(pwr),
                    mem_used=int(mu), mem_total=int(mt))
    except Exception:
        return None

def latest_ckpt():
    ckpts = sorted(CKPT_DIR.glob("epoch*_step*.pt")) if CKPT_DIR.exists() else []
    return ckpts[-1].name if ckpts else "none"

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

def ema(series, alpha):
    return series.ewm(alpha=1 - alpha, adjust=False).mean() if alpha > 0 else series

# ── Layout ────────────────────────────────────────────────────────────────────

st.title("🎓 Alpaca SFT Dashboard · 0.25B")

df = load_sft()
g  = gpu_stats()

# ── GPU row ───────────────────────────────────────────────────────────────────

st.subheader("GPU · RTX 4070")
c1, c2, c3, c4, c5 = st.columns(5)
if g:
    c1.metric("Utilisation", f"{g['util']}%")
    c2.metric("VRAM Used",   f"{g['mem_used']:,} / {g['mem_total']:,} MiB")
    c3.metric("Temperature", f"{g['temp']}°C")
    c4.metric("Power",       f"{g['power']:.0f} W")
    c5.metric("VRAM Free",   f"{g['mem_total']-g['mem_used']:,} MiB")
else:
    st.warning("GPU stats unavailable")

st.divider()

# ── Progress ──────────────────────────────────────────────────────────────────

st.subheader("Training Progress")

if df.empty:
    st.info("Waiting for training to start… (results/sft_alpaca.csv not found)")
else:
    latest = df.iloc[-1]
    gs      = int(latest["global_step"])
    epoch   = int(latest["epoch"])
    step    = int(latest["step"])
    pct     = gs / TOTAL_STEPS * 100
    tps     = float(latest["tokens_per_sec"])
    eta_h   = (TOTAL_STEPS - gs) * BATCH_SIZE * BLOCK_SIZE / tps / 3600 if tps > 0 else 0
    epoch_pct = step / STEPS_PER_EPOCH * 100

    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Overall",      f"{pct:.1f}%")
    p2.metric("Epoch",        f"{epoch} / {EPOCHS}  ({epoch_pct:.0f}%)")
    p3.metric("Global step",  f"{gs:,} / {TOTAL_STEPS:,}")
    p4.metric("Throughput",   f"{tps:,.0f} tok/s")
    p5.metric("MFU",          f"{latest['mfu_pct']}%")
    p6.metric("ETA",          f"{eta_h:.1f} h")

    st.progress(min(pct / 100, 1.0))

    # Per-epoch progress bars
    for e in range(1, EPOCHS + 1):
        epoch_df = df[df["epoch"] == e]
        if epoch_df.empty:
            ep = 0.0
        elif e < epoch:
            ep = 100.0
        else:
            ep = int(epoch_df.iloc[-1]["step"]) / STEPS_PER_EPOCH * 100
        st.caption(f"Epoch {e}")
        st.progress(min(ep / 100, 1.0))

    st.caption(f"Last checkpoint: {latest_ckpt()}   |   "
               f"MFU: {latest['mfu_pct']}%   |   "
               f"Elapsed: {float(latest['elapsed_s'])/3600:.1f}h")

st.divider()

# ── Loss curves ───────────────────────────────────────────────────────────────

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Loss Curves")
    if not df.empty:
        val_df = df[df["val_loss"].notna()].copy()
        smooth = st.slider("Smoothing (EMA α)", 0.0, 0.99, 0.85, 0.01, key="sft_smooth")

        fig = go.Figure()
        # Raw (faint)
        fig.add_trace(go.Scatter(
            x=val_df["global_step"], y=val_df["train_loss"],
            mode="lines", name="Train (raw)",
            line=dict(color="#636EFA", width=1, dash="dot"), opacity=0.3,
        ))
        fig.add_trace(go.Scatter(
            x=val_df["global_step"], y=val_df["val_loss"],
            mode="lines", name="Val (raw)",
            line=dict(color="#EF553B", width=1, dash="dot"), opacity=0.3,
        ))
        # Smoothed
        fig.add_trace(go.Scatter(
            x=val_df["global_step"], y=ema(val_df["train_loss"], smooth),
            mode="lines", name="Train (smooth)",
            line=dict(color="#636EFA", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=val_df["global_step"], y=ema(val_df["val_loss"], smooth),
            mode="lines", name="Val (smooth)",
            line=dict(color="#EF553B", width=2.5),
        ))
        # Epoch boundaries
        for e in range(1, EPOCHS):
            fig.add_vline(x=e * STEPS_PER_EPOCH, line_dash="dash",
                          line_color="rgba(255,255,255,0.2)",
                          annotation_text=f"Epoch {e}", annotation_position="top")
        fig.update_layout(
            xaxis_title="Global step",
            yaxis_title="Cross-entropy loss (response tokens only)",
            legend=dict(x=0.6, y=0.95),
            margin=dict(l=40, r=20, t=20, b=40),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        best_val   = val_df["val_loss"].min()
        latest_val = val_df["val_loss"].iloc[-1]
        st.caption(f"Best SFT val: **{best_val:.4f}**   Latest: **{latest_val:.4f}**   "
                   f"Δ from start: {latest_val - val_df['val_loss'].iloc[0]:+.4f}")
    else:
        st.info("No data yet.")

# ── LR schedule ───────────────────────────────────────────────────────────────

with col_right:
    st.subheader("Learning Rate Schedule")
    steps, lrs = cosine_lr_curve()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=steps, y=lrs,
        mode="lines", name="LR schedule",
        line=dict(color="#00CC96", width=2),
        fill="tozeroy", fillcolor="rgba(0,204,150,0.1)",
    ))
    if not df.empty:
        cur_step = int(df.iloc[-1]["global_step"])
        cur_lr   = float(df.iloc[-1]["lr"])
        fig2.add_trace(go.Scatter(
            x=[cur_step], y=[cur_lr],
            mode="markers", name="Current",
            marker=dict(color="#FF6692", size=12),
        ))
    for e in range(1, EPOCHS):
        fig2.add_vline(x=e * STEPS_PER_EPOCH, line_dash="dash",
                       line_color="rgba(255,255,255,0.2)",
                       annotation_text=f"Epoch {e}")
    fig2.update_layout(
        xaxis_title="Global step",
        yaxis_title="Learning rate",
        yaxis=dict(tickformat=".1e"),
        legend=dict(x=0.7, y=0.95),
        margin=dict(l=40, r=20, t=20, b=40),
        height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"Max LR: {MAX_LR:.0e}   Min LR: {MIN_LR:.0e}   "
               f"Warmup: {WARMUP_STEPS:,} steps   Epochs: {EPOCHS}")

st.divider()

# ── Catastrophic forgetting ───────────────────────────────────────────────────

st.subheader("Catastrophic Forgetting · FineWeb Val Loss")
if not df.empty:
    pt_df = df[df["pretrain_val_loss"].notna()].copy()
    if pt_df.empty:
        st.info("FineWeb val eval runs every 1,000 steps — check back soon.")
    else:
        smooth_pt = st.slider("Smoothing", 0.0, 0.99, 0.5, 0.01, key="pt_smooth")
        fig_pt = go.Figure()
        # Baseline reference line
        fig_pt.add_hline(y=3.87, line_dash="dot", line_color="rgba(255,255,255,0.4)",
                         annotation_text="Pretrain baseline (3.87)",
                         annotation_position="bottom right")
        fig_pt.add_trace(go.Scatter(
            x=pt_df["global_step"], y=pt_df["pretrain_val_loss"],
            mode="lines+markers", name="FineWeb val (raw)",
            line=dict(color="#FFD700", width=1, dash="dot"), opacity=0.4,
            marker=dict(size=5),
        ))
        fig_pt.add_trace(go.Scatter(
            x=pt_df["global_step"], y=ema(pt_df["pretrain_val_loss"], smooth_pt),
            mode="lines+markers", name="FineWeb val (smooth)",
            line=dict(color="#FFD700", width=2.5),
            marker=dict(size=7),
        ))
        for e in range(1, EPOCHS):
            fig_pt.add_vline(x=e * STEPS_PER_EPOCH, line_dash="dash",
                             line_color="rgba(255,255,255,0.2)",
                             annotation_text=f"Epoch {e}")
        fig_pt.update_layout(
            xaxis_title="Global step",
            yaxis_title="Cross-entropy loss on FineWeb val",
            legend=dict(x=0.7, y=0.95),
            margin=dict(l=40, r=20, t=20, b=40),
            height=260,
        )
        st.plotly_chart(fig_pt, use_container_width=True)
        pt_latest = pt_df["pretrain_val_loss"].iloc[-1]
        delta = pt_latest - 3.87
        colour = "🟢" if delta < 0.05 else "🟡" if delta < 0.15 else "🔴"
        st.caption(f"{colour} Latest FineWeb val: **{pt_latest:.4f}**   "
                   f"Δ from pretrain baseline: **{delta:+.4f}**   "
                   f"({'no forgetting' if delta < 0.05 else 'slight forgetting' if delta < 0.15 else 'forgetting detected'})")
else:
    st.info("No data yet.")

st.divider()

# ── Throughput ────────────────────────────────────────────────────────────────

st.subheader("Throughput & MFU over time")
if not df.empty and len(df) > 1:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["global_step"], y=df["tokens_per_sec"].astype(float),
        mode="lines", name="Tok/s",
        line=dict(color="#AB63FA", width=1.5), yaxis="y1",
    ))
    fig3.add_trace(go.Scatter(
        x=df["global_step"], y=df["mfu_pct"].astype(float),
        mode="lines", name="MFU%",
        line=dict(color="#FFA15A", width=1.5), yaxis="y2",
    ))
    fig3.update_layout(
        xaxis_title="Global step",
        yaxis=dict(title="Tok/s", side="left"),
        yaxis2=dict(title="MFU%", side="right", overlaying="y"),
        legend=dict(x=0.85, y=0.95),
        margin=dict(l=40, r=60, t=20, b=40),
        height=220,
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No data yet.")

st.caption("Auto-refreshes every 10s · SFT on Alpaca")
