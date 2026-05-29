"""
Training dashboard — Streamlit + Plotly.
Run: streamlit run dashboard_app.py
Opens in browser at http://localhost:8501, auto-refreshes every 10s.
"""

import math
import subprocess
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Config ──────────────────────────────────────────────────────────────────

RESULTS_250M  = Path("results/run_250m.csv")
RESULTS_LADDER = Path("results/ladder_results.csv")
CKPT_DIR      = Path("checkpoints/run_250m")

MAX_LR       = 1e-3
MIN_LR       = 1e-4
MAX_TOKENS   = 5_000_000_000
BATCH_SIZE   = 8
BLOCK_SIZE   = 1024
TOKENS_PER_STEP = BATCH_SIZE * BLOCK_SIZE
TOTAL_STEPS  = MAX_TOKENS // TOKENS_PER_STEP
WARMUP_STEPS = TOTAL_STEPS // 100
N_PARAMS     = 254e6

st.set_page_config(
    page_title="llm_train dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Auto-refresh every 10 seconds
st.markdown("""
<meta http-equiv="refresh" content="10">
<style>
    .block-container { padding-top: 1rem; }
    .metric-label { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def load_250m():
    if not RESULTS_250M.exists():
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_250M)
    df["tokens_seen_b"] = df["tokens_seen"] / 1e9
    return df

@st.cache_data(ttl=60)
def load_ladder():
    if not RESULTS_LADDER.exists():
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_LADDER)
    df["tokens_seen_m"] = df["tokens_seen"] / 1e6
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

def latest_ckpt():
    ckpts = sorted(CKPT_DIR.glob("step_*.pt")) if CKPT_DIR.exists() else []
    return ckpts[-1].name if ckpts else "none"

# ── Layout ───────────────────────────────────────────────────────────────────

st.title("🧠 llm_train · 0.25B Training Dashboard")

df = load_250m()
ladder_df = load_ladder()
g = gpu_stats()

# ── GPU row ──────────────────────────────────────────────────────────────────

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

# ── Training progress row ─────────────────────────────────────────────────────

st.subheader("Training Progress · 0.25B run")

if df.empty:
    st.info("Waiting for training to start… (results/run_250m.csv not found)")
else:
    latest = df.iloc[-1]
    step       = int(latest["step"])
    pct        = step / TOTAL_STEPS * 100
    tokens_b   = float(latest["tokens_seen"]) / 1e9
    tps        = float(latest["tokens_per_sec"])
    tpp        = float(latest["tokens_seen"]) / N_PARAMS
    eta_h      = (MAX_TOKENS - float(latest["tokens_seen"])) / tps / 3600 if tps > 0 else 0

    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Progress",        f"{pct:.1f}%")
    p2.metric("Step",            f"{step:,} / {TOTAL_STEPS:,}")
    p3.metric("Tokens seen",     f"{tokens_b:.2f}B / 5B")
    p4.metric("Tok/param (TPP)", f"{tpp:.1f}x  (Chinchilla={N_PARAMS*20/1e9:.0f}B)")
    p5.metric("Throughput",      f"{tps:,.0f} tok/s")
    p6.metric("ETA",             f"{eta_h:.1f} h")

    st.progress(pct / 100)
    st.caption(f"Last checkpoint: {latest_ckpt()}   |   "
               f"MFU: {latest['mfu_pct']}%   |   "
               f"Elapsed: {float(latest['elapsed_s'])/3600:.1f}h")

st.divider()

# ── Charts row ────────────────────────────────────────────────────────────────

col_left, col_right = st.columns(2)

# Loss curves
with col_left:
    st.subheader("Loss Curves")
    if not df.empty:
        val_df = df[df["val_loss"].notna()].copy()
        smooth = st.slider("Smoothing (EMA α)", 0.0, 0.99, 0.9, 0.01, key="loss_smooth")

        def ema(series, alpha):
            return series.ewm(alpha=1 - alpha, adjust=False).mean() if alpha > 0 else series

        fig = go.Figure()
        # Raw (faint)
        fig.add_trace(go.Scatter(
            x=val_df["tokens_seen_b"], y=val_df["train_loss"],
            mode="lines", name="Train (raw)",
            line=dict(color="#636EFA", width=1, dash="dot"),
            opacity=0.3,
        ))
        fig.add_trace(go.Scatter(
            x=val_df["tokens_seen_b"], y=val_df["val_loss"],
            mode="lines", name="Val (raw)",
            line=dict(color="#EF553B", width=1, dash="dot"),
            opacity=0.3,
        ))
        # Smoothed
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
            yaxis_title="Cross-entropy loss",
            legend=dict(x=0.6, y=0.95),
            margin=dict(l=40, r=20, t=20, b=40),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        best_val = val_df["val_loss"].min()
        latest_val = val_df["val_loss"].iloc[-1]
        st.caption(f"Best val loss: **{best_val:.4f}**   Latest: **{latest_val:.4f}**   "
                   f"Δ from start: {latest_val - val_df['val_loss'].iloc[0]:+.4f}")
    else:
        st.info("No data yet.")

# LR schedule
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
    st.caption(f"Warmup: {WARMUP_STEPS:,} steps ({WARMUP_STEPS*TOKENS_PER_STEP/1e6:.0f}M tokens)   "
               f"Max LR: {MAX_LR:.0e}   Min LR: {MIN_LR:.0e}")

st.divider()

# ── MFU & throughput over time ────────────────────────────────────────────────

st.subheader("Throughput & MFU over time")
if not df.empty and len(df) > 1:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df["tokens_seen_b"], y=df["tokens_per_sec"].astype(float),
        mode="lines", name="Tok/s",
        line=dict(color="#AB63FA", width=1.5),
        yaxis="y1",
    ))
    fig3.add_trace(go.Scatter(
        x=df["tokens_seen_b"], y=df["mfu_pct"].astype(float),
        mode="lines", name="MFU%",
        line=dict(color="#FFA15A", width=1.5),
        yaxis="y2",
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

st.divider()

# ── Ladder results ─────────────────────────────────────────────────────────────

st.subheader("Ladder Results (0.01B LR sweep)")
if not ladder_df.empty:
    fig4 = go.Figure()
    colors = {"lr1e-3": "#636EFA", "lr3e-3": "#EF553B",
              "lr6e-3": "#00CC96", "lr1e-2": "#AB63FA"}
    for run, grp in ladder_df.groupby("run_name"):
        val = grp[grp["val_loss"].notna()]
        fig4.add_trace(go.Scatter(
            x=val["tokens_seen_m"], y=val["val_loss"].astype(float),
            mode="lines", name=run,
            line=dict(color=colors.get(run, None), width=2),
        ))
    fig4.update_layout(
        xaxis_title="Tokens seen (M)",
        yaxis_title="Val loss",
        legend=dict(x=0.75, y=0.95),
        margin=dict(l=40, r=20, t=20, b=40),
        height=300,
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Summary table
    summary = []
    for run, grp in ladder_df.groupby("run_name"):
        last = grp.iloc[-1]
        summary.append({
            "Run": run, "LR": last["lr"],
            "Final Val Loss": float(last["val_loss"]),
            "Best Val Loss": grp["val_loss"].astype(float).min(),
            "Tok/s": int(float(last["tokens_per_sec"])),
            "MFU%": float(last["mfu_pct"]),
        })
    st.dataframe(
        pd.DataFrame(summary).sort_values("Best Val Loss"),
        use_container_width=True, hide_index=True,
    )
else:
    st.info("No ladder results found.")

st.caption("Auto-refreshes every 10s · llm_train")
