"""
Live dashboard for the 0.25B training run.
Run in a separate terminal: python dashboard_250m.py
Auto-refreshes every 5s from results/run_250m.csv
"""

import csv
import math
import subprocess
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

RESULTS_FILE = Path("results/run_250m.csv")
CKPT_DIR     = Path("checkpoints/run_250m")
REFRESH      = 5

MAX_LR       = 1e-3
MIN_LR       = 1e-4
MAX_TOKENS   = 5_000_000_000
BATCH_SIZE   = 8
BLOCK_SIZE   = 1024
TOKENS_PER_STEP = BATCH_SIZE * BLOCK_SIZE
TOTAL_STEPS  = MAX_TOKENS // TOKENS_PER_STEP
WARMUP_STEPS = TOTAL_STEPS // 100


def read_rows():
    if not RESULTS_FILE.exists():
        return []
    try:
        return list(csv.DictReader(open(RESULTS_FILE)))
    except Exception:
        return []


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


def cosine_lr(step):
    ws = WARMUP_STEPS
    if step < ws:
        return MAX_LR * step / ws
    t = (step - ws) / (TOTAL_STEPS - ws)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + math.cos(math.pi * t))


def gpu_panel(g):
    W = 28
    if not g:
        return Panel("[dim]GPU unavailable", title="GPU", border_style="red")
    u  = int(g["util"] / 100 * W)
    m  = int(g["mem_used"] / g["mem_total"] * W)
    col = "green" if g["util"] > 80 else "yellow" if g["util"] > 30 else "red"
    t = Text()
    t.append(f"  Util   [{'█'*u}{'░'*(W-u)}] {g['util']:3d}%\n", style=col)
    t.append(f"  VRAM   [{'█'*m}{'░'*(W-m)}] {g['mem_used']:,}/{g['mem_total']:,} MiB\n")
    t.append(f"  Temp   {g['temp']}°C      Power {g['power']:.0f} / 200 W")
    return Panel(t, title="[bold]RTX 4070", border_style=col)


def lr_curve_panel(rows):
    """Show LR schedule with current position marked."""
    W = 60
    curve = []
    for i in range(W):
        s = int(i / W * TOTAL_STEPS)
        lr = cosine_lr(s)
        norm = (lr - MIN_LR) / (MAX_LR - MIN_LR)
        curve.append(norm)

    # ASCII sparkline (8 levels)
    chars = "▁▂▃▄▅▆▇█"
    spark = "".join(chars[max(0, min(7, int(v * 7)))] for v in curve)

    current_step = int(rows[-1]["step"]) if rows else 0
    cursor_pos = min(W - 1, int(current_step / TOTAL_STEPS * W))
    current_lr = float(rows[-1]["lr"]) if rows else MAX_LR

    t = Text()
    t.append("  " + spark[:cursor_pos], style="dim cyan")
    t.append(spark[cursor_pos], style="bold yellow")
    t.append(spark[cursor_pos+1:] + "\n", style="dim")
    t.append(f"  Current LR: {current_lr:.2e}   "
             f"Step {current_step:,} / {TOTAL_STEPS:,}   "
             f"Warmup ends: {WARMUP_STEPS:,}")
    return Panel(t, title="[bold]Learning Rate Schedule", border_style="yellow")


def loss_curve_panel(rows):
    """Sparkline of val loss history."""
    vals = [float(r["val_loss"]) for r in rows if r.get("val_loss", "nan") != "nan"]
    if len(vals) < 2:
        return Panel("[dim]Waiting for first val checkpoint...",
                     title="[bold]Loss Curve", border_style="green")

    lo, hi = min(vals), max(vals)
    chars = " ▁▂▃▄▅▆▇█"

    # Full history compressed to 70 chars
    W = 70
    if len(vals) > W:
        step = len(vals) / W
        compressed = [vals[int(i * step)] for i in range(W)]
    else:
        compressed = vals

    spark = ""
    for v in compressed:
        if hi > lo:
            idx = int((v - lo) / (hi - lo) * (len(chars) - 1))
            spark += chars[len(chars) - 1 - idx]
        else:
            spark += "─"

    latest = vals[-1]
    best   = min(vals)
    delta  = latest - vals[0] if len(vals) > 1 else 0

    t = Text()
    t.append(f"  {hi:.2f} ┐\n", style="dim")
    t.append(f"       │ {spark}\n", style="green")
    t.append(f"  {lo:.2f} ┘\n", style="dim")
    t.append(f"  Latest: {latest:.4f}   Best: {best:.4f}   "
             f"Δ from start: {delta:+.4f}   "
             f"Checkpoints: {len(vals)}")
    return Panel(t, title="[bold]Val Loss Curve", border_style="green")


def progress_panel(rows):
    if not rows:
        return Panel("[dim]Waiting for training to start...",
                     title="Progress", border_style="blue")
    latest = rows[-1]
    step   = int(latest["step"])
    pct    = step / TOTAL_STEPS
    tokens = int(latest["tokens_seen"])
    tps    = float(latest["tokens_per_sec"])
    mfu    = latest["mfu_pct"]
    elapsed = float(latest["elapsed_s"])

    remaining_tokens = MAX_TOKENS - tokens
    eta_s = remaining_tokens / tps if tps > 0 else 0
    eta_h = eta_s / 3600

    W = 50
    filled = int(pct * W)
    bar = "█" * filled + "░" * (W - filled)
    col = "green" if pct > 0.9 else "cyan"

    t = Text()
    t.append(f"  [{bar}] {pct*100:.1f}%\n", style=col)
    t.append(f"  Step {step:,} / {TOTAL_STEPS:,}   "
             f"Tokens {tokens/1e9:.2f}B / {MAX_TOKENS/1e9:.0f}B\n")
    t.append(f"  {tps:,.0f} tok/s   MFU {mfu}%   "
             f"Elapsed {elapsed/3600:.1f}h   ETA {eta_h:.1f}h")

    ckpts = sorted(CKPT_DIR.glob("step_*.pt")) if CKPT_DIR.exists() else []
    last_ckpt = ckpts[-1].name if ckpts else "none"
    t.append(f"\n  Last checkpoint: {last_ckpt}", style="dim")

    return Panel(t, title="[bold]Progress", border_style=col)


def stats_table(rows):
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold",
              title="[bold]Recent Checkpoints")
    t.add_column("Step",       justify="right", style="cyan",   width=10)
    t.add_column("Tokens",     justify="right", width=10)
    t.add_column("LR",         justify="right", style="yellow", width=10)
    t.add_column("Train Loss", justify="right", width=12)
    t.add_column("Val Loss",   justify="right", style="green",  width=10)
    t.add_column("MFU%",       justify="right", width=7)
    t.add_column("Tok/s",      justify="right", width=10)

    val_rows = [r for r in rows if r.get("val_loss", "nan") != "nan"]
    for r in val_rows[-12:]:
        t.add_row(
            f"{int(r['step']):,}",
            f"{int(r['tokens_seen'])/1e6:.0f}M",
            r["lr"],
            r["train_loss"],
            r["val_loss"],
            r["mfu_pct"],
            f"{int(float(r['tokens_per_sec'])):,}",
        )
    if not val_rows:
        t.add_row("[dim]—", "—", "—", "—", "—", "—", "—")
    return t


def main():
    console = Console()
    console.print("[bold cyan]0.25B training dashboard started. Ctrl+C to exit.[/]")

    with Live(console=console, refresh_per_second=1 / REFRESH, screen=False) as live:
        while True:
            rows = read_rows()
            g    = gpu_stats()
            done = rows and int(rows[-1]["step"]) >= TOTAL_STEPS

            content = Group(
                gpu_panel(g),
                progress_panel(rows),
                lr_curve_panel(rows),
                loss_curve_panel(rows),
                stats_table(rows),
            )
            title = "[bold green]✓ Training complete!" if done \
                else "[bold white]llm_train · 0.25B run"
            live.update(Panel(content, title=title, border_style="bright_blue"))

            if done:
                break
            time.sleep(REFRESH)

    console.print("[bold green]Done.[/]")


if __name__ == "__main__":
    main()
