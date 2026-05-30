"""
Live ladder training dashboard.
Run in a separate terminal: python dashboard.py

Reads results/ladder_results.csv and nvidia-smi every 2s.
"""

import csv
import subprocess
import time
from pathlib import Path
from collections import defaultdict

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box

RESULTS_FILE = Path("results/ladder_results.csv")
ALL_RUNS = ["lr1e-3", "lr3e-3", "lr6e-3", "lr1e-2"]
TOTAL_STEPS = 12207
REFRESH = 2  # seconds


def read_results():
    if not RESULTS_FILE.exists():
        return {}
    rows = defaultdict(list)
    try:
        with open(RESULTS_FILE) as f:
            for row in csv.DictReader(f):
                rows[row["run_name"]].append(row)
    except Exception:
        pass
    return rows


def gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        ).strip()
        temp, util, pwr, mem_used, mem_total = out.split(", ")
        return {
            "temp": int(temp), "util": int(util),
            "power": float(pwr),
            "mem_used": int(mem_used), "mem_total": int(mem_total),
        }
    except Exception:
        return None


def make_gpu_panel(g):
    if g is None:
        return Panel("GPU unavailable", title="GPU")
    bar_len = 20
    util_bar = "█" * int(g["util"] / 100 * bar_len) + "░" * (bar_len - int(g["util"] / 100 * bar_len))
    mem_pct  = g["mem_used"] / g["mem_total"]
    mem_bar  = "█" * int(mem_pct * bar_len) + "░" * (bar_len - int(mem_pct * bar_len))
    color = "green" if g["util"] > 80 else "yellow" if g["util"] > 40 else "red"
    text = Text()
    text.append(f"  Util  [{util_bar}] {g['util']:3d}%\n", style=color)
    text.append(f"  VRAM  [{mem_bar}] {g['mem_used']:,}/{g['mem_total']:,} MiB\n")
    text.append(f"  Temp  {g['temp']}°C    Power {g['power']:.0f}W / 200W")
    return Panel(text, title="[bold]RTX 4070", border_style=color)


def make_progress_table(rows):
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta",
              title="[bold]Ladder Sweep  (0.01B · 200M tokens · LR search)")
    t.add_column("Run",        style="cyan",   width=10)
    t.add_column("LR",         style="yellow", width=8)
    t.add_column("Progress",   width=24)
    t.add_column("Val Loss",   justify="right", style="green", width=10)
    t.add_column("Δ Loss",     justify="right", width=8)
    t.add_column("Tok/s",      justify="right", width=10)
    t.add_column("MFU%",       justify="right", width=7)
    t.add_column("Status",     width=12)

    lr_map = {"lr1e-3": "1e-3", "lr3e-3": "3e-3", "lr6e-3": "6e-3", "lr1e-2": "1e-2"}

    for run in ALL_RUNS:
        run_rows = rows.get(run, [])
        if not run_rows:
            t.add_row(run, lr_map[run], "─" * 20, "—", "—", "—", "—", "[dim]waiting")
            continue

        latest = run_rows[-1]
        step = int(latest["step"])
        done = step >= TOTAL_STEPS - 1

        pct = step / TOTAL_STEPS
        bar_fill = int(pct * 20)
        bar = "█" * bar_fill + "░" * (20 - bar_fill)
        prog_text = Text()
        prog_text.append(bar, style="green" if done else "cyan")
        prog_text.append(f" {pct*100:.0f}%")

        val_loss = float(latest["val_loss"])
        first_val = float(run_rows[1]["val_loss"]) if len(run_rows) > 1 else val_loss
        delta = val_loss - first_val
        delta_str = f"{delta:+.3f}"
        delta_style = "green" if delta < 0 else "red"

        tps = f"{int(latest['tokens_per_sec']):,}"
        mfu = latest["mfu_pct"]
        status = "[bold green]done" if done else "[cyan]running"

        t.add_row(
            run, lr_map[run], prog_text,
            f"{val_loss:.4f}", Text(delta_str, style=delta_style),
            tps, mfu, status,
        )

    return t


def make_loss_sparkline(rows, run):
    """Simple ASCII sparkline of val loss over time."""
    run_rows = rows.get(run, [])
    vals = [float(r["val_loss"]) for r in run_rows if r["val_loss"] != "nan"]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return "─" * len(vals)
    chars = " ▁▂▃▄▅▆▇█"
    spark = ""
    for v in vals[-40:]:
        idx = int((v - lo) / (hi - lo) * (len(chars) - 1))
        spark += chars[len(chars) - 1 - idx]
    return spark


def make_loss_panel(rows):
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("run", style="cyan", width=8)
    t.add_column("curve", width=42)
    t.add_column("final", justify="right", width=7)

    for run in ALL_RUNS:
        run_rows = rows.get(run, [])
        spark = make_loss_sparkline(rows, run)
        final = f"{float(run_rows[-1]['val_loss']):.4f}" if run_rows else "—"
        t.add_row(run, spark, final)

    return Panel(t, title="[bold]Val Loss Curves (recent 40 checkpoints, high→low = top→bottom)")


def build_layout(rows, g):
    return Columns([
        Panel(
            Columns([make_gpu_panel(g)], equal=True),
            box=box.SIMPLE, expand=False,
        ),
        make_progress_table(rows),
    ], expand=True)


def main():
    console = Console()
    with Live(console=console, refresh_per_second=1 / REFRESH, screen=False) as live:
        while True:
            rows = read_results()
            g = gpu_stats()

            from rich.console import Group
            content = Group(
                make_gpu_panel(g),
                make_progress_table(rows),
                make_loss_panel(rows),
            )
            live.update(Panel(content, title="[bold white]llm_train ladder dashboard",
                              border_style="bright_blue"))

            # Stop when all runs done
            all_done = all(
                any(int(r["step"]) >= TOTAL_STEPS - 1 for r in rows.get(run, []))
                for run in ALL_RUNS
            )
            if all_done:
                live.update(Panel(content, title="[bold green]✓ All runs complete!",
                                  border_style="green"))
                break

            time.sleep(REFRESH)


if __name__ == "__main__":
    main()
