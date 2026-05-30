"""
Live dashboard for profile_mfu.py sweep.
Run in a separate terminal: python profile_dashboard.py
"""

import csv
import subprocess
import time
from pathlib import Path
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Console, Group
from rich.text import Text
from rich import box

RESULTS_FILE = Path("results/mfu_profile.csv")
REFRESH = 2

ARCHS       = ["1024-16L", "896-20L"]
BATCH_SIZES = [4, 8, 16, 32]
BLOCK_SIZES = [512, 1024]
TOTAL_CONFIGS = len(ARCHS) * len(BATCH_SIZES) * len(BLOCK_SIZES)


def read_results():
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


def gpu_panel(g):
    if not g:
        return Panel("unavailable", title="GPU")
    W = 24
    u = int(g["util"] / 100 * W)
    m = int(g["mem_used"] / g["mem_total"] * W)
    col = "green" if g["util"] > 80 else "yellow" if g["util"] > 30 else "red"
    t = Text()
    t.append(f"  Util  [{'█'*u}{'░'*(W-u)}] {g['util']:3d}%\n", style=col)
    t.append(f"  VRAM  [{'█'*m}{'░'*(W-m)}] {g['mem_used']:,}/{g['mem_total']:,} MiB\n")
    t.append(f"  Temp  {g['temp']}°C    Power {g['power']:.0f} / 200 W")
    return Panel(t, title="[bold]RTX 4070", border_style=col)


def results_table(rows):
    t = Table(
        box=box.ROUNDED, show_header=True, header_style="bold magenta",
        title=f"[bold]MFU Profile Sweep — 0.25B  ({len(rows)}/{TOTAL_CONFIGS} configs done)",
    )
    t.add_column("Arch",    style="cyan",   width=12)
    t.add_column("Params",  justify="right",width=8)
    t.add_column("Batch",   justify="right",width=6)
    t.add_column("Ctx",     justify="right",width=6)
    t.add_column("Tok/s",   justify="right",width=12)
    t.add_column("MFU%",    justify="right",width=7)
    t.add_column("VRAM MB", justify="right",width=9)
    t.add_column("",        width=6)

    if not rows:
        t.add_row("[dim]waiting for first result...", "", "", "", "", "", "", "")
        return t

    best_tps = max((float(r["tokens_per_sec"]) for r in rows if r["tokens_per_sec"] != "OOM"), default=0)

    for r in rows:
        oom = r["tokens_per_sec"] == "OOM"
        tps_str = "[red]OOM" if oom else f"{int(float(r['tokens_per_sec'])):,}"
        mfu_str = "[red]OOM" if oom else r["mfu_pct"]
        vram_str = "[red]OOM" if oom else r["vram_mb"]
        is_best = not oom and float(r["tokens_per_sec"]) == best_tps
        marker = "[bold green]★ best" if is_best else ""
        mfu_style = "green" if not oom and float(r["mfu_pct"]) >= 35 else \
                    "yellow" if not oom and float(r["mfu_pct"]) >= 25 else "white"
        t.add_row(
            r["arch"], f"{r['params_m']}M",
            r["batch_size"], r["block_size"],
            tps_str,
            Text(mfu_str, style=mfu_style),
            vram_str, marker,
        )

    return t


def progress_panel(rows):
    done = len(rows)
    pct = done / TOTAL_CONFIGS
    W = 40
    bar = "█" * int(pct * W) + "░" * (W - int(pct * W))
    col = "green" if pct == 1 else "cyan"
    t = Text()
    t.append(f"  [{bar}] {done}/{TOTAL_CONFIGS} configs  ({pct*100:.0f}%)\n", style=col)
    if rows:
        best = max((r for r in rows if r["tokens_per_sec"] != "OOM"),
                   key=lambda r: float(r["tokens_per_sec"]), default=None)
        if best:
            t.append(f"  Best so far: {best['arch']}  batch={best['batch_size']}  "
                     f"ctx={best['block_size']}  →  "
                     f"{int(float(best['tokens_per_sec'])):,} tok/s  "
                     f"MFU={best['mfu_pct']}%", style="bold green")
    return Panel(t, title="[bold]Progress", border_style="blue")


def main():
    console = Console()
    console.print("[bold cyan]MFU profile dashboard started. Waiting for results...[/]")

    with Live(console=console, refresh_per_second=1 / REFRESH, screen=False) as live:
        while True:
            rows = read_results()
            g = gpu_stats()
            content = Group(gpu_panel(g), progress_panel(rows), results_table(rows))
            done = len(rows) >= TOTAL_CONFIGS
            title = "[bold green]✓ Profile complete!" if done else "[bold white]llm_train · MFU profiler"
            live.update(Panel(content, title=title, border_style="bright_blue"))
            if done:
                break
            time.sleep(REFRESH)

    console.print("\n[bold green]All configs profiled. Check results/mfu_profile.csv[/]")


if __name__ == "__main__":
    main()
