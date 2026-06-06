#!/usr/bin/env bash
# LR ladder sweep on H200, optimised for the 0.45B training config.
#
# Changes vs original run_ladders.sh:
#   - 7 LR values (wider + finer sweep around expected optimal)
#   - 1B tokens per run (5× more than original 200M → cleaner signal)
#   - batch=64, ctx=1024 (set after MFU profiling — change if profiler says different)
#   - Min LR = max_lr / 10 (cosine decay, same ratio as full run)
#   - Results in results/ladder_h200_results.csv
#
# Usage:
#   bash run_ladders_h200.sh              # uses default BATCH=64
#   BATCH=128 bash run_ladders_h200.sh   # override batch after profiling

set -e
cd "$(dirname "$0")"

PYTHON=python
BATCH=${BATCH:-64}
CTX=1024
MAX_TOKENS=1_000_000_000   # 1B tokens per run
RESULTS=results/ladder_h200_results.csv

export TORCHINDUCTOR_FX_GRAPH_CACHE=1

echo "=== H200 LR ladder sweep ==="
echo "    batch=$BATCH  ctx=$CTX  tokens_per_run=$MAX_TOKENS"
echo "    LRs: [3e-4, 5e-4, 8e-4, 1e-3, 1.5e-3, 2e-3, 3e-3]"
echo ""

# Wipe old results so we get a clean CSV
rm -f "$RESULTS"

for LR in 3e-4 5e-4 8e-4 1e-3 1.5e-3 2e-3 3e-3; do
    RUN_NAME="lr${LR}"
    echo "--- $RUN_NAME ---"
    $PYTHON train.py \
        --lr          "$LR" \
        --run_name    "$RUN_NAME" \
        --batch_size  "$BATCH" \
        --block_size  "$CTX" \
        --max_tokens  "$MAX_TOKENS" \
        --results_file "$RESULTS"
    echo ""
done

echo "=== All runs complete ==="
$PYTHON - <<'EOF'
import csv
rows = list(csv.DictReader(open("results/ladder_h200_results.csv")))
# Last row per run = final metrics
final = {}
for r in rows:
    final[r["run_name"]] = r
print(f"\n{'Run':<12} {'LR':<8} {'Best val':>9} {'Final val':>10} {'Tok/s':>9} {'MFU%':>6}")
print("─" * 58)
best_run = min(final.values(), key=lambda r: float(r["val_loss"]))
for name in sorted(final):
    r = final[name]
    marker = " ◀ best" if name == best_run["run_name"] else ""
    print(f"{r['run_name']:<12} {r['lr']:<8} {float(r['val_loss']):>9.4f} {float(r['val_loss']):>10.4f} "
          f"{r['tokens_per_sec']:>9} {r['mfu_pct']:>6}{marker}")
EOF
