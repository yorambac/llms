#!/usr/bin/env bash
# LR ladder sweep on H200, tuned for batch=48 (MFU-optimal config).
#
# Changes vs original run_ladders.sh:
#   - 5 LRs centered around 1.3e-3 (linear scale from batch=32 winner lr=1e-3)
#   - 200M tokens per run (~1.25 min each on H200, same as original local ladder)
#   - batch=48, ctx=1024 (MFU-optimal from H200 fine-sweep)
#   - Results in results/ladder_h200_results.csv

set -e
cd "$(dirname "$0")"

PYTHON=python
BATCH=${BATCH:-48}
CTX=1024
MAX_TOKENS=200_000_000     # 200M tokens per run (~1.25 min on H200)
RESULTS=results/ladder_h200_results.csv

export TORCHINDUCTOR_FX_GRAPH_CACHE=1

echo "=== H200 LR ladder sweep ==="
echo "    batch=$BATCH  ctx=$CTX  tokens_per_run=$MAX_TOKENS"
echo "    LRs: [7e-4, 1e-3, 1.3e-3, 1.7e-3, 2.3e-3]"
echo "    (centered around 1.3e-3 = lr=1e-3 × 48/32 linear scaling from batch=32 winner)"
echo ""

# Wipe old results so we get a clean CSV
rm -f "$RESULTS"

for LR in 7e-4 1e-3 1.3e-3 1.7e-3 2.3e-3; do
    RUN_NAME="lr${LR}"
    echo "--- $RUN_NAME ---"
    rm -rf "checkpoints/${RUN_NAME}"   # always start fresh, never resume
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
