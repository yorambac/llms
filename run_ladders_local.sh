#!/usr/bin/env bash
# LR ladder sweep on local RTX 4070, tuned for batch=4 (MFU-optimal config).
# 4 LRs centered around sqrt-scaled prediction from H100 winner (1.9e-3 @ b=40 → ~6e-4 @ b=4).
# 200M tokens per run (~3.9h each, ~16h total).
# Results: results/ladder_local_results.csv

set -e
cd "$(dirname "$0")"

PYTHON=/home/yoram/miniconda3/envs/llm_train/bin/python
BATCH=4
CTX=1024
MAX_TOKENS=200000000
RESULTS=results/ladder_local_results.csv
CKPT_BASE=checkpoints/ladder_local

export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torch/inductor/llm_train"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Local RTX 4070 LR ladder sweep ==="
echo "    batch=$BATCH  ctx=$CTX  tokens_per_run=$MAX_TOKENS"
echo "    LRs: [2e-4, 5e-4, 1e-3, 2e-3]"
echo "    sqrt-scale from H100 (1.9e-3 @ b=40) predicts ~6e-4 @ b=4"
echo ""

rm -f "$RESULTS"

for LR in 2e-4 5e-4 1e-3 2e-3; do
    RUN_NAME="lr${LR}"
    CKPT_DIR="${CKPT_BASE}/${RUN_NAME}"
    echo "--- $RUN_NAME ---"
    rm -rf "$CKPT_DIR"   # always start fresh, never resume
    $PYTHON -u train_500m.py \
        --lr          "$LR" \
        --batch_size  "$BATCH" \
        --max_tokens  "$MAX_TOKENS" \
        --ckpt_dir    "$CKPT_DIR" \
        --results_file "$RESULTS"
    echo ""
done

echo "=== All runs complete ==="
$PYTHON - <<'EOF'
import csv
rows = list(csv.DictReader(open("results/ladder_local_results.csv")))
final = {}
for r in rows:
    final[r["run_name"]] = r
print(f"\n{'Run':<12} {'LR':<8} {'Val Loss':>9} {'Tok/s':>9} {'MFU%':>6}")
print("─" * 50)
best_run = min(final.values(), key=lambda r: float(r["val_loss"]))
for name in sorted(final):
    r = final[name]
    marker = " ◀ best" if name == best_run["run_name"] else ""
    print(f"{r['run_name']:<12} {r['lr']:<8} {float(r['val_loss']):>9.4f} "
          f"{r['tokens_per_sec']:>9} {r['mfu_pct']:>6}{marker}")
EOF
