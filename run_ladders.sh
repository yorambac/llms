#!/usr/bin/env bash
# Run 4 LR ladder experiments sequentially at 0.01B scale.
# Each run: 200M tokens, ~6-10 min on RTX 4070.
# Results appended to results/ladder_results.csv

set -e
PYTHON=/home/yoram/miniconda3/envs/llm_train/bin/python
cd "$(dirname "$0")"

# Cache compiled kernels across runs — only compiles once per machine
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torch/inductor/llm_train"

echo "=== Preparing data ==="
$PYTHON prepare_data.py

echo "=== Ladder sweep: lr in [1e-3, 3e-3, 6e-3, 1e-2] ==="

for LR in 1e-3 3e-3 6e-3 1e-2; do
    RUN_NAME="lr${LR}"
    echo ""
    echo "--- Starting run: $RUN_NAME ---"
    $PYTHON train.py \
        --lr "$LR" \
        --run_name "$RUN_NAME" \
        --batch_size 32 \
        --block_size 512 \
        --max_tokens 200000000
done

echo ""
echo "=== All ladder runs complete ==="
echo "Results in results/ladder_results.csv"
$PYTHON -c "
import csv
rows = list(csv.DictReader(open('results/ladder_results.csv')))
final = {}
for r in rows:
    final[r['run_name']] = r
print(f'{'Run':<12} {'LR':<8} {'Val Loss':<10} {'Tok/s':<10} {'MFU%':<8}')
print('-' * 50)
for name, r in sorted(final.items()):
    print(f\"{r['run_name']:<12} {r['lr']:<8} {r['val_loss']:<10} {r['tokens_per_sec']:<10} {r['mfu_pct']:<8}\")
"
