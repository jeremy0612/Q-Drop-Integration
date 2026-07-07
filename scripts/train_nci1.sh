#!/usr/bin/env bash
# NCI1: baseline vs QFI, identical tuned hyperparams (bug-028 config).
# Two separate processes -> each arm gets a fresh seed 42.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${OUTPUT_DIR:-ci_results}"
for algo in baseline qfi; do
  echo "=== NCI1 / $algo $(date) ==="
  python src/train_quantum_models.py \
    --datasets nci1 \
    --algorithm "$algo" \
    --epochs "${EPOCHS:-100}" \
    --lr 0.0005 \
    --use-strongly-entangling \
    --output-dir "$OUT" \
    --seed 42
done
touch "$OUT/DONE"
echo "=== ALL DONE $(date) ==="
