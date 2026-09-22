#!/usr/bin/env bash
# NCI1: baseline vs QFI, FINAL paper config — must match _train_quantum_one.yml:
#   lr 0.001, q-depths [2,2], AngleEmbedding Y, 10 folds, n_qubits 8.
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
    --folds 10 \
    --lr 0.001 \
    --q-depths 2 2 \
    --embedding-rotation Y \
    --n-qubits "${NQ:-8}" \
    --output-dir "$OUT" \
    --seed 42
done
touch "$OUT/DONE"
echo "=== ALL DONE $(date) ==="
