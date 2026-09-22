#!/usr/bin/env bash
# Phase-1 hyperparam grid: tune the BASELINE honestly per ansatz; QFI is tested
# afterwards only on the winning cell(s) (strongest-baseline principle, bug-028).
#
# Local quick pass:   EPOCHS=30 FOLDS=3 bash scripts/grid_search_phase1.sh
# Full P6000 pass:    bash scripts/grid_search_phase1.sh          (100 ep, 10 folds)
# Include QFI arm:    ALGOS="baseline qfi" bash scripts/grid_search_phase1.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-100}"
FOLDS="${FOLDS:-10}"
DATASETS="${DATASETS:-mutag}"
ALGOS="${ALGOS:-baseline}"
OUT="${OUTPUT_DIR:-grid_phase1}"

# Cells: ansatz x lr x depth. basic-X = historical reference (QFI dead there);
# basic-Y / strongly-entangling = the two degeneracy-breakers.
# ponytail: plain nested loops, no configs/yaml — edit the lists to change the grid.
for ansatz in basicX basicY strong; do
  case $ansatz in
    basicX) FLAGS="--embedding-rotation X" ;;
    basicY) FLAGS="--embedding-rotation Y" ;;
    strong) FLAGS="--use-strongly-entangling" ;;
  esac
  for lr in 0.0005 0.001; do
    for depth in "1 1" "2 2"; do
      for algo in $ALGOS; do
        tag="${ansatz}_lr${lr}_d${depth// /}_${algo}"
        echo "=== $tag $(date +%H:%M) ==="
        python src/train_quantum_models.py \
          --datasets $DATASETS \
          --algorithm "$algo" \
          --epochs "$EPOCHS" --folds "$FOLDS" \
          --lr "$lr" --q-depths $depth \
          $FLAGS \
          --output-dir "$OUT/$tag" \
          --seed 42
      done
    done
  done
done

echo; echo "=== RESULTS (sorted by accuracy) ==="
python - "$OUT" <<'PY'
import json, glob, sys
rows = []
for f in glob.glob(f"{sys.argv[1]}/*/quantum_graph_training_*/summary.json"):
    tag = f.split("/")[1]
    for ds, e in json.load(open(f)).items():
        s = e.get("summary", {})
        if s.get("mean_accuracy") is not None:
            rows.append((s["mean_accuracy"], s.get("std_accuracy", 0), ds, tag))
for acc, std, ds, tag in sorted(rows, reverse=True):
    print(f"{acc:.4f} ±{std:.3f}  {ds:10} {tag}")
PY
