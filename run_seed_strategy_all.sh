#!/bin/bash
# Multi-seed seed-strategy driver: 6 variants, GPU 1.
#
# Methodology: run each (variant, light-dataset) over 5 seeds and report
# mean +/- std across seeds (see aggregate_seeds.py). nci1 is heavy so it runs
# single-seed (42). One dataset per python process (VRAM released between
# runs); failures isolated; resume skips completed (metrics.json) runs.
#
# Entry points (IMPORTANT):
#   QGCN variants -> train_quantum_models.py (--model-type qgcn)
#   QGAT variants -> train_qgat.py  (tuned QGAT defaults: n_qubits=8,
#                    q_depths=[2,2], multiscale pool, MLP head, residual,
#                    lr=5e-4, attn_dropout=0.2)
#
# Qubit policy: QGCN on the datasets that default to 16 qubits (proteins,
# imdb_binary, imdb_multi) is pinned to 8 qubits + batch 8. QGAT already
# defaults to 8 qubits. nci1/mutag keep their defaults.
#
# Variants (name | algorithm | entry):
#   qgcn_baseline | baseline | qgcn   (QGCN no reg)
#   qgcn_dropout  | dropout  | qgcn   (QGCN + Q-Drop)
#   qgcn_pruning  | pruning  | qgcn   (QGCN + SP)
#   qgcn_both     | both     | qgcn   (QGCN + Q-Drop + SP)
#   qgat_baseline | baseline | qgat   (QGAT no reg)
#   qgat_both     | both     | qgat   (QGAT + Q-Drop + SP)
set -uo pipefail

PROJECT_ROOT="/home/cislab301b/Khanh/Q-Drop-Integration"
SRC_DIR="$PROJECT_ROOT/src"
RESULTS_ROOT="$PROJECT_ROOT/training_results/seed_strategy"
LOG_DIR="$RESULTS_ROOT/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEEDS=(42 123 456 789 2024)        # light datasets: multi-seed
SEEDS_NCI1=(42)                    # nci1: single seed (too heavy x5)

# dataset run order (nci1 last)
DATASETS=(mutag proteins imdb_binary imdb_multi nci1)

# QGCN-only extra flags for the datasets that otherwise default to 16 qubits.
declare -A QGCN_EXTRA=(
  [proteins]="--n-qubits 8 --batch-size 8"
  [imdb_binary]="--n-qubits 8 --batch-size 8"
  [imdb_multi]="--n-qubits 8 --batch-size 8"
)

# variant_name|algorithm|entry(qgcn|qgat)
VARIANTS=(
  "qgcn_baseline|baseline|qgcn"
  "qgcn_dropout|dropout|qgcn"
  "qgcn_pruning|pruning|qgcn"
  "qgcn_both|both|qgcn"
  "qgat_baseline|baseline|qgat"
  "qgat_both|both|qgat"
)

cd "$SRC_DIR" || exit 1
echo "=== Multi-seed seed-strategy sweep started: $(date) ==="
echo "GPU=$CUDA_VISIBLE_DEVICES | seeds(light)=${SEEDS[*]} | seeds(nci1)=${SEEDS_NCI1[*]}"
echo "Dataset order: ${DATASETS[*]}"
echo ""

for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r name algo kind <<< "$entry"
  for ds in "${DATASETS[@]}"; do
    # choose seed list + entry script + extra flags
    if [[ "$ds" == "nci1" ]]; then
      seed_list=("${SEEDS_NCI1[@]}")
    else
      seed_list=("${SEEDS[@]}")
    fi

    for seed in "${seed_list[@]}"; do
      out_dir="$RESULTS_ROOT/$name/$ds/seed${seed}"
      log_file="$LOG_DIR/${name}__${ds}__seed${seed}.log"

      if find "$out_dir" -name metrics.json 2>/dev/null | grep -q .; then
        echo ">>> [$(date +%H:%M:%S)] SKIP $name / $ds / seed$seed (done)"
        continue
      fi

      if [[ "$kind" == "qgat" ]]; then
        script="train_qgat.py"
        extra=""                      # QGAT defaults already 8 qubits
      else
        script="train_quantum_models.py"
        extra="${QGCN_EXTRA[$ds]:-}"
      fi

      echo ">>> [$(date +%H:%M:%S)] $name / $ds / seed$seed ($script ${extra:-default})"
      python "$script" \
          --datasets "$ds" \
          --algorithm "$algo" \
          --seed "$seed" \
          ${extra} \
          --output-dir "$out_dir" \
          > "$log_file" 2>&1
      status=$?
      if [ $status -eq 0 ]; then
        acc=$(grep -aoE "results: acc=[0-9.]+" "$log_file" | grep -oE "[0-9.]+" | head -1)
        echo "    [OK]   $name / $ds / seed$seed  acc=${acc:-?}"
      else
        echo "    [FAIL] $name / $ds / seed$seed (status $status) -> $log_file"
      fi
    done
  done
done

echo ""
echo "=== Sweep finished: $(date) ==="
echo "Aggregate with: python $PROJECT_ROOT/aggregate_seeds.py"
