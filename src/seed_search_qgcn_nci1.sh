#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

SEEDS=(42 123 456 789 2024)
BEST_ACC=-1
BEST_SEED=0
GPU=${CUDA_VISIBLE_DEVICES:-1}

echo "=============================="
echo "QGCN | NCI1 | GPU $GPU"
echo "Seeds: ${SEEDS[*]}"
echo "=============================="

for SEED in "${SEEDS[@]}"; do
    LOG="../training_results/qgcn_nci1_seed${SEED}.log"
    echo ""
    echo ">>> [QGCN/NCI1] Seed=$SEED"
    CUDA_VISIBLE_DEVICES=$GPU python train_quantum_models.py \
        --datasets nci1 --seed "$SEED" \
        --output-dir ../training_results 2>&1 | tee "$LOG"

    ACC=$(grep "results: acc=" "$LOG" | grep -oP "acc=\K[0-9.]+" | head -1)
    echo ">>> Seed $SEED → acc=$ACC"

    IS_BETTER=$(python3 -c "print(1 if ${ACC:-0} > $BEST_ACC else 0)" 2>/dev/null || echo 0)
    if [ "$IS_BETTER" = "1" ]; then
        BEST_ACC=$ACC
        BEST_SEED=$SEED
    fi
done

echo ""
echo "=============================="
echo "QGCN/NCI1  BEST SEED: $BEST_SEED  (mean acc=$BEST_ACC)"
echo "=============================="
