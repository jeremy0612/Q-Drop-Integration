#!/usr/bin/env bash
# Phase-2: test "qfi >= baseline + delta" on the tuned winning cell.
# Loops dataset x algorithm x seed (one process each -> every run gets a fresh
# seed, avoiding the set_seed-once-per-process drift). Prints a per-dataset
# verdict: baseline vs qfi mean±std, qfi engagement (log10 cond > 0), and whether
# the gap clears the noise.
#
# Local quick:  DATASETS=mutag EPOCHS=30 FOLDS=3 SEEDS="42 123 456" bash scripts/hypothesis_test.sh
# P6000 full:   DATASETS="mutag proteins" EPOCHS=100 FOLDS=10 SEEDS="42 123 456 789 2024" bash scripts/hypothesis_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATASETS="${DATASETS:-mutag}"           # space-separated
EPOCHS="${EPOCHS:-100}"; FOLDS="${FOLDS:-10}"
SEEDS="${SEEDS:-42 123 456}"
NQ="${NQ:-8}"                            # proteins final: also try 16
OUT="${OUTPUT_DIR:-hyp_test}"
# winning cell from phase-1 (edit here if the grid picks differently):
CELL="--lr 0.001 --q-depths 2 2 --embedding-rotation Y --n-qubits $NQ"

for ds in $DATASETS; do
  for algo in baseline qfi; do
    for seed in $SEEDS; do
      echo "=== $ds / $algo / seed=$seed $(date +%H:%M) ==="
      python src/train_quantum_models.py \
        --datasets "$ds" --algorithm "$algo" \
        --epochs "$EPOCHS" --folds "$FOLDS" $CELL --seed "$seed" \
        --output-dir "$OUT/${ds}_${algo}_s${seed}"
    done
  done
done

echo; echo "=== VERDICT ==="
python - "$OUT" <<'PY'
import json, glob, sys, statistics as st
from collections import defaultdict
acc=defaultdict(list); cond=defaultdict(list)
for f in glob.glob(f"{sys.argv[1]}/*/quantum_graph_training_*/summary.json"):
    parts=f.split("/")[-3].split("_")           # <ds>_<algo>_s<seed> dir (abs/rel safe)
    ds, algo = parts[0], parts[1]
    for _, e in json.load(open(f)).items():
        s=e.get("summary",{})
        if s.get("mean_accuracy") is not None: acc[(ds,algo)].append(s["mean_accuracy"])
        for fold in e.get("folds",[]):
            for pt in (fold.get("qdrop_curve") or []):
                if "qfi_log10_condition" in pt: cond[(ds,algo)].append(pt["qfi_log10_condition"])
def ms(v): return (st.mean(v), (st.pstdev(v) if len(v)>1 else 0.0)) if v else (float("nan"),0.0)
for ds in sorted({k[0] for k in acc}):
    bm,bs=ms(acc[(ds,"baseline")]); qm,qs=ms(acc[(ds,"qfi")])
    c=cond[(ds,"qfi")]; cmax=max(c) if c else 0.0
    delta=qm-bm; noise=(bs**2+qs**2)**0.5
    engaged = cmax>1e-6
    if not engaged: verdict="INVALID — qfi never engaged (log10cond~0); check ansatz"
    elif delta>noise: verdict=f"qfi WINS +{delta:.4f} (clears noise {noise:.4f})"
    elif delta>0: verdict=f"qfi ahead +{delta:.4f} but WITHIN noise {noise:.4f} — inconclusive, add seeds"
    else: verdict=f"qfi LOSES {delta:.4f}"
    print(f"[{ds}] baseline {bm:.4f}±{bs:.3f} | qfi {qm:.4f}±{qs:.3f} | qfi_cond_max={cmax:.2f} | {verdict}")
PY
