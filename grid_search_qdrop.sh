#!/bin/bash
# Q-Drop Grid Search — sweeps all Q-Drop hyperparameters on MUTAG & PROTEINS
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${CONDA_PREFIX:-/Users/quangnguyen/miniconda3/envs/Penny2}/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "Activate the Penny2 conda env or set CONDA_PREFIX."
    exit 1
fi

echo "=== Q-Drop Grid Search ==="
echo "Python: $PYTHON"
echo "Datasets: ${*:-mutag proteins}"
echo ""

cd "$SCRIPT_DIR/src"
exec "$PYTHON" grid_search_qdrop.py "$@"
