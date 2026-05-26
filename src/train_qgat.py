"""Train QGAT (Quantum Graph Attention Network) on IMDB-BINARY, IMDB-MULTI, and NCI1.

Ported from Quantum_Graph_Attention_Network and integrated into the Q-Drop-Integration
training framework for unified data loading, 10-fold CV, and metrics logging.

Key improvements over the original repo:
  - default.qubit + backprop (GPU-native via PyTorch autograd) instead of
    lightning.gpu + adjoint (crashes on cuStateVec < 0.6 / CUDA 12.8).
  - Multi-scale pooling (mean + max + add concatenated) for better graph
    classification signal.
  - MLP head with BatchNorm + Dropout instead of single Linear.
  - Residual connections + LayerNorm inside each QGATConv layer.
  - 10-fold stratified cross-validation, early stopping, OneCycleLR.
"""

import os
import sys

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from training.graph_training import build_train_parser, config_from_args, run_experiments


def _parse_args():
    parser = build_train_parser(
        description="Train QGAT on IMDB-BINARY, IMDB-MULTI, NCI1",
        default_datasets=["imdb_binary", "imdb_multi", "nci1"],
        default_batch_size=32,
        default_weight_decay=1e-3,
        default_use_scheduler=True,
        default_use_class_weights=True,
    )

    # QGAT-specific defaults — user can still override all of these via CLI.
    parser.set_defaults(
        model_type="qgat",
        n_qubits=8,
        q_depths=[2, 2],
        epochs=100,
        folds=10,
        early_stop_patience=15,
        pool_type="multiscale",
        use_mlp_head=True,
        mlp_hidden=64,
        mlp_dropout=0.3,
        use_residual=True,
        attn_dropout=0.2,
        lr=5e-4,
        val_frequency=5,
    )

    return config_from_args(parser.parse_args())


def main() -> None:
    config = _parse_args()
    run_experiments(config)


if __name__ == "__main__":
    main()
