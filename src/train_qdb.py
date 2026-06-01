"""CLI entry point for QDB (Quantum-Diffusion-Block) training.

Mirrors `train_quantum_models.py` but trains the QDB model via block-wise
score-matching instead of end-to-end backprop. Does NOT support the
`baseline` algorithm — QDB always uses Q-Drop pruning/dropout/both per
plan decision (no baseline CI line).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Cap CPU thread pools BEFORE importing torch/numpy/pennylane. PennyLane,
# numpy/scipy (via OpenMP/MKL) and PyTorch each spawn their own thread pool
# of size=nproc by default. Without caps a 16-qubit training run on a
# 32-core runner produces 100+ threads fighting for the GIL, near-zero
# parallelism, and 100% CPU but no forward progress.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from qdb.models import QDBGCN, QDBGAT
from qdb.training import QDBTrainConfig, evaluate, train_qdb
from training.graph_training import (
    DATASET_SPECS,
    DATASET_NQUBITS_OVERRIDES,
    load_dataset_by_name,
    normalize_dataset_name,
    select_cuda_device,
    set_seed,
)


def make_qgcn_conv_factory(n_layers: int = 1):
    """Return a callable (in_channels, n_qubits) -> QGCNConv-wrapped conv."""
    from models.gcn_conv_layers import QGCNConv

    def _factory(in_channels: int, n_qubits: int):
        return QGCNConv(in_channels, n_layers=n_layers, n_qubits=n_qubits)

    return _factory


def make_qgat_conv_factory(n_layers: int = 1, attn_qubits: int = 4, attn_dropout: float = 0.2):
    """Return a callable (in_channels, n_qubits) -> QGATConv."""
    from models.gat_conv_layers.qgat_conv import QGATConv

    def _factory(in_channels: int, n_qubits: int):
        return QGATConv(
            in_channels=in_channels,
            n_qubits=n_qubits,
            n_layers=n_layers,
            attn_qubits=attn_qubits,
            attn_dropout=attn_dropout,
        )

    return _factory


def build_model(model_type: str, n_blocks: int, in_channels: int, n_qubits: int, n_classes: int) -> QDBGCN:
    if model_type == "qgcn":
        return QDBGCN(
            n_blocks=n_blocks,
            in_channels=in_channels,
            n_qubits=n_qubits,
            n_classes=n_classes,
            conv_factory=make_qgcn_conv_factory(n_layers=1),
        )
    if model_type == "qgat":
        return QDBGAT(
            n_blocks=n_blocks,
            in_channels=in_channels,
            n_qubits=n_qubits,
            n_classes=n_classes,
            conv_factory=make_qgat_conv_factory(),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def resolve_n_qubits(dataset_key: str, override: int | None) -> int:
    if override:
        return override
    return DATASET_NQUBITS_OVERRIDES.get(dataset_key, 8)


def train_dataset(
    dataset_name: str,
    config: QDBTrainConfig,
    model_type: str,
    device: torch.device,
    output_dir: Path,
) -> Dict:
    dataset_key = normalize_dataset_name(dataset_name)
    spec = DATASET_SPECS[dataset_key]
    graphs = load_dataset_by_name(dataset_key)
    labels = [int(g.y.item()) for g in graphs]
    in_channels = graphs[0].x.size(1)
    n_qubits = resolve_n_qubits(dataset_key, config.n_qubits)
    n_classes = spec.n_classes

    print(f"\n=== QDB training on {spec.name} ===")
    print(f"  graphs={len(graphs)}, in_channels={in_channels}, n_qubits={n_qubits}, classes={n_classes}")
    print(f"  n_blocks={config.n_blocks}, batch_size={config.batch_size}, epochs={config.epochs}")

    splitter = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    fold_results: List[Dict] = []

    dataset_dir = output_dir / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(graphs, labels)):
        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]
        train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=config.batch_size, shuffle=False)

        model = build_model(model_type, config.n_blocks, in_channels, n_qubits, n_classes).to(device)

        # Incremental per-epoch checkpoint so operators see progress before
        # all folds complete. Without this, a 100-epoch x 10-fold run on a
        # 16-qubit circuit emits zero artifacts for hours.
        epoch_checkpoint_path = dataset_dir / f"fold_{fold_idx + 1}_epoch_progress.json"

        def _on_epoch(epoch: int, hist: Dict) -> None:
            with open(epoch_checkpoint_path, "w") as fh:
                json.dump(
                    {
                        "dataset": spec.name,
                        "fold": fold_idx + 1,
                        "epoch": epoch,
                        "block_loss": hist["block_loss"],
                        "val_accuracy_curve": hist["val_accuracy"],
                    },
                    fh,
                    indent=2,
                )

        history = train_qdb(
            model,
            train_loader,
            test_loader,
            config,
            device,
            epoch_callback=_on_epoch,
            desc=f"{spec.name}/f{fold_idx + 1}",
        )
        final = evaluate(model, test_loader, device)
        print(f"  fold {fold_idx + 1}: accuracy={final['accuracy']:.4f}", flush=True)

        # Build report-script-compatible curves. The existing
        # scripts/generate_cml_report.py expects train_curve / val_curve
        # as lists of {epoch, loss, accuracy, f1, ...} dicts. QDB stores
        # block_loss[b][epoch] and val_accuracy[epoch]. Synthesize the
        # required schema by averaging block losses per epoch.
        n_epochs = len(history["block_loss"][0]) if history["block_loss"] else 0
        train_curve = []
        for ep in range(n_epochs):
            losses_at_ep = [
                history["block_loss"][b][ep]
                for b in range(model.n_blocks)
                if not np.isnan(history["block_loss"][b][ep])
            ]
            avg_loss = float(np.mean(losses_at_ep)) if losses_at_ep else 0.0
            train_curve.append({"epoch": ep + 1, "loss": avg_loss, "accuracy": 0.0, "f1": 0.0})
        val_curve = [
            {"epoch": ep + 1, "loss": 0.0, "accuracy": acc, "f1": 0.0}
            for ep, acc in enumerate(history["val_accuracy"])
        ]

        fold_result = {
            "fold": fold_idx + 1,
            "accuracy": final["accuracy"],
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "test_loss": 0.0,
            "train_curve": train_curve,
            "val_curve": val_curve,
            "qdrop_curve": [],
            "block_loss": history["block_loss"],
            "val_accuracy_curve": history["val_accuracy"],
        }
        fold_results.append(fold_result)

        # Persist after every fold — operators can inspect partial progress
        # without waiting for all 10 folds.
        with open(dataset_dir / f"fold_{fold_idx + 1}.json", "w") as fh:
            json.dump(fold_result, fh, indent=2)
        running_summary = _build_dataset_summary(
            spec, config, model_type, n_qubits, in_channels, fold_results, len(graphs), partial=True
        )
        with open(dataset_dir / "metrics.json", "w") as fh:
            json.dump(running_summary, fh, indent=2)

    summary = _build_dataset_summary(
        spec, config, model_type, n_qubits, in_channels, fold_results, len(graphs), partial=False
    )
    with open(dataset_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _build_dataset_summary(
    spec,
    config,
    model_type: str,
    n_qubits: int,
    in_channels: int,
    fold_results: List[Dict],
    n_graphs: int,
    partial: bool,
) -> Dict:
    """Shape per-dataset summary to match the existing report script.

    scripts/generate_cml_report.py expects:
      - top-level keys: dataset (label), config, summary{mean_*, std_*}, folds
      - per fold: train_curve, val_curve, qdrop_curve, accuracy, fold

    We also keep QDB-native keys (block_loss, n_blocks, fold_results alias)
    for downstream consumers.
    """
    accs = [r["accuracy"] for r in fold_results] or [0.0]
    summary_block = {
        f"mean_{m}": float(np.mean([r.get(m, 0.0) for r in fold_results])) if fold_results else 0.0
        for m in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
    }
    summary_block.update(
        {
            f"std_{m}": float(np.std([r.get(m, 0.0) for r in fold_results])) if fold_results else 0.0
            for m in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
        }
    )

    config_dict = vars(config) if not isinstance(config, dict) else dict(config)
    config_dict.setdefault("n_qubits", n_qubits)
    config_dict.setdefault("model", model_type)

    return {
        "timestamp": datetime.now().isoformat(),
        "dataset": spec.name,
        "dataset_source": spec.source,
        "task": spec.task,
        "n_classes": spec.n_classes,
        "n_graphs": n_graphs,
        "node_feature_dim": in_channels,
        "model": f"QDB-{model_type.upper()}",
        "config": config_dict,
        "summary": summary_block,
        "folds": fold_results,
        "fold_results": fold_results,  # alias for QDB-native callers
        "completed_folds": len(fold_results),
        "partial": partial,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QDB (Quantum-Diffusion-Block) training")
    p.add_argument("--datasets", nargs="+", default=["mutag", "proteins"])
    p.add_argument("--model-type", choices=["qgcn", "qgat"], default="qgcn")
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-qubits", type=int, default=None)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument(
        "--algorithm",
        choices=["pruning", "dropout", "both"],
        default="pruning",
        help="Q-Drop algorithm (no baseline allowed for QDB)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="training_results/qdb")
    return p


def main() -> None:
    args = build_parser().parse_args()
    config = QDBTrainConfig(
        n_blocks=args.n_blocks,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        n_qubits=args.n_qubits,
        n_folds=args.folds,
        early_stop_patience=args.early_stop_patience,
        algorithm=args.algorithm,
        seed=args.seed,
    )
    set_seed(config.seed)
    device = select_cuda_device(preferred_index=1)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"qdb_training_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"QDB training output: {output_dir.resolve()}")
    print(f"Device: {device}")

    results: Dict[str, Dict] = {}
    for ds in args.datasets:
        results[normalize_dataset_name(ds)] = train_dataset(
            ds, config, args.model_type, device, output_dir
        )
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
