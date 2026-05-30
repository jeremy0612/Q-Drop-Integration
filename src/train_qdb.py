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

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

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

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(graphs, labels)):
        train_graphs = [graphs[i] for i in train_idx]
        test_graphs = [graphs[i] for i in test_idx]
        train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=config.batch_size, shuffle=False)

        model = build_model(model_type, config.n_blocks, in_channels, n_qubits, n_classes).to(device)
        history = train_qdb(model, train_loader, test_loader, config, device)
        final = evaluate(model, test_loader, device)
        print(f"  fold {fold_idx + 1}: accuracy={final['accuracy']:.4f}")
        fold_results.append(
            {
                "fold": fold_idx + 1,
                "accuracy": final["accuracy"],
                "block_loss": history["block_loss"],
                "val_accuracy_curve": history["val_accuracy"],
            }
        )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": spec.name,
        "model": model_type,
        "n_blocks": config.n_blocks,
        "n_qubits": n_qubits,
        "algorithm": config.algorithm,
        "config": vars(config),
        "fold_results": fold_results,
        "mean_accuracy": float(np.mean([r["accuracy"] for r in fold_results])),
        "std_accuracy": float(np.std([r["accuracy"] for r in fold_results])),
    }
    dataset_dir = output_dir / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


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
