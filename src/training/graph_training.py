"""Shared Torch training core for graph benchmarks with Q-Drop support."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.nn import LeakyReLU
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data.load_mutag import load_mutag
from data.load_proteins import load_proteins
from models.quantum_gcn import QGCN
from qdrop import QDropConfig, QDropRuntimeFactory, TorchQDropRuntime


@dataclass(frozen=True)
class GraphDatasetSpec:
    name: str
    source: str
    task: str
    n_classes: int = 2


DATASET_SPECS: Dict[str, GraphDatasetSpec] = {
    "mutag": GraphDatasetSpec(
        name="MUTAG",
        source="graphs-datasets/MUTAG",
        task="binary classification (mutagenic vs non-mutagenic)",
    ),
    "proteins": GraphDatasetSpec(
        name="PROTEINS",
        source="graphs-datasets/PROTEINS",
        task="binary classification (enzyme vs non-enzyme)",
    ),
}


@dataclass
class GraphTrainConfig:
    datasets: Sequence[str]
    epochs: int = 100
    lr: float = 5e-3
    weight_decay: float = 1e-3
    batch_size: int = 32
    q_depths: Tuple[int, int] = (1, 1)
    n_qubits: Optional[int] = None
    n_folds: int = 10
    early_stop_patience: int = 15
    val_frequency: int = 5
    grad_clip: float = 1.0
    use_scheduler: bool = True
    use_class_weights: bool = True
    algorithm: str = "baseline"
    accumulate_window: int = 10
    prune_window: int = 8
    prune_ratio: float = 0.8
    qdrop_schedule: bool = True
    dropout_prob: float = 0.5
    n_drop_wires: int = 1
    enable_forward_mask: bool = True
    output_dir: str = "training_results"
    seed: int = 42


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_score: Optional[float] = None
        self.counter = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None

    def step(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            return False

        self.counter += 1
        return self.counter >= self.patience


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_by_name(name: str):
    dataset_name = name.lower()
    if dataset_name == "mutag":
        return load_mutag()
    if dataset_name in {"proteins", "protein"}:
        return load_proteins()
    raise ValueError(f"Unsupported dataset: {name}")


def normalize_dataset_name(name: str) -> str:
    dataset_name = name.lower()
    if dataset_name == "protein":
        return "proteins"
    return dataset_name


def compute_class_weight(labels: Sequence[int], device: torch.device) -> torch.Tensor:
    counter = Counter(labels)
    n_pos = counter.get(1, 1)
    n_neg = counter.get(0, 1)
    pos_weight = n_neg / max(n_pos, 1)
    print(f"  Class distribution: {dict(counter)}")
    print(f"  Positive class weight: {pos_weight:.4f}")
    return torch.tensor([pos_weight], dtype=torch.float, device=device)


def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, float]:
    y_true_np = np.asarray(y_true).reshape(-1)
    y_pred_np = np.asarray(y_pred).reshape(-1)
    y_prob_np = np.asarray(y_prob).reshape(-1)

    metrics = {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "precision": float(precision_score(y_true_np, y_pred_np, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred_np, zero_division=0)),
        "f1": float(f1_score(y_true_np, y_pred_np, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_np, y_prob_np))
    except ValueError:
        metrics["roc_auc"] = 0.0

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true_np, y_prob_np))
    except ValueError:
        metrics["pr_auc"] = 0.0

    return metrics


def build_model(input_dims: int, config: GraphTrainConfig) -> nn.Module:
    return QGCN(
        input_dims=input_dims,
        q_depths=list(config.q_depths),
        output_dims=1,
        activ_fn=LeakyReLU(0.2),
        readout=False,
        n_qubits=config.n_qubits,
    )


def build_optimizer(model: nn.Module, config: GraphTrainConfig) -> optim.Optimizer:
    if config.weight_decay > 0:
        return optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return optim.Adam(model.parameters(), lr=config.lr)


def build_qdrop_manager(model: nn.Module, config: GraphTrainConfig):
    return QDropRuntimeFactory.create_torch(
        quantum_layers=model.qdrop_layers(),
        config=QDropConfig(
            algorithm=config.algorithm,
            accumulate_window=config.accumulate_window,
            prune_window=config.prune_window,
            prune_ratio=config.prune_ratio,
            schedule=config.qdrop_schedule,
            dropout_prob=config.dropout_prob,
            n_drop_wires=config.n_drop_wires,
            enable_forward_mask=config.enable_forward_mask,
        ),
    )


def format_qdrop_status(qdrop_manager: TorchQDropRuntime) -> str:
    state = qdrop_manager.describe_state()
    parts = [
        f"mode={state['active_step_mode']}",
        f"phase={'accum' if state['accumulate_phase'] else 'prune'}",
        f"a={state['accumulate_remaining']}",
        f"p={state['prune_remaining']}",
        f"r={state['current_prune_ratio']:.3f}",
        f"ps={state['pruning_step_count']}",
    ]

    if state["dropout_enabled"]:
        drop_parts = []
        for layer_id, dropped_wires in state["active_dropout_states"].items():
            wire_text = ",".join(str(wire_id) for wire_id in dropped_wires)
            drop_parts.append(f"{layer_id}[{wire_text}]")
        parts.append("drop=" + ";".join(drop_parts))
    else:
        parts.append("drop=off")

    return " | ".join(parts)


def snapshot_qdrop_state(qdrop_manager: TorchQDropRuntime, epoch: int) -> dict:
    state = qdrop_manager.describe_state()
    active_dropout_states = state["active_dropout_states"]
    dropped_wire_count = sum(len(dropped_wires) for dropped_wires in active_dropout_states.values())
    return {
        "epoch": epoch,
        "mode": state["active_step_mode"],
        "phase": "accumulate" if state["accumulate_phase"] else "prune",
        "is_prune_phase": 0 if state["accumulate_phase"] else 1,
        "dropout_enabled": 1 if state["dropout_enabled"] else 0,
        "active_dropout_layers": len(active_dropout_states),
        "dropped_wire_count": dropped_wire_count,
        "accumulate_remaining": state["accumulate_remaining"],
        "prune_remaining": state["prune_remaining"],
        "pruning_step_count": state["pruning_step_count"],
        "prune_ratio": state["current_prune_ratio"],
        "quantum_param_count": state["quantum_param_count"],
        "quantum_scalar_count": state["quantum_scalar_count"],
        "dropped_wires_by_layer": {
            layer_id: list(dropped_wires) for layer_id, dropped_wires in active_dropout_states.items()
        },
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch_index: Optional[int] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[OneCycleLR] = None,
    grad_clip: float = 1.0,
    qdrop_manager: Optional[TorchQDropRuntime] = None,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    if qdrop_manager is not None:
        if is_train:
            qdrop_manager.start_epoch(epoch_index or 0)
        else:
            qdrop_manager.clear_forward_masks()

    total_loss = 0.0
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.batch)
            if logits.dim() > 1 and logits.size(1) == 1:
                logits = logits.squeeze(1)

            target = batch.y.float()
            loss = criterion(logits, target)

            if is_train:
                loss.backward()
                if qdrop_manager is not None:
                    qdrop_manager.after_backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if qdrop_manager is not None:
                    qdrop_manager.after_step()
                if scheduler is not None:
                    scheduler.step()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            total_loss += float(loss.item())
            all_labels.extend(target.long().cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())

    return total_loss / max(len(loader), 1), compute_metrics(all_labels, all_preds, all_probs)


def train_fold(
    train_graphs: Sequence,
    test_graphs: Sequence,
    config: GraphTrainConfig,
    device: torch.device,
    dataset_name: str,
    fold_idx: int,
) -> Dict:
    train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=config.batch_size, shuffle=False)

    model = build_model(input_dims=train_graphs[0].x.size(1), config=config).to(device)

    pos_weight = None
    if config.use_class_weights:
        labels = [int(graph.y.item()) for graph in train_graphs]
        pos_weight = compute_class_weight(labels, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = build_optimizer(model, config)
    scheduler = None
    if config.use_scheduler:
        total_steps = max(len(train_loader) * config.epochs, 1)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.lr * 5.0,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=50.0,
        )

    qdrop_manager = build_qdrop_manager(model, config)
    if config.algorithm != "baseline":
        print(
            f"  Q-Drop mode: {config.algorithm} | "
            f"quantum tensors: {qdrop_manager.quantum_param_count} | "
            f"quantum scalars: {qdrop_manager.quantum_scalar_count}"
        )

    stopper = EarlyStopping(config.early_stop_patience)
    train_curve = []
    val_curve = []
    qdrop_curve = []

    print(f"  Fold {fold_idx + 1}: training...")
    progress = tqdm(range(1, config.epochs + 1), leave=False, desc=f"{dataset_name}-F{fold_idx + 1}")
    for epoch in progress:
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            epoch_index=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=config.grad_clip,
            qdrop_manager=qdrop_manager,
        )
        train_curve.append({"epoch": epoch, "loss": train_loss, **train_metrics})

        if config.algorithm != "baseline":
            qdrop_curve.append(snapshot_qdrop_state(qdrop_manager, epoch))
            progress.set_postfix_str(format_qdrop_status(qdrop_manager))

        if epoch % config.val_frequency != 0 and epoch != config.epochs:
            continue

        val_loss, val_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            qdrop_manager=qdrop_manager,
        )
        val_curve.append({"epoch": epoch, "loss": val_loss, **val_metrics})

        if config.algorithm != "baseline":
            print(
                f"    Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, val_acc={val_metrics['accuracy']:.4f} | "
                f"{format_qdrop_status(qdrop_manager)}"
            )

        if stopper.step(val_metrics["accuracy"], model):
            print(f"    Early stopping at epoch {epoch} (best val acc: {stopper.best_score:.4f})")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    test_loss, test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        qdrop_manager=qdrop_manager,
    )
    print(
        f"    Fold {fold_idx + 1} test: "
        f"acc={test_metrics['accuracy']:.4f}, f1={test_metrics['f1']:.4f}, "
        f"roc_auc={test_metrics['roc_auc']:.4f}"
    )

    return {
        "fold": fold_idx + 1,
        "test_loss": test_loss,
        **test_metrics,
        "train_curve": train_curve,
        "val_curve": val_curve,
        "qdrop_curve": qdrop_curve,
    }


def aggregate_fold_results(fold_results: Sequence[Dict]) -> Dict[str, float]:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    metrics: Dict[str, float] = {}
    for key in keys:
        values = [float(result[key]) for result in fold_results]
        metrics[f"mean_{key}"] = float(np.mean(values))
        metrics[f"std_{key}"] = float(np.std(values))
    return metrics


def serialize_result_payload(
    dataset_name: str,
    graphs: Sequence,
    config: GraphTrainConfig,
    fold_results: Sequence[Dict],
) -> Dict:
    dataset_spec = DATASET_SPECS[dataset_name.lower()]
    return {
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_spec.name,
        "config": {
            "epochs": config.epochs,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "batch_size": config.batch_size,
            "q_depths": list(config.q_depths),
            "n_qubits": config.n_qubits,
            "n_folds": config.n_folds,
            "early_stop_patience": config.early_stop_patience,
            "val_frequency": config.val_frequency,
            "grad_clip": config.grad_clip,
            "use_scheduler": config.use_scheduler,
            "use_class_weights": config.use_class_weights,
            "algorithm": config.algorithm,
            "accumulate_window": config.accumulate_window,
            "prune_window": config.prune_window,
            "prune_ratio": config.prune_ratio,
            "qdrop_schedule": config.qdrop_schedule,
            "dropout_prob": config.dropout_prob,
            "n_drop_wires": config.n_drop_wires,
            "enable_forward_mask": config.enable_forward_mask,
            "seed": config.seed,
        },
        "summary": aggregate_fold_results(fold_results),
        "folds": list(fold_results),
        "dataset_source": dataset_spec.source,
        "n_graphs": len(graphs),
        "n_classes": dataset_spec.n_classes,
        "node_feature_dim": graphs[0].x.size(1),
        "task": dataset_spec.task,
        "model": "QGCN",
    }


def train_dataset(
    dataset_name: str,
    config: GraphTrainConfig,
    device: torch.device,
    base_output: Optional[Path] = None,
) -> Dict:
    dataset_key = normalize_dataset_name(dataset_name)
    dataset_spec = DATASET_SPECS[dataset_key]

    print("\n" + "=" * 72)
    print(f"Training QGCN on {dataset_spec.name}")
    print("=" * 72)

    graphs = load_dataset_by_name(dataset_key)
    labels = [int(graph.y.item()) for graph in graphs]
    print(
        f"Loaded {len(graphs)} graphs | "
        f"Classes: {set(labels)} | Feature dim: {graphs[0].x.size(1)}"
    )

    splitter = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    fold_results: List[Dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(graphs, labels)):
        train_graphs = [graphs[index] for index in train_idx]
        test_graphs = [graphs[index] for index in test_idx]
        fold_results.append(
            train_fold(
                train_graphs=train_graphs,
                test_graphs=test_graphs,
                config=config,
                device=device,
                dataset_name=dataset_spec.name,
                fold_idx=fold_idx,
            )
        )

    result_payload = serialize_result_payload(dataset_key, graphs, config, fold_results)
    metrics = result_payload["summary"]
    print(
        f"{dataset_spec.name} results: "
        f"acc={metrics['mean_accuracy']:.4f}±{metrics['std_accuracy']:.4f}, "
        f"f1={metrics['mean_f1']:.4f}±{metrics['std_f1']:.4f}"
    )

    if base_output is not None:
        dataset_dir = base_output / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        out_path = dataset_dir / "metrics.json"
        with open(out_path, "w", encoding="utf-8") as output_file:
            json.dump(result_payload, output_file, indent=2)
        print(f"Saved metrics to: {out_path}")

    return result_payload


def build_train_parser(
    description: str,
    default_datasets: Sequence[str],
    default_batch_size: int = 32,
    default_weight_decay: float = 1e-3,
    default_grad_clip: float = 1.0,
    default_use_scheduler: bool = True,
    default_use_class_weights: bool = True,
    default_output_dir: str = "training_results",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--datasets", nargs="+", default=list(default_datasets), help="Datasets to train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=default_weight_decay)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--q-depths", nargs="+", type=int, default=[1, 1])
    parser.add_argument("--n-qubits", type=int, default=None, help="Override quantum tensor width (bucketed to 8 or 16)")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--val-frequency", type=int, default=5)
    parser.add_argument("--grad-clip", type=float, default=default_grad_clip)
    parser.add_argument(
        "--disable-scheduler",
        action="store_true",
        default=not default_use_scheduler,
    )
    parser.add_argument(
        "--disable-class-weights",
        action="store_true",
        default=not default_use_class_weights,
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="baseline",
        choices=["baseline", "pruning", "dropout", "both"],
        help="Q-Drop algorithm mode for graph quantum weights",
    )
    parser.add_argument("--accumulate-window", type=int, default=10)
    parser.add_argument("--prune-window", type=int, default=8)
    parser.add_argument("--prune-ratio", type=float, default=0.8)
    parser.add_argument("--disable-qdrop-schedule", action="store_true")
    parser.add_argument("--drop-prob", type=float, default=0.5)
    parser.add_argument("--n-drop-wires", type=int, default=1)
    parser.add_argument("--disable-forward-mask", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    return parser


def config_from_args(args: argparse.Namespace) -> GraphTrainConfig:
    return GraphTrainConfig(
        datasets=args.datasets,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        q_depths=tuple(args.q_depths),
        n_qubits=args.n_qubits,
        n_folds=args.folds,
        early_stop_patience=args.early_stop_patience,
        val_frequency=args.val_frequency,
        grad_clip=args.grad_clip,
        use_scheduler=not args.disable_scheduler,
        use_class_weights=not args.disable_class_weights,
        algorithm=args.algorithm,
        accumulate_window=args.accumulate_window,
        prune_window=args.prune_window,
        prune_ratio=args.prune_ratio,
        qdrop_schedule=not args.disable_qdrop_schedule,
        dropout_prob=args.drop_prob,
        n_drop_wires=args.n_drop_wires,
        enable_forward_mask=not args.disable_forward_mask,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def run_experiments(config: GraphTrainConfig) -> Tuple[Path, Dict[str, Dict]]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(config.output_dir) / f"quantum_graph_training_{timestamp}"
    base_output.mkdir(parents=True, exist_ok=True)

    print("Unified quantum training started")
    print(f"Device: {device}")
    print(f"Datasets: {[dataset.upper() for dataset in config.datasets]}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Output: {base_output.resolve()}")

    all_results: Dict[str, Dict] = {}
    for dataset_name in config.datasets:
        dataset_key = normalize_dataset_name(dataset_name)
        all_results[dataset_key] = train_dataset(
            dataset_name=dataset_name,
            config=config,
            device=device,
            base_output=base_output,
        )

    summary_path = base_output / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(all_results, summary_file, indent=2)
    print(f"\nSaved global summary to: {summary_path}")

    return base_output, all_results
