"""Shared Torch training core for graph benchmarks with Q-Drop support."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from models.quantum_gat import QGAT
from qdrop import QDropConfig, QDropRuntimeFactory, TorchQDropRuntime

# Allow importing from the top-level data_loader package that lives two
# directories above src/training/ (project root → data_loader/).
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data_loader.load_nci1 import load_nci1
from data_loader.load_imdb_binary import load_imdb_binary
from data_loader.load_imdb_multi import load_imdb_multi


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
    "nci1": GraphDatasetSpec(
        name="NCI1",
        source="TUDataset/NCI1",
        task="binary classification (active vs inactive, lung cancer)",
    ),
    "imdb_binary": GraphDatasetSpec(
        name="IMDB-BINARY",
        source="TUDataset/IMDB-BINARY",
        task="binary classification (Action vs Romance)",
    ),
    "imdb_multi": GraphDatasetSpec(
        name="IMDB-MULTI",
        source="TUDataset/IMDB-MULTI",
        task="3-class classification (Comedy / Romance / Sci-Fi)",
        n_classes=3,
    ),
}


# Per-dataset quantum-tensor-width overrides. All clamped to 8 qubits:
# 2^16 state vector on lightning.qubit (CPU C++) still costs ~650 M
# complex ops per forward at n_qubits=16, and 10-fold CV × 100 epochs ×
# ~126 steps/fold on PROTEINS (~80 trillion ops total) exceeds the
# 6-hour self-hosted runner timeout. Capacity-experiments at n=16 are
# kept for ad-hoc runs via the --n-qubits CLI flag; routine CI uses 8.
# PROTEINS (3 features), IMDB (136/89 degree one-hot), NCI1 (37 features)
# all train usefully at 8 qubits.
DATASET_NQUBITS_OVERRIDES: Dict[str, int] = {
    "proteins": 8,
    "nci1": 8,
    "imdb_binary": 8,
    "imdb_multi": 8,
}


# Per-dataset batch-size overrides. With n_qubits clamped to 8 the
# simulator state is 2^8 = 256 amplitudes (vs 65 536 at n=16), and the
# previous batch=8 emergency cap is no longer needed. Letting batch
# return to the default 32 restores SGD signal quality and 4x throughput.
DATASET_BATCHSIZE_OVERRIDES: Dict[str, int] = {}

# Per-dataset model-architecture overrides that activate alongside the
# n_qubits override (same applied_nqubits_override gate). Only fires when
# the operator has not pinned n_qubits explicitly, preserving reproducibility
# of pinned experiments. Each key maps to a dict of GraphTrainConfig fields.
DATASET_MODEL_OVERRIDES: Dict[str, Dict] = {
    # NCI1 overrides: multi-scale pooling + MLP head + residual give the
    # largest accuracy gains at zero extra quantum-circuit cost.
    # StronglyEntanglingLayers is omitted: it triples circuit cost
    # (~6h total on 10-fold/150-epoch), while mean+max+add pooling and
    # the MLP head already give the biggest accuracy boost.
    "nci1": {
        "pool_type": "multiscale",
        "use_mlp_head": True,
        "mlp_hidden": 64,
        "mlp_dropout": 0.3,
        "use_residual": True,
        "q_depths": (3, 3),
        "epochs": 150,
        "early_stop_patience": 20,
    },
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
    quantum_lr_scale: float = 0.1
    pool_type: str = "mean"
    use_mlp_head: bool = False
    mlp_hidden: int = 64
    mlp_dropout: float = 0.5
    use_residual: bool = False
    use_strongly_entangling: bool = False
    model_type: str = "qgcn"
    attn_dropout: float = 0.2
    technique: str = "baseline"  # baseline | small_angle | qng | layerwise


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
    if dataset_name == "nci1":
        return load_nci1()
    if dataset_name in {"imdb_binary", "imdb-binary"}:
        return load_imdb_binary()
    if dataset_name in {"imdb_multi", "imdb-multi"}:
        return load_imdb_multi()
    raise ValueError(f"Unsupported dataset: {name}")


def normalize_dataset_name(name: str) -> str:
    dataset_name = name.lower().replace("-", "_")
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


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob,
    n_classes: int = 2,
) -> Dict[str, float]:
    y_true_np = np.asarray(y_true).reshape(-1)
    y_pred_np = np.asarray(y_pred).reshape(-1)

    avg = "binary" if n_classes == 2 else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "precision": float(precision_score(y_true_np, y_pred_np, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred_np, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true_np, y_pred_np, average=avg, zero_division=0)),
    }

    if n_classes == 2:
        y_prob_np = np.asarray(y_prob).reshape(-1)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true_np, y_prob_np))
        except ValueError:
            metrics["roc_auc"] = 0.0
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true_np, y_prob_np))
        except ValueError:
            metrics["pr_auc"] = 0.0
    else:
        y_prob_np = np.asarray(y_prob)  # (N, n_classes)
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true_np, y_prob_np, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0  # not well-defined for multi-class

    return metrics


def build_model(input_dims: int, config: GraphTrainConfig, n_classes: int = 2) -> nn.Module:
    output_dims = 1 if n_classes == 2 else n_classes

    # Resolve QML technique hooks (weight_init for the quantum TorchLayer).
    from qml_techniques.registry import get_technique

    technique_spec = get_technique(getattr(config, "technique", "baseline"))

    if config.model_type == "qgat":
        return QGAT(
            input_dims=input_dims,
            q_depths=list(config.q_depths),
            output_dims=output_dims,
            attn_dropout=config.attn_dropout,
            layer_dropout=config.attn_dropout,
            max_qubits=config.n_qubits or 8,
            pool_type=config.pool_type,
            use_mlp_head=config.use_mlp_head,
            mlp_hidden=config.mlp_hidden,
            mlp_dropout=config.mlp_dropout,
            use_residual=config.use_residual,
        )

    return QGCN(
        input_dims=input_dims,
        q_depths=list(config.q_depths),
        output_dims=output_dims,
        activ_fn=LeakyReLU(0.2),
        readout=False,
        n_qubits=config.n_qubits,
        pool_type=config.pool_type,
        use_mlp_head=config.use_mlp_head,
        mlp_hidden=config.mlp_hidden,
        mlp_dropout=config.mlp_dropout,
        use_residual=config.use_residual,
        use_strongly_entangling=config.use_strongly_entangling,
        weight_init=technique_spec.weight_init,
    )


def split_quantum_classical_params(
    model: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Separate the trainable quantum-circuit weights from classical params.

    PennyLane's ``TorchLayer`` registers its rotation angles as
    ``quantum_layer.weights`` inside ``QuantumCircuitAdapter`` instances.
    Those tensors are tiny (n_layers x n_qubits = 16 scalars by default)
    and live on a much noisier loss surface than the surrounding Linear
    layers because every Q-Drop pruning step masks a fraction of their
    gradient. Splitting them into their own optimizer group lets the
    trainer dial in a lower learning rate without slowing the classical
    encoder / classifier.
    """
    quantum_params: List[nn.Parameter] = []
    classical_params: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "quantum_layer.weights" in name or name.endswith(".quantum_layer.weights"):
            quantum_params.append(parameter)
        else:
            classical_params.append(parameter)
    return quantum_params, classical_params


def build_optimizer(model: nn.Module, config: GraphTrainConfig) -> optim.Optimizer:
    quantum_params, classical_params = split_quantum_classical_params(model)

    # QNG branch: precondition quantum gradients by the Fubini-Study metric
    # tensor evaluated at current angles + a zero-input sample. Only fires
    # for QGCN (QGAT uses a different quantum_layer attribute layout).
    if getattr(config, "technique", "baseline") == "qng" and quantum_params:
        try:
            import pennylane as qml  # noqa: F401  # needed for metric_tensor
            from qml_techniques.qng import QNGAdam

            def _metric_provider(p: nn.Parameter) -> torch.Tensor:
                for layer in getattr(model, "layers", []):
                    quantum_layer = getattr(layer, "quantum_layer", None)
                    if quantum_layer is None:
                        continue
                    if getattr(quantum_layer, "weights", None) is p:
                        bare_qnode = getattr(quantum_layer, "bare_qnode", None)
                        if bare_qnode is None:
                            break
                        sample_input = torch.zeros(quantum_layer.n_qubits, device=p.device)
                        try:
                            metric_fn = qml.metric_tensor(bare_qnode, approx="block-diag")
                            f = metric_fn(sample_input, p.detach())
                            if not isinstance(f, torch.Tensor):
                                f = torch.as_tensor(f, device=p.device, dtype=p.dtype)
                            return f.detach().to(device=p.device, dtype=p.dtype).reshape(p.numel(), p.numel())
                        except Exception:  # noqa: BLE001 — fall back to identity on failure
                            return torch.eye(p.numel(), device=p.device, dtype=p.dtype)
                return torch.eye(p.numel(), device=p.device, dtype=p.dtype)

            return QNGAdam(
                quantum_params=quantum_params,
                classical_params=classical_params,
                metric_provider=_metric_provider,
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        except ImportError:
            pass  # qml_techniques unavailable; fall through to standard optimizer

    quantum_lr = config.lr * float(config.quantum_lr_scale)
    if quantum_params:
        param_groups: List[Dict] = [
            {"params": classical_params, "lr": config.lr, "name": "classical"},
            {"params": quantum_params, "lr": quantum_lr, "name": "quantum"},
        ]
    else:
        param_groups = [
            {"params": classical_params, "lr": config.lr, "name": "classical"},
        ]

    if config.weight_decay > 0:
        return optim.AdamW(param_groups, lr=config.lr, weight_decay=config.weight_decay)
    return optim.Adam(param_groups, lr=config.lr)


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
    n_classes: int = 2,
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
    all_probs: List = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index, batch.batch)

            if n_classes == 2:
                if logits.dim() > 1 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                target = batch.y.float()
                loss = criterion(logits, target)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                batch_probs = probs.detach().cpu().tolist()
            else:
                target = batch.y.long()
                loss = criterion(logits, target)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                batch_probs = probs.detach().cpu().tolist()  # List[List[float]]

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

            total_loss += float(loss.item())
            all_labels.extend(target.long().cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(batch_probs)

    return total_loss / max(len(loader), 1), compute_metrics(all_labels, all_preds, all_probs, n_classes=n_classes)


def train_fold(
    train_graphs: Sequence,
    test_graphs: Sequence,
    config: GraphTrainConfig,
    device: torch.device,
    dataset_name: str,
    fold_idx: int,
    n_classes: int = 2,
) -> Dict:
    train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=config.batch_size, shuffle=False)

    model = build_model(input_dims=train_graphs[0].x.size(1), config=config, n_classes=n_classes).to(device)

    if n_classes == 2:
        pos_weight = None
        if config.use_class_weights:
            labels = [int(graph.y.item()) for graph in train_graphs]
            pos_weight = compute_class_weight(labels, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = None
    if config.use_scheduler:
        total_steps = max(len(train_loader) * config.epochs, 1)
        # OneCycleLR rescales every parameter group to the same ``max_lr``
        # unless we hand it a list. With per-group quantum/classical learning
        # rates we have to keep the quantum branch on the lower ramp so the
        # smaller, noisier quantum gradient sees a proportionally smaller
        # peak step.
        classical_max_lr = config.lr * 5.0
        max_lr_per_group = [
            classical_max_lr if group.get("name") != "quantum"
            else classical_max_lr * float(config.quantum_lr_scale)
            for group in optimizer.param_groups
        ]
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr_per_group,
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

    # Layerwise learning schedule: freeze per-conv quantum weights and
    # progressively unfreeze across training phases. No-op for other techniques.
    layerwise_schedule = None
    if getattr(config, "technique", "baseline") == "layerwise":
        try:
            from qml_techniques.layerwise import LayerwiseSchedule

            n_phases = max(1, len(config.q_depths))
            layerwise_schedule = LayerwiseSchedule(
                model=model, total_epochs=config.epochs, n_phases=n_phases
            )
            print(f"  Layerwise: {n_phases} phases over {config.epochs} epochs")
        except ImportError:
            layerwise_schedule = None

    stopper = EarlyStopping(config.early_stop_patience)
    train_curve = []
    val_curve = []
    qdrop_curve = []

    print(f"  Fold {fold_idx + 1}: training...")
    progress = tqdm(range(1, config.epochs + 1), leave=False, desc=f"{dataset_name}-F{fold_idx + 1}")
    for epoch in progress:
        if layerwise_schedule is not None:
            layerwise_schedule.apply_for_epoch(epoch - 1)
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
            n_classes=n_classes,
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
            n_classes=n_classes,
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
        n_classes=n_classes,
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
            "quantum_lr_scale": config.quantum_lr_scale,
            "seed": config.seed,
            "pool_type": config.pool_type,
            "use_mlp_head": config.use_mlp_head,
            "mlp_hidden": config.mlp_hidden,
            "mlp_dropout": config.mlp_dropout,
            "use_residual": config.use_residual,
            "use_strongly_entangling": config.use_strongly_entangling,
            "model_type": config.model_type,
            "attn_dropout": config.attn_dropout,
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

    # Apply per-dataset n_qubits override unless the operator pinned a
    # width explicitly via --n-qubits. Keeps reproducibility intact while
    # letting PROTEINS run on a wider circuit than MUTAG.
    applied_nqubits_override = False
    if config.n_qubits is None and dataset_key in DATASET_NQUBITS_OVERRIDES:
        override = DATASET_NQUBITS_OVERRIDES[dataset_key]
        config = replace(config, n_qubits=override)
        applied_nqubits_override = True
        print(f"  Per-dataset override: n_qubits -> {override}")

    # When the wider quantum circuit is active, also shrink the graph
    # batch so the simulator state fits in GPU memory. Only triggers when
    # we picked the override above, so an explicit --n-qubits run keeps
    # the operator-chosen batch size.
    if applied_nqubits_override and dataset_key in DATASET_BATCHSIZE_OVERRIDES:
        batch_override = DATASET_BATCHSIZE_OVERRIDES[dataset_key]
        if config.batch_size > batch_override:
            config = replace(config, batch_size=batch_override)
            print(f"  Per-dataset override: batch_size -> {batch_override}")

    # Apply per-dataset model architecture overrides (e.g. NCI1 gets
    # multi-scale pooling + MLP head + StronglyEntanglingLayers). Gated on
    # applied_nqubits_override so that explicit --n-qubits runs keep the
    # operator-chosen architecture for reproducibility.
    _MODEL_DEFAULT_SENTINELS = {
        "pool_type": "mean",
        "use_mlp_head": False,
        "use_residual": False,
        "use_strongly_entangling": False,
    }
    if applied_nqubits_override and dataset_key in DATASET_MODEL_OVERRIDES:
        patch: Dict = {}
        model_overrides = DATASET_MODEL_OVERRIDES[dataset_key]
        for field, sentinel in _MODEL_DEFAULT_SENTINELS.items():
            if getattr(config, field) == sentinel and field in model_overrides:
                patch[field] = model_overrides[field]
        for field in ("mlp_hidden", "mlp_dropout", "epochs", "early_stop_patience", "q_depths"):
            if field in model_overrides:
                patch[field] = model_overrides[field]
        if patch:
            config = replace(config, **patch)
            print(f"  Per-dataset model overrides: {patch}")

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

    # Pre-create the dataset output dir so per-fold checkpoints have a
    # destination on disk before fold 1 even finishes. If the runner is
    # canceled mid-training, all completed folds remain visible in the
    # uploaded artifact (upload-artifact runs with `if: always()`).
    dataset_dir: Optional[Path] = None
    if base_output is not None:
        dataset_dir = base_output / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)

    n_classes = dataset_spec.n_classes
    fold_iter = tqdm(
        enumerate(splitter.split(graphs, labels)),
        total=config.n_folds,
        desc=f"{dataset_spec.name} folds",
        leave=True,
    )
    for fold_idx, (train_idx, test_idx) in fold_iter:
        train_graphs = [graphs[index] for index in train_idx]
        test_graphs = [graphs[index] for index in test_idx]
        fold_result = train_fold(
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            config=config,
            device=device,
            dataset_name=dataset_spec.name,
            fold_idx=fold_idx,
            n_classes=n_classes,
        )
        fold_results.append(fold_result)

        # Per-fold checkpoint: rewrite metrics.json after EVERY fold so a
        # mid-run cancel preserves partial results. Also snapshot the
        # single fold as fold_<n>.json for granular post-mortem.
        if dataset_dir is not None:
            try:
                partial_payload = serialize_result_payload(
                    dataset_key, graphs, config, fold_results
                )
                partial_payload["partial"] = fold_idx + 1 < config.n_folds
                partial_payload["completed_folds"] = len(fold_results)
                with open(dataset_dir / "metrics.json", "w", encoding="utf-8") as fh:
                    json.dump(partial_payload, fh, indent=2)
                with open(
                    dataset_dir / f"fold_{fold_idx + 1}.json", "w", encoding="utf-8"
                ) as fh:
                    json.dump(fold_result, fh, indent=2)
            except Exception as exc:  # noqa: BLE001 — never break training on persistence failure
                print(f"WARN: per-fold checkpoint write failed: {exc}", flush=True)

        try:
            fold_iter.set_postfix_str(
                f"acc={fold_result.get('accuracy', 0.0):.3f}"
            )
        except AttributeError:
            pass

    result_payload = serialize_result_payload(dataset_key, graphs, config, fold_results)
    result_payload["partial"] = False
    result_payload["completed_folds"] = len(fold_results)
    metrics = result_payload["summary"]
    print(
        f"{dataset_spec.name} results: "
        f"acc={metrics['mean_accuracy']:.4f}±{metrics['std_accuracy']:.4f}, "
        f"f1={metrics['mean_f1']:.4f}±{metrics['std_f1']:.4f}"
    )

    if dataset_dir is not None:
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
    parser.add_argument(
        "--quantum-lr-scale",
        type=float,
        default=0.1,
        help="Multiplier applied to --lr for the quantum parameter group (default 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--pool-type", type=str, default="mean", choices=["mean", "multiscale"])
    parser.add_argument("--use-mlp-head", action="store_true", default=False)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--mlp-dropout", type=float, default=0.5)
    parser.add_argument("--use-residual", action="store_true", default=False)
    parser.add_argument("--use-strongly-entangling", action="store_true", default=False)
    parser.add_argument("--model-type", type=str, default="qgcn", choices=["qgcn", "qgat"])
    parser.add_argument("--attn-dropout", type=float, default=0.2)
    parser.add_argument(
        "--technique",
        type=str,
        default="baseline",
        choices=["baseline", "small_angle", "qng", "layerwise"],
        help="QML quantum-weight optimization technique",
    )
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
        quantum_lr_scale=args.quantum_lr_scale,
        output_dir=args.output_dir,
        seed=args.seed,
        pool_type=args.pool_type,
        use_mlp_head=args.use_mlp_head,
        mlp_hidden=args.mlp_hidden,
        mlp_dropout=args.mlp_dropout,
        use_residual=args.use_residual,
        use_strongly_entangling=args.use_strongly_entangling,
        model_type=args.model_type,
        attn_dropout=args.attn_dropout,
        technique=args.technique,
    )


def select_cuda_device(preferred_index: int = 1) -> torch.device:
    """Pick a CUDA device, preferring ``preferred_index`` when available.

    The Q-Drop training rigs ship with dual RTX PRO 6000 cards. GPU 0 is
    usually shared with display / Isaac Sim, so the trainer prefers GPU 1
    when it exists and is visible. Falls back to GPU 0, then CPU, so the
    same code path runs on single-GPU dev boxes and CI runners without
    changes. Respects ``CUDA_VISIBLE_DEVICES``: if the operator pinned a
    single device, ``torch.cuda.device_count()`` already reports 1 and we
    quietly use the only visible card.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    device_count = torch.cuda.device_count()
    if device_count > preferred_index:
        return torch.device(f"cuda:{preferred_index}")
    return torch.device("cuda:0")


def run_experiments(config: GraphTrainConfig) -> Tuple[Path, Dict[str, Dict]]:
    set_seed(config.seed)
    device = select_cuda_device(preferred_index=1)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(config.output_dir) / f"quantum_graph_training_{timestamp}"
    base_output.mkdir(parents=True, exist_ok=True)

    print("Unified quantum training started")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"Device: {device} ({gpu_name})")
    else:
        print(f"Device: {device}")
    print(f"Datasets: {[dataset.upper() for dataset in config.datasets]}")
    print(f"Algorithm: {config.algorithm}")
    print(f"Output: {base_output.resolve()}")

    summary_path = base_output / "summary.json"
    all_results: Dict[str, Dict] = {}
    for dataset_name in config.datasets:
        dataset_key = normalize_dataset_name(dataset_name)
        all_results[dataset_key] = train_dataset(
            dataset_name=dataset_name,
            config=config,
            device=device,
            base_output=base_output,
        )
        # Rewrite the global summary after every dataset so a mid-run
        # cancel preserves the completed datasets in the uploaded artifact.
        try:
            with open(summary_path, "w", encoding="utf-8") as summary_file:
                json.dump(all_results, summary_file, indent=2)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: per-dataset summary write failed: {exc}", flush=True)

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(all_results, summary_file, indent=2)
    print(f"\nSaved global summary to: {summary_path}")

    return base_output, all_results
