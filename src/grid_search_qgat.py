"""Exhaustive grid search over QGAT architecture + Q-Drop hyperparameters for MUTAG/PROTEINS.

Usage:
    python grid_search_qgat.py --datasets mutag proteins --output-dir grid_search_results
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from itertools import product
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
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data.load_mutag import load_mutag
from data.load_proteins import load_proteins
from models.Quantum_GAT import QGAT
from qdrop import QDropConfig, QDropRuntimeFactory
from qdrop.types import QDropLayerSpec, QDropTensorSpec
from training.graph_training import normalize_dataset_name, set_seed

# ── parameter grids ───────────────────────────────────────────────────────────

QGAT_ARCH_GRID = {
    "max_qubits": [4, 8],
    "use_layer_norm": [True, False],
}

PRUNING_GRID = {
    "accumulate_window": [5, 10, 20],
    "prune_window": [4, 8, 16],
    "prune_ratio": [0.5, 0.6, 0.7, 0.8, 0.9],
    "qdrop_schedule": [True, False],
}

DROPOUT_GRID = {
    "dropout_prob": [0.3, 0.5, 0.7],
    "n_drop_wires": [1, 2, 4],
    "enable_forward_mask": [True, False],
}

FIXED_CONFIG: Dict = {
    "epochs": 100,
    "lr": 5e-3,
    "weight_decay": 1e-3,
    "batch_size": 32,
    "q_depths": (1,),
    "n_folds": 10,
    "early_stop_patience": 15,
    "val_frequency": 5,
    "grad_clip": 1.0,
    "gat_dropout": 0.2,
    "seed": 42,
}

CSV_FIELDS = [
    "dataset", "algorithm",
    "max_qubits", "use_layer_norm",
    "accumulate_window", "prune_window", "prune_ratio", "qdrop_schedule",
    "dropout_prob", "n_drop_wires", "enable_forward_mask",
    "mean_accuracy", "std_accuracy", "mean_f1", "std_f1",
    "mean_roc_auc", "std_roc_auc", "mean_precision", "std_precision",
    "mean_recall", "std_recall", "mean_pr_auc", "std_pr_auc",
]


def _make_combo_key(dataset: str, params: Dict) -> tuple:
    return (
        dataset,
        params.get("algorithm", ""),
        str(params.get("max_qubits", "")),
        str(params.get("use_layer_norm", "")),
        str(params.get("accumulate_window", "")),
        str(params.get("prune_window", "")),
        str(params.get("prune_ratio", "")),
        str(params.get("qdrop_schedule", "")),
        str(params.get("dropout_prob", "")),
        str(params.get("n_drop_wires", "")),
        str(params.get("enable_forward_mask", "")),
    )


def build_algorithm_grids() -> Dict[str, List[Dict]]:
    grids: Dict[str, List[Dict]] = {}
    arch_keys = list(QGAT_ARCH_GRID.keys())
    pruning_keys = list(PRUNING_GRID.keys())
    dropout_keys = list(DROPOUT_GRID.keys())
    all_qdrop_keys = pruning_keys + dropout_keys

    grids["baseline"] = []
    for arch_vals in product(*QGAT_ARCH_GRID.values()):
        combo = dict(zip(arch_keys, arch_vals))
        combo["algorithm"] = "baseline"
        grids["baseline"].append(combo)

    grids["pruning"] = []
    for arch_vals in product(*QGAT_ARCH_GRID.values()):
        arch_combo = dict(zip(arch_keys, arch_vals))
        for qdrop_vals in product(*PRUNING_GRID.values()):
            combo = {**arch_combo, **dict(zip(pruning_keys, qdrop_vals))}
            combo["algorithm"] = "pruning"
            grids["pruning"].append(combo)

    grids["dropout"] = []
    for arch_vals in product(*QGAT_ARCH_GRID.values()):
        arch_combo = dict(zip(arch_keys, arch_vals))
        for qdrop_vals in product(*DROPOUT_GRID.values()):
            combo = {**arch_combo, **dict(zip(dropout_keys, qdrop_vals))}
            combo["algorithm"] = "dropout"
            grids["dropout"].append(combo)

    grids["both"] = []
    for arch_vals in product(*QGAT_ARCH_GRID.values()):
        arch_combo = dict(zip(arch_keys, arch_vals))
        for qdrop_vals in product(*PRUNING_GRID.values(), *DROPOUT_GRID.values()):
            combo = {**arch_combo, **dict(zip(all_qdrop_keys, qdrop_vals))}
            combo["algorithm"] = "both"
            grids["both"].append(combo)

    return grids


# ── QGAT Q-Drop layer spec builder ────────────────────────────────────────────

def _build_qgat_layer_specs(model: QGAT) -> List[QDropLayerSpec]:
    """Build QDropLayerSpec objects for the VQC and HEA attention circuits in each QGATConv."""
    specs = []
    for layer_idx, conv in enumerate(model.layers):
        # VQC: q_weights shape (n_layers, n_qubits, 2)
        vqc_param = conv.vqc.q_weights
        n_q = conv.n_qubits

        def _vqc_mask(wire_ids, p=vqc_param, nq=n_q):
            mask = torch.zeros(p.shape, dtype=torch.bool, device=p.device)
            for w in wire_ids:
                if 0 <= w < nq:
                    mask[:, w, :] = True
            return mask

        specs.append(QDropLayerSpec(
            layer_id=f"layers.{layer_idx}.vqc",
            tensor_specs=[QDropTensorSpec(
                tensor_id=f"layers.{layer_idx}.vqc.q_weights",
                parameter=vqc_param,
                num_wires=n_q,
                supports_gradient_mask=True,
                supports_forward_mask=False,
                mask_builder=_vqc_mask,
            )],
        ))

        # HEA attention: weights shape (n_attn_layers * attn_qubits * 2 + 1,)
        # Layout per attention layer: [RY_0..RY_{q-1}, RZ_0..RZ_{q-1}], plus final RY on last qubit.
        attn_param = conv.quantum_attention.weights
        attn_q = conv.attn_qubits
        n_attn_w = attn_param.shape[0]
        n_attn_layers = max(1, (n_attn_w - 1) // max(attn_q * 2, 1))

        def _attn_mask(wire_ids, p=attn_param, aq=attn_q, nl=n_attn_layers):
            mask = torch.zeros(p.shape, dtype=torch.bool, device=p.device)
            for w in wire_ids:
                if 0 <= w < aq:
                    for layer in range(nl):
                        base = layer * aq * 2
                        mask[base + w] = True        # RY for wire w
                        mask[base + aq + w] = True   # RZ for wire w
            return mask

        specs.append(QDropLayerSpec(
            layer_id=f"layers.{layer_idx}.attn",
            tensor_specs=[QDropTensorSpec(
                tensor_id=f"layers.{layer_idx}.attn.weights",
                parameter=attn_param,
                num_wires=attn_q,
                supports_gradient_mask=True,
                supports_forward_mask=False,
                mask_builder=_attn_mask,
            )],
        ))

    return specs


# ── training helpers ──────────────────────────────────────────────────────────

def _pos_weight(train_graphs: list, device: torch.device) -> torch.Tensor:
    labels = [int(g.y.item()) for g in train_graphs]
    counter = Counter(labels)
    w = counter.get(0, 1) / max(counter.get(1, 1), 1)
    return torch.tensor([w], dtype=torch.float, device=device)


def _compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_prob = np.asarray(y_prob).ravel()
    m = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        m["roc_auc"] = 0.0
    try:
        m["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        m["pr_auc"] = 0.0
    return m


def _aggregate_folds(fold_results: List[Dict]) -> Dict[str, float]:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in fold_results]
        out[f"mean_{k}"] = float(np.mean(vals))
        out[f"std_{k}"]  = float(np.std(vals))
    return out


class _EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_score: Optional[float] = None
        self.counter = 0
        self.best_state: Optional[Dict] = None

    def step(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch_index: Optional[int] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler=None,
    grad_clip: float = 1.0,
    qdrop_manager=None,
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

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
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
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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

    return total_loss / max(len(loader), 1), _compute_metrics(all_labels, all_preds, all_probs)


def _train_fold(
    train_graphs: list,
    test_graphs: list,
    combo_params: Dict,
    device: torch.device,
    fold_idx: int,
    bar_desc: str,
) -> Dict:
    algo = combo_params["algorithm"]

    train_loader = DataLoader(train_graphs, batch_size=FIXED_CONFIG["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=FIXED_CONFIG["batch_size"], shuffle=False)

    model = QGAT(
        input_dims=train_graphs[0].x.size(1),
        q_depths=list(FIXED_CONFIG["q_depths"]),
        output_dims=1,
        dropout=FIXED_CONFIG["gat_dropout"],
        max_qubits=combo_params["max_qubits"],
        use_layer_norm=combo_params["use_layer_norm"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(train_graphs, device))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=FIXED_CONFIG["lr"],
        weight_decay=FIXED_CONFIG["weight_decay"],
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=FIXED_CONFIG["lr"] * 5,
        total_steps=max(len(train_loader) * FIXED_CONFIG["epochs"], 1),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=50.0,
    )

    qdrop_manager = QDropRuntimeFactory.create_torch(
        quantum_layers=_build_qgat_layer_specs(model),
        config=QDropConfig(
            algorithm=algo,
            accumulate_window=combo_params.get("accumulate_window", 10),
            prune_window=combo_params.get("prune_window", 8),
            prune_ratio=combo_params.get("prune_ratio", 0.8),
            schedule=combo_params.get("qdrop_schedule", True),
            dropout_prob=combo_params.get("dropout_prob", 0.5),
            n_drop_wires=combo_params.get("n_drop_wires", 1),
            enable_forward_mask=combo_params.get("enable_forward_mask", True),
        ),
    )

    stopper    = _EarlyStopping(FIXED_CONFIG["early_stop_patience"])
    train_hist = []
    val_hist   = []

    for epoch in tqdm(
        range(1, FIXED_CONFIG["epochs"] + 1),
        leave=False,
        desc=f"{bar_desc}-F{fold_idx + 1}",
    ):
        tr_loss, tr_m = _run_epoch(
            model, train_loader, criterion, device,
            epoch_index=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=FIXED_CONFIG["grad_clip"],
            qdrop_manager=qdrop_manager,
        )
        train_hist.append({"epoch": epoch, "loss": tr_loss, **tr_m})

        if epoch % FIXED_CONFIG["val_frequency"] != 0 and epoch != FIXED_CONFIG["epochs"]:
            continue

        val_loss, val_m = _run_epoch(
            model, test_loader, criterion, device, qdrop_manager=qdrop_manager,
        )
        val_hist.append({"epoch": epoch, "loss": val_loss, **val_m})

        if stopper.step(val_m["accuracy"], model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    _, test_m = _run_epoch(model, test_loader, criterion, device, qdrop_manager=qdrop_manager)
    print(
        f"    Fold {fold_idx + 1}: "
        f"acc={test_m['accuracy']:.4f}  f1={test_m['f1']:.4f}  roc={test_m['roc_auc']:.4f}"
    )
    return {"fold": fold_idx + 1, **test_m, "train_hist": train_hist, "val_hist": val_hist}


def _load_dataset(name: str):
    n = name.lower()
    if n == "mutag":
        return load_mutag()
    if n in ("proteins", "protein"):
        return load_proteins()
    raise ValueError(f"Unsupported dataset: {name}")


def _train_dataset_qgat(
    dataset_key: str,
    graphs: list,
    labels: list,
    combo_params: Dict,
    device: torch.device,
) -> Dict:
    algo   = combo_params["algorithm"]
    n_q    = combo_params["max_qubits"]
    bar_ds = dataset_key[:3].upper()
    bar_desc = f"{bar_ds}-{algo[:3].upper()}-Q{n_q}"

    splitter = StratifiedKFold(
        n_splits=FIXED_CONFIG["n_folds"],
        shuffle=True,
        random_state=FIXED_CONFIG["seed"],
    )
    fold_results = []
    for fi, (tr_idx, te_idx) in enumerate(splitter.split(graphs, labels)):
        fold_results.append(
            _train_fold(
                train_graphs=[graphs[i] for i in tr_idx],
                test_graphs= [graphs[i] for i in te_idx],
                combo_params=combo_params,
                device=device,
                fold_idx=fi,
                bar_desc=bar_desc,
            )
        )

    return {"summary": _aggregate_folds(fold_results)}


# ── main grid search ──────────────────────────────────────────────────────────

def run_grid_search(
    datasets: Sequence[str],
    output_dir: str = "grid_search_results",
    resume: bool = True,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    grids = build_algorithm_grids()
    total_combos = sum(len(combos) for combos in grids.values())
    print(f"Device: {device}")
    print(f"Total combinations: {total_combos} "
          f"({', '.join(f'{k}: {len(v)}' for k, v in grids.items())})")
    print(f"Datasets: {list(datasets)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(output_dir) / f"qgat_grid_search_{timestamp}"
    base_output.mkdir(parents=True, exist_ok=True)

    results_csv = base_output / "grid_results.csv"
    all_results: List[Dict] = []
    existing_keys: set = set()

    if resume and results_csv.exists():
        with open(results_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = _make_combo_key(row["dataset"], row)
                existing_keys.add(key)
                all_results.append(row)
        print(f"Resuming from {len(existing_keys)} existing results")

    # Pre-load each dataset once to avoid redundant I/O across combos.
    dataset_cache: Dict[str, tuple] = {}
    for dataset_name in datasets:
        dataset_key = normalize_dataset_name(dataset_name)
        if dataset_key not in dataset_cache:
            graphs = _load_dataset(dataset_key)
            labels = [int(g.y.item()) for g in graphs]
            print(
                f"  {dataset_key.upper()}: {len(graphs)} graphs | "
                f"feat_dim={graphs[0].x.size(1)} | classes={set(labels)}"
            )
            dataset_cache[dataset_key] = (graphs, labels)

    csv_exists = results_csv.exists()

    try:
        for dataset_name in datasets:
            dataset_key = normalize_dataset_name(dataset_name)
            graphs, labels = dataset_cache[dataset_key]
            for algo, combos in grids.items():
                for idx, combo_params in enumerate(combos):
                    key = _make_combo_key(dataset_key, combo_params)
                    if key in existing_keys:
                        print(f"  [SKIP] {dataset_key}/{algo} combo {idx + 1}/{len(combos)}")
                        continue

                    set_seed(FIXED_CONFIG["seed"])

                    combo_str = ", ".join(
                        f"{k}={v}" for k, v in combo_params.items() if k != "algorithm"
                    )
                    print(f"\n{'=' * 60}")
                    print(f"[{dataset_key}/{algo} {idx + 1}/{len(combos)}] {combo_str}")
                    print(f"{'=' * 60}")

                    result = _train_dataset_qgat(
                        dataset_key=dataset_key,
                        graphs=graphs,
                        labels=labels,
                        combo_params=combo_params,
                        device=device,
                    )

                    row = {
                        "dataset": dataset_key,
                        "algorithm": algo,
                        "max_qubits": combo_params.get("max_qubits", ""),
                        "use_layer_norm": combo_params.get("use_layer_norm", ""),
                        "accumulate_window": combo_params.get("accumulate_window", ""),
                        "prune_window": combo_params.get("prune_window", ""),
                        "prune_ratio": combo_params.get("prune_ratio", ""),
                        "qdrop_schedule": combo_params.get("qdrop_schedule", ""),
                        "dropout_prob": combo_params.get("dropout_prob", ""),
                        "n_drop_wires": combo_params.get("n_drop_wires", ""),
                        "enable_forward_mask": combo_params.get("enable_forward_mask", ""),
                    }
                    for metric_key, metric_val in result["summary"].items():
                        row[metric_key] = metric_val

                    all_results.append(row)
                    existing_keys.add(key)

                    write_mode = "a" if csv_exists else "w"
                    with open(results_csv, write_mode, newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                        if not csv_exists:
                            writer.writeheader()
                        writer.writerow(row)
                    csv_exists = True

                    print(
                        f"  -> acc={row.get('mean_accuracy', '?'):.4f}"
                        f"±{row.get('std_accuracy', '?'):.4f}, "
                        f"f1={row.get('mean_f1', '?'):.4f}"
                        f"±{row.get('std_f1', '?'):.4f}"
                    )

    except KeyboardInterrupt:
        print("\nInterrupted. Partial results saved to", results_csv)
        return

    full_json = base_output / "grid_results.json"
    with open(full_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("BEST RESULTS")
    print("=" * 70)
    for dataset_key in sorted({normalize_dataset_name(d) for d in datasets}):
        for algo in ["baseline", "pruning", "dropout", "both"]:
            algo_results = [
                r for r in all_results
                if r["dataset"] == dataset_key and r["algorithm"] == algo
            ]
            if not algo_results:
                continue
            best = max(algo_results, key=lambda r: float(r.get("mean_accuracy", 0)))
            print(
                f"\n{dataset_key}/{algo} — "
                f"accuracy={best['mean_accuracy']}±{best['std_accuracy']}, "
                f"f1={best['mean_f1']}±{best['std_f1']}"
            )
            for k in ["max_qubits", "use_layer_norm", "accumulate_window", "prune_window",
                       "prune_ratio", "qdrop_schedule", "dropout_prob", "n_drop_wires",
                       "enable_forward_mask"]:
                if best.get(k, "") != "":
                    print(f"  {k}: {best[k]}")

    print(f"\nResults: {results_csv}")
    print(f"Full JSON: {full_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Grid search over QGAT architecture + Q-Drop hyperparameters"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["mutag", "proteins"],
        help="Datasets to search over (default: mutag proteins)",
    )
    parser.add_argument(
        "--output-dir", default="grid_search_results",
        help="Output directory (default: grid_search_results)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Do not resume from existing results",
    )
    args = parser.parse_args()

    run_grid_search(
        datasets=args.datasets,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )
