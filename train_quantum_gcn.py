"""
QGCN grid search: algorithms × datasets × qubits × Q-Drop hyperparams.

Grid axes (per algorithm):
  ALL       : dataset, n_layers, n_qubits
  baseline  : (no Q-Drop params)
  pruning   : prune_ratio, accumulate_window, prune_window
  dropout   : drop_prob, n_drop_wires
  both      : prune_ratio × drop_prob  (windows/wires fixed)

Each cell = n_folds-fold stratified CV.

Results → training_results/qgcn_gridsearch_<timestamp>/
  run_<key>/metrics.json
  results.csv
  summary.json

Usage examples:
  python train_quantum_gcn.py
  python train_quantum_gcn.py --qubit-options 4 8 --q-depths 1 3
  python train_quantum_gcn.py --prune-ratios 0.5 0.8 0.9 --drop-probs 0.3 0.5 0.7
  python train_quantum_gcn.py --epochs 50 --folds 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

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

# ── path setup ────────────────────────────────────────────────────────────────
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data.load_mutag import load_mutag
from data.load_proteins import load_proteins
from models.Quantum_GCN import QGCN
from utils.torch_qdrop import TorchQDropConfig, TorchQDropManager

# QGCN quantum param: layers.X.qc.weights  (from TorchLayer weight_shapes={"weights": ...})
_QGCN_PARAM_PATTERNS = (".qc.weights",)

# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class GridConfig:
    # Primary axes (all algorithms)
    algorithms: Tuple[str, ...]     = ("baseline", "pruning", "dropout", "both")
    datasets: Tuple[str, ...]       = ("MUTAG", "PROTEINS")
    q_depths: Tuple[int, ...]       = (1,)
    qubit_options: Tuple[int, ...]  = (4, 8)

    # Q-Drop param grids (relevant subset applied per algorithm)
    prune_ratios: Tuple[float, ...] = (0.5, 0.8, 0.9)      # pruning, both
    drop_probs: Tuple[float, ...]   = (0.3, 0.5, 0.7)      # dropout, both
    accumulate_windows: Tuple[int, ...] = (10,)             # pruning, both
    prune_windows: Tuple[int, ...]  = (8,)                  # pruning, both
    n_drop_wires_options: Tuple[int, ...] = (1,)            # dropout, both

    # Training hyperparams (fixed)
    epochs: int             = 100
    lr: float               = 5e-3
    weight_decay: float     = 1e-3
    batch_size: int         = 32
    n_folds: int            = 10
    early_stop_patience: int = 15
    val_frequency: int      = 5
    grad_clip: float        = 1.0
    seed: int               = 42
    output_dir: str         = "training_results"


# ── grid generation ───────────────────────────────────────────────────────────

def _qdrop_combos(algo: str, cfg: GridConfig) -> List[Dict]:
    if algo == "baseline":
        return [{"prune_ratio": None, "drop_prob": None,
                 "accumulate_window": None, "prune_window": None, "n_drop_wires": None}]

    if algo == "pruning":
        return [
            {"prune_ratio": pr, "drop_prob": None,
             "accumulate_window": aw, "prune_window": pw, "n_drop_wires": None}
            for pr, aw, pw in product(cfg.prune_ratios, cfg.accumulate_windows, cfg.prune_windows)
        ]

    if algo == "dropout":
        return [
            {"prune_ratio": None, "drop_prob": dp,
             "accumulate_window": None, "prune_window": None, "n_drop_wires": ndw}
            for dp, ndw in product(cfg.drop_probs, cfg.n_drop_wires_options)
        ]

    # both
    return [
        {"prune_ratio": pr, "drop_prob": dp,
         "accumulate_window": aw, "prune_window": pw, "n_drop_wires": ndw}
        for pr, dp, aw, pw, ndw in product(
            cfg.prune_ratios, cfg.drop_probs,
            cfg.accumulate_windows, cfg.prune_windows, cfg.n_drop_wires_options,
        )
    ]


def _run_key(algo: str, ds: str, depth: int, qubits: int, qd: Dict) -> str:
    key = f"{algo}_{ds}_L{depth}_Q{qubits}"
    if algo in ("pruning", "both") and qd["prune_ratio"] is not None:
        key += f"_PR{qd['prune_ratio']}"
    if algo in ("pruning", "both") and qd["accumulate_window"] is not None:
        key += f"_AW{qd['accumulate_window']}_PW{qd['prune_window']}"
    if algo in ("dropout", "both") and qd["drop_prob"] is not None:
        key += f"_DP{qd['drop_prob']}"
    if algo in ("dropout", "both") and qd["n_drop_wires"] is not None:
        key += f"_NW{qd['n_drop_wires']}"
    return key


def generate_grid(cfg: GridConfig) -> List[Dict]:
    cells = []
    for algo, ds, depth, qubits in product(
        cfg.algorithms, cfg.datasets, cfg.q_depths, cfg.qubit_options
    ):
        for qd in _qdrop_combos(algo, cfg):
            cells.append({
                "algorithm": algo,
                "dataset": ds,
                "n_layers": depth,
                "n_qubits": qubits,
                **qd,
            })
    return cells


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(name: str):
    n = name.lower()
    if n == "mutag":
        return load_mutag()
    if n in ("proteins", "protein"):
        return load_proteins()
    raise ValueError(f"Unknown dataset: {name}")


def pos_weight_tensor(train_graphs, device: torch.device) -> torch.Tensor:
    labels = [int(g.y.item()) for g in train_graphs]
    c = Counter(labels)
    w = c.get(0, 1) / max(c.get(1, 1), 1)
    return torch.tensor([w], dtype=torch.float, device=device)


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
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


def aggregate(fold_results: List[Dict]) -> Dict[str, float]:
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in fold_results]
        out[f"mean_{k}"] = float(np.mean(vals))
        out[f"std_{k}"]  = float(np.std(vals))
    return out


# ── training loop ─────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int):
        self.patience   = patience
        self.best_score = None
        self.counter    = 0
        self.best_state = None

    def step(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter    = 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    scheduler=None,
    grad_clip: float = 1.0,
    qdrop: TorchQDropManager = None,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch  = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            if logits.dim() > 1 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            target = batch.y.float()
            loss   = criterion(logits, target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if qdrop is not None:
                    qdrop.apply()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
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
    train_graphs,
    test_graphs,
    cell: Dict,
    cfg: GridConfig,
    device: torch.device,
    fold_idx: int,
    bar_desc: str,
) -> Dict:
    algo     = cell["algorithm"]
    n_layers = cell["n_layers"]
    n_qubits = cell["n_qubits"]

    train_loader = DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=cfg.batch_size, shuffle=False)

    model = QGCN(
        input_dims=train_graphs[0].x.size(1),
        q_depths=[n_layers],
        output_dims=1,
        max_qubits=n_qubits,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor(train_graphs, device))
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = OneCycleLR(
        optimizer, max_lr=cfg.lr * 5,
        total_steps=max(len(train_loader) * cfg.epochs, 1),
        pct_start=0.1, anneal_strategy="cos",
        div_factor=10.0, final_div_factor=50.0,
    )

    qdrop_cfg = TorchQDropConfig(
        algorithm=algo,
        accumulate_window=cell["accumulate_window"] or 10,
        prune_window=cell["prune_window"] or 8,
        prune_ratio=cell["prune_ratio"] or 0.8,
        drop_prob=cell["drop_prob"] or 0.5,
        n_drop_wires=cell["n_drop_wires"] or 1,
        quantum_param_patterns=_QGCN_PARAM_PATTERNS,
    )
    qdrop = TorchQDropManager(model=model, config=qdrop_cfg)

    stopper    = EarlyStopping(cfg.early_stop_patience)
    train_hist = []
    val_hist   = []

    for epoch in tqdm(range(1, cfg.epochs + 1), leave=False, desc=f"{bar_desc}-F{fold_idx+1}"):
        tr_loss, tr_m = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, scheduler=scheduler,
            grad_clip=cfg.grad_clip, qdrop=qdrop,
        )
        train_hist.append({"epoch": epoch, "loss": tr_loss, **tr_m})

        if epoch % cfg.val_frequency != 0 and epoch != cfg.epochs:
            continue
        val_loss, val_m = run_epoch(model, test_loader, criterion, device)
        val_hist.append({"epoch": epoch, "loss": val_loss, **val_m})
        if stopper.step(val_m["accuracy"], model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    _, test_m = run_epoch(model, test_loader, criterion, device)
    print(
        f"    Fold {fold_idx+1}: "
        f"acc={test_m['accuracy']:.4f}  f1={test_m['f1']:.4f}  roc={test_m['roc_auc']:.4f}"
    )
    return {"fold": fold_idx + 1, **test_m, "train_curve": train_hist, "val_curve": val_hist}


def run_cell(
    cell: Dict,
    graphs,
    labels: List[int],
    cfg: GridConfig,
    device: torch.device,
    out_dir: Path,
    cell_idx: int,
    total_cells: int,
) -> Dict:
    algo    = cell["algorithm"]
    ds      = cell["dataset"]
    depth   = cell["n_layers"]
    qubits  = cell["n_qubits"]
    run_key = _run_key(algo, ds, depth, qubits, cell)
    short   = f"{ds[:3].upper()}-{algo[:3].upper()}-L{depth}Q{qubits}"

    print(f"\n[{cell_idx}/{total_cells}]  {run_key}")
    if algo != "baseline":
        qdrop_info = {k: v for k, v in cell.items()
                      if k in ("prune_ratio", "drop_prob", "accumulate_window",
                                "prune_window", "n_drop_wires") and v is not None}
        print(f"  Q-Drop params: {qdrop_info}")

    splitter = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold_results = []
    for fi, (tr_idx, te_idx) in enumerate(splitter.split(graphs, labels)):
        fold_results.append(
            train_fold(
                train_graphs=[graphs[i] for i in tr_idx],
                test_graphs= [graphs[i] for i in te_idx],
                cell=cell,
                cfg=cfg,
                device=device,
                fold_idx=fi,
                bar_desc=short,
            )
        )

    summary = aggregate(fold_results)
    print(
        f"  → acc={summary['mean_accuracy']:.4f}±{summary['std_accuracy']:.4f}"
        f"  f1={summary['mean_f1']:.4f}  roc={summary['mean_roc_auc']:.4f}"
    )

    run_dir = out_dir / f"run_{run_key}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_key": run_key,
        **{k: v for k, v in cell.items()},
        "summary": summary,
        "folds": [{k: v for k, v in r.items() if k not in ("train_curve", "val_curve")}
                  for r in fold_results],
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    return {"run_key": run_key, **cell, **summary}


# ── CLI / main ────────────────────────────────────────────────────────────────

def parse_args() -> GridConfig:
    p = argparse.ArgumentParser(description="QGCN grid search — algorithms × qubits × Q-Drop params")
    p.add_argument("--algorithms", nargs="+",
                   default=["baseline", "pruning", "dropout", "both"],
                   choices=["baseline", "pruning", "dropout", "both"])
    p.add_argument("--datasets",  nargs="+", default=["MUTAG", "PROTEINS"])
    p.add_argument("--q-depths",  nargs="+", type=int, default=[1])
    p.add_argument("--qubit-options", nargs="+", type=int, default=[4, 8],
                   help="Max qubits per QGCN layer (snapped to nearest power of 2: 2/4/8/16)")

    # Q-Drop param grids
    p.add_argument("--prune-ratios", nargs="+", type=float, default=[0.5, 0.8, 0.9])
    p.add_argument("--drop-probs",   nargs="+", type=float, default=[0.3, 0.5, 0.7])
    p.add_argument("--accumulate-windows", nargs="+", type=int, default=[10])
    p.add_argument("--prune-windows",      nargs="+", type=int, default=[8])
    p.add_argument("--n-drop-wires-options", nargs="+", type=int, default=[1])

    # Training
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=5e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--folds",      type=int,   default=10)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--val-frequency", type=int, default=5)
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--output-dir", type=str,   default="training_results")
    args = p.parse_args()

    return GridConfig(
        algorithms=tuple(args.algorithms),
        datasets=tuple(d.upper() for d in args.datasets),
        q_depths=tuple(args.q_depths),
        qubit_options=tuple(args.qubit_options),
        prune_ratios=tuple(args.prune_ratios),
        drop_probs=tuple(args.drop_probs),
        accumulate_windows=tuple(args.accumulate_windows),
        prune_windows=tuple(args.prune_windows),
        n_drop_wires_options=tuple(args.n_drop_wires_options),
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        n_folds=args.folds,
        early_stop_patience=args.early_stop_patience,
        val_frequency=args.val_frequency,
        grad_clip=args.grad_clip,
        seed=args.seed,
        output_dir=args.output_dir,
    )


def main():
    cfg    = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / f"qgcn_gridsearch_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = generate_grid(cfg)

    from collections import Counter as _Counter
    algo_counts = _Counter(c["algorithm"] for c in grid)

    print("=" * 72)
    print("QGCN GRID SEARCH  —  algorithms × qubits × Q-Drop params")
    print("=" * 72)
    print(f"Datasets        : {list(cfg.datasets)}")
    print(f"Layer depths    : {list(cfg.q_depths)}")
    print(f"Qubit options   : {list(cfg.qubit_options)}  (snapped to power of 2)")
    print(f"── Q-Drop param grids ──────────────────────────────────")
    print(f"  prune_ratios        (pruning/both)  : {list(cfg.prune_ratios)}")
    print(f"  accumulate_windows  (pruning/both)  : {list(cfg.accumulate_windows)}")
    print(f"  prune_windows       (pruning/both)  : {list(cfg.prune_windows)}")
    print(f"  drop_probs          (dropout/both)  : {list(cfg.drop_probs)}")
    print(f"  n_drop_wires        (dropout/both)  : {list(cfg.n_drop_wires_options)}")
    print(f"── Cells per algorithm ─────────────────────────────────")
    for algo in cfg.algorithms:
        print(f"  {algo:<10}: {algo_counts[algo]} cells")
    print(f"  TOTAL     : {len(grid)} cells  ×  {cfg.n_folds} folds  =  {len(grid)*cfg.n_folds} runs")
    print(f"Epochs / folds  : {cfg.epochs} / {cfg.n_folds}")
    print(f"Device          : {device}")
    print(f"Output          : {out_dir.resolve()}")
    print("=" * 72)

    # pre-load datasets once
    dataset_cache: Dict[str, tuple] = {}
    for ds in cfg.datasets:
        graphs = load_dataset(ds)
        labels = [int(g.y.item()) for g in graphs]
        print(f"  {ds}: {len(graphs)} graphs | feat_dim={graphs[0].x.size(1)} | classes={set(labels)}")
        dataset_cache[ds] = (graphs, labels)

    all_rows: List[Dict] = []
    for idx, cell in enumerate(grid, 1):
        graphs, labels = dataset_cache[cell["dataset"]]
        row = run_cell(
            cell=cell,
            graphs=graphs,
            labels=labels,
            cfg=cfg,
            device=device,
            out_dir=out_dir,
            cell_idx=idx,
            total_cells=len(grid),
        )
        all_rows.append(row)

        # flush results.csv after every cell
        csv_path = out_dir / "results.csv"
        flat_keys = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat_keys)
            writer.writeheader()
            writer.writerows(all_rows)

    # summary.json
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"config": asdict(cfg), "results": all_rows}, f, indent=2)

    print(f"\nDone. Results saved to {out_dir.resolve()}")
    print(f"  results.csv   : {csv_path}")
    print(f"  summary.json  : {summary_path}")


if __name__ == "__main__":
    main()
