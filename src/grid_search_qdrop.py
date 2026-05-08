"""Exhaustive grid search over Q-Drop hyperparameters for MUTAG/PROTEINS datasets.

Usage:
    python grid_search_qdrop.py --datasets mutag proteins --output-dir grid_search_results
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from training.graph_training import (
    GraphTrainConfig,
    normalize_dataset_name,
    set_seed,
    train_dataset,
)

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
    "weight_decay": 0.0,
    "batch_size": 16,
    "q_depths": (1, 1),
    "n_qubits": None,
    "n_folds": 10,
    "early_stop_patience": 15,
    "val_frequency": 5,
    "grad_clip": 0.0,
    "use_scheduler": False,
    "use_class_weights": False,
    "seed": 42,
}

CSV_FIELDS = [
    "dataset", "algorithm",
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

    pruning_keys = list(PRUNING_GRID.keys())
    dropout_keys = list(DROPOUT_GRID.keys())
    all_qdrop_keys = pruning_keys + dropout_keys

    grids["pruning"] = []
    for values in product(*PRUNING_GRID.values()):
        combo = dict(zip(pruning_keys, values))
        combo["algorithm"] = "pruning"
        grids["pruning"].append(combo)

    grids["dropout"] = []
    for values in product(*DROPOUT_GRID.values()):
        combo = dict(zip(dropout_keys, values))
        combo["algorithm"] = "dropout"
        grids["dropout"].append(combo)

    grids["both"] = []
    for values in product(*PRUNING_GRID.values(), *DROPOUT_GRID.values()):
        combo = dict(zip(all_qdrop_keys, values))
        combo["algorithm"] = "both"
        grids["both"].append(combo)

    return grids


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
    base_output = Path(output_dir) / f"grid_search_{timestamp}"
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

    csv_exists = results_csv.exists()

    try:
        for dataset_name in datasets:
            dataset_key = normalize_dataset_name(dataset_name)
            for algo, combos in grids.items():
                for idx, combo_params in enumerate(combos):
                    key = _make_combo_key(dataset_key, combo_params)
                    if key in existing_keys:
                        print(f"  [SKIP] {dataset_key}/{algo} "
                              f"combo {idx + 1}/{len(combos)}")
                        continue

                    set_seed(FIXED_CONFIG["seed"])

                    config = GraphTrainConfig(
                        datasets=[dataset_name],
                        algorithm=algo,
                        epochs=FIXED_CONFIG["epochs"],
                        lr=FIXED_CONFIG["lr"],
                        weight_decay=FIXED_CONFIG["weight_decay"],
                        batch_size=FIXED_CONFIG["batch_size"],
                        q_depths=FIXED_CONFIG["q_depths"],
                        n_qubits=FIXED_CONFIG["n_qubits"],
                        n_folds=FIXED_CONFIG["n_folds"],
                        early_stop_patience=FIXED_CONFIG["early_stop_patience"],
                        val_frequency=FIXED_CONFIG["val_frequency"],
                        grad_clip=FIXED_CONFIG["grad_clip"],
                        use_scheduler=FIXED_CONFIG["use_scheduler"],
                        use_class_weights=FIXED_CONFIG["use_class_weights"],
                        accumulate_window=combo_params.get("accumulate_window", 10),
                        prune_window=combo_params.get("prune_window", 8),
                        prune_ratio=combo_params.get("prune_ratio", 0.8),
                        qdrop_schedule=combo_params.get("qdrop_schedule", True),
                        dropout_prob=combo_params.get("dropout_prob", 0.5),
                        n_drop_wires=combo_params.get("n_drop_wires", 1),
                        enable_forward_mask=combo_params.get("enable_forward_mask", True),
                        output_dir=str(base_output),
                        seed=FIXED_CONFIG["seed"],
                    )

                    qdrop_str = ", ".join(
                        f"{k}={v}" for k, v in combo_params.items() if k != "algorithm"
                    )
                    print(f"\n{'=' * 60}")
                    print(f"[{dataset_key}/{algo} {idx + 1}/{len(combos)}] {qdrop_str}")
                    print(f"{'=' * 60}")

                    result = train_dataset(
                        dataset_name=dataset_name,
                        config=config,
                        device=device,
                        base_output=None,
                    )

                    row = {
                        "dataset": dataset_key,
                        "algorithm": algo,
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
        for algo in ["pruning", "dropout", "both"]:
            algo_results = [
                r for r in all_results
                if r["dataset"] == dataset_key and r["algorithm"] == algo
            ]
            if not algo_results:
                continue
            best = max(algo_results, key=lambda r: float(r.get("mean_accuracy", 0)))
            print(f"\n{dataset_key}/{algo} — accuracy={best['mean_accuracy']}"
                  f"±{best['std_accuracy']}, f1={best['mean_f1']}±{best['std_f1']}")
            for k in ["accumulate_window", "prune_window", "prune_ratio",
                       "qdrop_schedule", "dropout_prob", "n_drop_wires",
                       "enable_forward_mask"]:
                if best.get(k, "") != "":
                    print(f"  {k}: {best[k]}")

    print(f"\nResults: {results_csv}")
    print(f"Full JSON: {full_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Grid search over Q-Drop hyperparameters"
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
