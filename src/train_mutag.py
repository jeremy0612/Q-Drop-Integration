"""MUTAG experiment wrapper over the shared Torch graph training core."""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from training.graph_training import build_train_parser, config_from_args, set_seed, train_dataset


def build_parser():
    repo_root = str(Path(__file__).resolve().parents[1])
    return build_train_parser(
        description="Train QGCN on MUTAG",
        default_datasets=["mutag"],
        default_batch_size=16,
        default_weight_decay=0.0,
        default_grad_clip=0.0,
        default_use_scheduler=False,
        default_use_class_weights=False,
        default_output_dir=repo_root,
    )


def plot_results(fold_results, output_path: Path):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    fold_accuracies = [result["accuracy"] for result in fold_results]
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    axes[0].bar(range(1, len(fold_accuracies) + 1), fold_accuracies, color="#4C72B0", edgecolor="black")
    axes[0].axhline(mean_accuracy, color="red", linestyle="--", label=f"Mean={mean_accuracy:.3f}±{std_accuracy:.3f}")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("10-Fold CV Accuracy per Fold")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    for result in fold_results:
        train_accuracy = [point["accuracy"] for point in result["train_curve"]]
        axes[1].plot(train_accuracy, alpha=0.4, linewidth=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Train Accuracy")
    axes[1].set_title("Training Accuracy (All Folds)")
    axes[1].grid(alpha=0.3)

    metrics = ["accuracy", "precision", "recall", "f1"]
    means = [np.mean([result[metric] for result in fold_results]) for metric in metrics]
    stds = [np.std([result[metric] for result in fold_results]) for metric in metrics]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = axes[2].bar(metrics, means, yerr=stds, color=colors, edgecolor="black", capsize=5)
    axes[2].set_ylim(0, 1.1)
    axes[2].set_ylabel("Score")
    axes[2].set_title("Mean Metrics (±std) over 10 Folds")
    axes[2].grid(axis="y", alpha=0.3)
    for bar, mean in zip(bars, means):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.suptitle(f"QGCN on MUTAG — {mean_accuracy:.3f}±{std_accuracy:.3f} accuracy", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[+] Plot saved to {output_path}")


def build_legacy_metrics(result_payload, config):
    optimizer_name = "AdamW" if config.weight_decay > 0 else "Adam"
    return {
        "timestamp": result_payload["timestamp"],
        "dataset": result_payload["dataset"],
        "dataset_source": result_payload["dataset_source"],
        "n_graphs": result_payload["n_graphs"],
        "n_classes": result_payload["n_classes"],
        "node_feature_dim": result_payload["node_feature_dim"],
        "task": result_payload["task"],
        "model": result_payload["model"],
        "hyperparameters": {
            "epochs": config.epochs,
            "learning_rate": config.lr,
            "batch_size": config.batch_size,
            "q_depths": list(config.q_depths),
            "n_folds": config.n_folds,
            "early_stop_patience": config.early_stop_patience,
            "val_frequency": config.val_frequency,
            "optimizer": optimizer_name,
            "loss": "BCEWithLogitsLoss",
            "activation": "LeakyReLU(0.2)",
            "random_seed": config.seed,
            "algorithm": config.algorithm,
        },
        "n_folds": config.n_folds,
        "mean_accuracy": float(result_payload["summary"]["mean_accuracy"]),
        "std_accuracy": float(result_payload["summary"]["std_accuracy"]),
        "mean_f1": float(result_payload["summary"]["mean_f1"]),
        "per_fold": [
            {
                key: float(value) if isinstance(value, (float, np.floating)) else value
                for key, value in result.items()
                if key not in ("train_curve", "val_curve")
            }
            for result in result_payload["folds"]
        ],
    }


def main():
    config = config_from_args(build_parser().parse_args())
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print("Loading MUTAG from HuggingFace...")

    result_payload = train_dataset("mutag", config, device=device)
    legacy_metrics = build_legacy_metrics(result_payload, config)

    metrics_path = output_dir / "mutag_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(legacy_metrics, metrics_file, indent=2)
    print(f"[+] Metrics saved to {metrics_path}")

    plot_results(result_payload["folds"], output_dir / "mutag_results.png")


if __name__ == "__main__":
    main()
