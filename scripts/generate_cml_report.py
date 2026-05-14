"""Render the CML training report (Markdown + charts) from ``summary.json``.

This script is the single source of truth for the report that the
``train_quantum_graph.yml`` workflow posts back to GitHub. It was hoisted
out of an inline ``python - <<'PYEOF'`` block so that the same logic can
run either:

* directly on the training runner (when the runner can talk to CML's
  asset-upload backend, e.g. the A6000), or
* on a downstream report-runner that downloaded the training artifact
  from a different machine (e.g. P6000 -> A6000 hand-off).

Environment inputs:

* ``RESULTS_DIR`` (required): path containing ``summary.json`` and the
  ``report_assets/`` subdirectory the script will populate with PNGs.
* ``ALGORITHM`` (optional, default ``baseline``): Q-Drop algorithm name
  for the report header.
* ``REPORT_INCLUDE_INLINE_ASSETS`` (optional, default ``true``): when
  truthy, the script appends ``![alt](./path)`` Markdown references for
  each generated chart so that ``cml comment create`` will upload them
  inline. Disable from machines whose CML asset endpoint is broken.
* GitHub Actions environment variables (``GITHUB_REF_NAME``,
  ``GITHUB_SHA``, ``GITHUB_RUN_ID``, ``GITHUB_REPOSITORY``,
  ``GITHUB_WORKFLOW``) feed the run-overview table.

Outputs ``report.md`` in the current working directory and chart PNGs in
``$RESULTS_DIR/report_assets/``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
PRIMARY_METRICS = ("accuracy", "f1", "roc_auc", "pr_auc")
PREFERRED_DATASET_ORDER = ("mutag", "proteins")


def truthy(value: str | None) -> bool:
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


def format_scalar(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "—"
        return f"{value:.4f}"
    return str(value)


def format_mean_std(mean_value, std_value):
    if mean_value is None or std_value is None:
        return "—"
    if any(math.isnan(v) or math.isinf(v) for v in (mean_value, std_value)):
        return "—"
    return f"{mean_value:.4f} +/- {std_value:.4f}"


def select_best_fold(result):
    folds = result.get("folds", [])
    if not folds:
        return None
    return max(folds, key=lambda fold: fold.get("accuracy", 0.0))


def render_overview_chart(summary, ordered_keys, overview_chart: Path) -> None:
    metric_labels = ["Accuracy", "F1", "ROC AUC", "PR AUC"]
    x_positions = np.arange(len(metric_labels))
    bar_width = 0.35 if len(ordered_keys) > 1 else 0.55
    colors = ["#2E86AB", "#F18F01", "#C73E1D", "#3B8B5A"]

    figure, axis = plt.subplots(figsize=(12, 6))
    for index, dataset_key in enumerate(ordered_keys):
        result = summary[dataset_key]
        stats = result.get("summary", {})
        means = [stats.get(f"mean_{metric}", 0.0) for metric in PRIMARY_METRICS]
        stds = [stats.get(f"std_{metric}", 0.0) for metric in PRIMARY_METRICS]
        offset = (index - (len(ordered_keys) - 1) / 2) * bar_width
        axis.bar(
            x_positions + offset,
            means,
            width=bar_width,
            yerr=stds,
            capsize=4,
            label=result.get("dataset", dataset_key.upper()),
            color=colors[index % len(colors)],
            alpha=0.92,
        )

    axis.set_title("Cross-Dataset Performance Overview", fontsize=14, weight="bold")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(metric_labels)
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.05)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(overview_chart, dpi=200, bbox_inches="tight")
    plt.close(figure)


def render_learning_curves(summary, ordered_keys, curve_chart: Path) -> None:
    figure, axes = plt.subplots(
        len(ordered_keys),
        2,
        figsize=(14, 5 * max(len(ordered_keys), 1)),
        squeeze=False,
    )
    for row_index, dataset_key in enumerate(ordered_keys):
        result = summary[dataset_key]
        dataset_label = result.get("dataset", dataset_key.upper())
        best_fold = select_best_fold(result)
        loss_axis = axes[row_index][0]
        score_axis = axes[row_index][1]

        if best_fold is None:
            loss_axis.set_axis_off()
            score_axis.set_axis_off()
            continue

        train_curve = best_fold.get("train_curve", [])
        val_curve = best_fold.get("val_curve", [])
        train_epochs = [point["epoch"] for point in train_curve]
        val_epochs = [point["epoch"] for point in val_curve]

        if train_curve:
            loss_axis.plot(
                train_epochs,
                [point.get("loss", 0.0) for point in train_curve],
                label="Train Loss",
                color="#2E86AB",
                linewidth=2,
            )
        if val_curve:
            loss_axis.plot(
                val_epochs,
                [point.get("loss", 0.0) for point in val_curve],
                label="Val Loss",
                color="#C73E1D",
                linewidth=2,
            )
        loss_axis.set_title(
            f"{dataset_label} Best Fold #{best_fold['fold']} Loss",
            fontsize=12,
            weight="bold",
        )
        loss_axis.set_xlabel("Epoch")
        loss_axis.set_ylabel("Loss")
        loss_axis.grid(alpha=0.3, linestyle="--")
        loss_axis.legend(frameon=False)

        if train_curve:
            score_axis.plot(
                train_epochs,
                [point.get("accuracy", 0.0) for point in train_curve],
                label="Train Acc",
                color="#2E86AB",
                linewidth=2,
            )
        if val_curve:
            score_axis.plot(
                val_epochs,
                [point.get("accuracy", 0.0) for point in val_curve],
                label="Val Acc",
                color="#F18F01",
                linewidth=2,
            )
            score_axis.plot(
                val_epochs,
                [point.get("f1", 0.0) for point in val_curve],
                label="Val F1",
                color="#3B8B5A",
                linewidth=2,
            )
        score_axis.set_title(
            f"{dataset_label} Best Fold #{best_fold['fold']} Accuracy / F1",
            fontsize=12,
            weight="bold",
        )
        score_axis.set_xlabel("Epoch")
        score_axis.set_ylabel("Score")
        score_axis.set_ylim(0.0, 1.05)
        score_axis.grid(alpha=0.3, linestyle="--")
        score_axis.legend(frameon=False)

    figure.tight_layout()
    figure.savefig(curve_chart, dpi=200, bbox_inches="tight")
    plt.close(figure)


def render_qdrop_progress(summary, ordered_keys, qdrop_chart: Path) -> bool:
    figure, axes = plt.subplots(
        len(ordered_keys),
        2,
        figsize=(14, 4.5 * max(len(ordered_keys), 1)),
        squeeze=False,
    )
    any_data = False
    for row_index, dataset_key in enumerate(ordered_keys):
        result = summary[dataset_key]
        dataset_label = result.get("dataset", dataset_key.upper())
        best_fold = select_best_fold(result)
        phase_axis = axes[row_index][0]
        drop_axis = axes[row_index][1]

        if best_fold is None:
            phase_axis.set_axis_off()
            drop_axis.set_axis_off()
            continue

        qdrop_curve = best_fold.get("qdrop_curve", [])
        if not qdrop_curve:
            phase_axis.set_axis_off()
            drop_axis.set_axis_off()
            continue

        any_data = True
        qdrop_epochs = [point.get("epoch", 0) for point in qdrop_curve]
        prune_ratio = [point.get("prune_ratio", 0.0) for point in qdrop_curve]
        prune_steps = [point.get("pruning_step_count", 0) for point in qdrop_curve]
        prune_phase = [point.get("is_prune_phase", 0) for point in qdrop_curve]
        dropout_enabled = [point.get("dropout_enabled", 0) for point in qdrop_curve]
        dropped_wire_count = [point.get("dropped_wire_count", 0) for point in qdrop_curve]
        active_dropout_layers = [point.get("active_dropout_layers", 0) for point in qdrop_curve]

        phase_axis.plot(qdrop_epochs, prune_ratio, color="#2E86AB", linewidth=2, label="Prune Ratio")
        phase_axis.step(qdrop_epochs, prune_phase, where="mid", color="#C73E1D", linewidth=2, label="Prune Phase")
        phase_axis.plot(qdrop_epochs, prune_steps, color="#3B8B5A", linewidth=2, label="Pruning Steps")
        phase_axis.set_title(
            f"{dataset_label} Best Fold #{best_fold['fold']} Pruning Progress",
            fontsize=12,
            weight="bold",
        )
        phase_axis.set_xlabel("Epoch")
        phase_axis.set_ylabel("Value")
        phase_axis.grid(alpha=0.3, linestyle="--")
        phase_axis.legend(frameon=False)

        drop_axis.step(qdrop_epochs, dropout_enabled, where="mid", color="#F18F01", linewidth=2, label="Dropout Enabled")
        drop_axis.plot(qdrop_epochs, dropped_wire_count, color="#7B2CBF", linewidth=2, label="Dropped Wires")
        drop_axis.plot(qdrop_epochs, active_dropout_layers, color="#00897B", linewidth=2, label="Active Dropout Layers")
        drop_axis.set_title(
            f"{dataset_label} Best Fold #{best_fold['fold']} Dropout Progress",
            fontsize=12,
            weight="bold",
        )
        drop_axis.set_xlabel("Epoch")
        drop_axis.set_ylabel("Count / Flag")
        drop_axis.grid(alpha=0.3, linestyle="--")
        drop_axis.legend(frameon=False)

    if any_data:
        figure.tight_layout()
        figure.savefig(qdrop_chart, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return any_data


def build_report_markdown(
    summary,
    ordered_keys,
    algorithm,
    ref,
    sha,
    run_id,
    run_url,
    workflow_name,
    baseline_path: Path,
) -> list[str]:
    lines: list[str] = [
        "# Quantum Graph Training Report — MUTAG & PROTEINS",
        "",
        f"**Branch:** `{ref}` | **Commit:** `{sha}` | **Run:** [{run_id}]({run_url})",
        f"**Algorithm:** `{algorithm}` | **Model:** QGCN",
        "",
        "## Run Overview",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Workflow | `{workflow_name}` |",
        f"| Branch | `{ref}` |",
        f"| Commit | `{sha}` |",
        f"| Run | [{run_id}]({run_url}) |",
        f"| Algorithm | `{algorithm}` |",
        f"| Model | `QGCN` |",
        f"| Datasets | {', '.join(summary[key].get('dataset', key.upper()) for key in ordered_keys)} |",
        "",
        "## Dataset Overview",
        "",
        "| Dataset | Source | Graphs | Classes | Node Feature Dim | Task |",
        "|---------|--------|-------:|--------:|-----------------:|------|",
    ]

    for dataset_key in ordered_keys:
        result = summary[dataset_key]
        source = result.get("dataset_source", "—")
        source_url = f"https://huggingface.co/datasets/{source}" if source != "—" else "#"
        lines.append(
            f"| {result.get('dataset', dataset_key.upper())} | [{source}]({source_url}) | "
            f"{result.get('n_graphs', '—')} | {result.get('n_classes', '—')} | "
            f"{result.get('node_feature_dim', '—')} | {result.get('task', '—')} |"
        )

    reference_config = summary[ordered_keys[0]].get("config", {}) if ordered_keys else {}
    config_rows = [
        ("Epochs", reference_config.get("epochs")),
        ("Learning rate", reference_config.get("lr")),
        ("Weight decay", reference_config.get("weight_decay")),
        ("Batch size", reference_config.get("batch_size")),
        ("Q-depths", reference_config.get("q_depths")),
        ("Quantum width", reference_config.get("n_qubits")),
        ("Folds", reference_config.get("n_folds")),
        ("Early stop patience", reference_config.get("early_stop_patience")),
        ("Validation frequency", reference_config.get("val_frequency")),
        ("Gradient clip", reference_config.get("grad_clip")),
        ("LR scheduler", reference_config.get("use_scheduler")),
        ("Class weights", reference_config.get("use_class_weights")),
        ("Q-Drop schedule", reference_config.get("qdrop_schedule")),
        ("Dropout probability", reference_config.get("dropout_prob")),
        ("Dropped wires / layer", reference_config.get("n_drop_wires")),
        ("Forward output masking", reference_config.get("enable_forward_mask")),
        ("Quantum lr scale", reference_config.get("quantum_lr_scale")),
        ("Seed", reference_config.get("seed")),
    ]
    if algorithm in {"pruning", "both"}:
        config_rows.extend(
            [
                ("Accumulate window", reference_config.get("accumulate_window")),
                ("Prune window", reference_config.get("prune_window")),
                ("Prune ratio", reference_config.get("prune_ratio")),
            ]
        )

    lines += [
        "",
        "## Shared Training Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for label, value in config_rows:
        lines.append(f"| {label} | {format_scalar(value)} |")

    lines += [
        "",
        "## Aggregate Results",
        "",
        "| Dataset | Accuracy | F1 | ROC AUC | PR AUC | Precision | Recall |",
        "|---------|----------|----|---------|--------|-----------|--------|",
    ]
    for dataset_key in ordered_keys:
        result = summary[dataset_key]
        stats = result.get("summary", {})
        lines.append(
            f"| {result.get('dataset', dataset_key.upper())} | "
            f"{format_mean_std(stats.get('mean_accuracy'), stats.get('std_accuracy'))} | "
            f"{format_mean_std(stats.get('mean_f1'), stats.get('std_f1'))} | "
            f"{format_mean_std(stats.get('mean_roc_auc'), stats.get('std_roc_auc'))} | "
            f"{format_mean_std(stats.get('mean_pr_auc'), stats.get('std_pr_auc'))} | "
            f"{format_mean_std(stats.get('mean_precision'), stats.get('std_precision'))} | "
            f"{format_mean_std(stats.get('mean_recall'), stats.get('std_recall'))} |"
        )

    for dataset_key in ordered_keys:
        result = summary[dataset_key]
        dataset = result.get("dataset", dataset_key.upper())
        cfg = result.get("config", {})
        stats = result.get("summary", {})
        folds = result.get("folds", [])
        best_fold = select_best_fold(result)

        lines += [
            "",
            f"## {dataset}",
            "",
            "### Configuration",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Epochs | {format_scalar(cfg.get('epochs'))} |",
            f"| Learning rate | {format_scalar(cfg.get('lr'))} |",
            f"| Weight decay | {format_scalar(cfg.get('weight_decay'))} |",
            f"| Batch size | {format_scalar(cfg.get('batch_size'))} |",
            f"| Q-depths | {format_scalar(cfg.get('q_depths'))} |",
            f"| Quantum width | {format_scalar(cfg.get('n_qubits'))} |",
            f"| Folds | {format_scalar(cfg.get('n_folds'))} |",
            f"| Early stop patience | {format_scalar(cfg.get('early_stop_patience'))} |",
            f"| Validation frequency | {format_scalar(cfg.get('val_frequency'))} |",
            f"| Gradient clip | {format_scalar(cfg.get('grad_clip'))} |",
            f"| LR scheduler | {format_scalar(cfg.get('use_scheduler'))} |",
            f"| Class weights | {format_scalar(cfg.get('use_class_weights'))} |",
            f"| Quantum lr scale | {format_scalar(cfg.get('quantum_lr_scale'))} |",
            f"| Seed | {format_scalar(cfg.get('seed'))} |",
            "",
            "### Aggregate Results",
            "",
            "| Metric | Mean | Std |",
            "|--------|------|-----|",
        ]
        for metric in METRICS:
            lines.append(
                f"| {metric.upper()} | "
                f"{format_scalar(stats.get(f'mean_{metric}'))} | "
                f"{format_scalar(stats.get(f'std_{metric}'))} |"
            )

        if best_fold is not None:
            lines += [
                "",
                "### Best Fold Snapshot",
                "",
                "| Fold | Test Loss | Accuracy | F1 | Precision | Recall | ROC AUC | PR AUC |",
                "|------|-----------|----------|----|-----------|--------|---------|--------|",
                f"| {best_fold.get('fold', '—')} | {format_scalar(best_fold.get('test_loss'))} | "
                f"{format_scalar(best_fold.get('accuracy'))} | {format_scalar(best_fold.get('f1'))} | "
                f"{format_scalar(best_fold.get('precision'))} | {format_scalar(best_fold.get('recall'))} | "
                f"{format_scalar(best_fold.get('roc_auc'))} | {format_scalar(best_fold.get('pr_auc'))} |",
            ]

        lines += [
            "",
            "### Per-Fold Results",
            "",
            "| Fold | Test Loss | " + " | ".join(metric.upper() for metric in METRICS) + " |",
            "|" + " --- |" * (len(METRICS) + 2),
        ]
        for fold in folds:
            fold_values = " | ".join(format_scalar(fold.get(metric)) for metric in METRICS)
            lines.append(
                f"| {fold['fold']} | {format_scalar(fold.get('test_loss'))} | {fold_values} |"
            )

    if baseline_path.exists():
        with open(baseline_path, encoding="utf-8") as baseline_file:
            baseline = json.load(baseline_file)
        lines += [
            "",
            "## Baseline Comparison",
            "",
            "| Dataset | Metric | Baseline | Current | Delta | Status |",
            "|---------|--------|----------|---------|-------|--------|",
        ]
        for dataset_key in ordered_keys:
            current_stats = summary[dataset_key].get("summary", {})
            baseline_stats = baseline.get(dataset_key, {}).get("summary", {})
            dataset = summary[dataset_key].get("dataset", dataset_key.upper())
            for metric in ("accuracy", "f1", "roc_auc", "pr_auc"):
                current_value = current_stats.get(f"mean_{metric}", float("nan"))
                baseline_value = baseline_stats.get(f"mean_{metric}", float("nan"))
                delta = current_value - baseline_value
                sign = "+" if delta >= 0 else ""
                if delta > 0.005:
                    status = "Improved"
                elif delta < -0.01:
                    status = "Regression"
                else:
                    status = "Stable"
                lines.append(
                    f"| {dataset} | {metric.upper()} | {baseline_value:.4f} | "
                    f"{current_value:.4f} | {sign}{delta:.4f} | {status} |"
                )
    else:
        lines += [
            "",
            "> No baseline found; this run will become the baseline after merge to `main`.",
        ]

    return lines


def main() -> int:
    results_dir_env = os.environ.get("RESULTS_DIR")
    if not results_dir_env:
        print("ERROR: RESULTS_DIR is not set", file=sys.stderr)
        return 2

    results_dir = Path(results_dir_env)
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        print(f"ERROR: summary.json missing at {summary_path}", file=sys.stderr)
        return 2

    assets_dir = results_dir / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    algorithm = os.environ.get("ALGORITHM", "baseline")
    ref = os.environ.get("GITHUB_REF_NAME", "unknown")
    sha = os.environ.get("GITHUB_SHA", "")[:8]
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    workflow_name = os.environ.get("GITHUB_WORKFLOW", "unknown")
    include_inline_assets = truthy(os.environ.get("REPORT_INCLUDE_INLINE_ASSETS"))
    report_path_env = os.environ.get("REPORT_PATH", "report.md")
    report_path = Path(report_path_env)
    baseline_path = Path(os.environ.get("BASELINE_PATH", "quantum_graph_baseline.json"))

    run_url = f"https://github.com/{repo}/actions/runs/{run_id}" if repo and run_id else "#"

    with open(summary_path, encoding="utf-8") as summary_file:
        summary = json.load(summary_file)

    ordered_keys = [key for key in PREFERRED_DATASET_ORDER if key in summary] + [
        key for key in summary if key not in PREFERRED_DATASET_ORDER
    ]

    overview_chart = assets_dir / "quantum_graph_overview.png"
    curve_chart = assets_dir / "quantum_graph_learning_curves.png"
    qdrop_chart = assets_dir / "quantum_graph_qdrop_progress.png"

    qdrop_chart_rendered = False
    if ordered_keys:
        render_overview_chart(summary, ordered_keys, overview_chart)
        render_learning_curves(summary, ordered_keys, curve_chart)
        if algorithm in {"pruning", "dropout", "both"}:
            qdrop_chart_rendered = render_qdrop_progress(summary, ordered_keys, qdrop_chart)

    lines = build_report_markdown(
        summary=summary,
        ordered_keys=ordered_keys,
        algorithm=algorithm,
        ref=ref,
        sha=sha,
        run_id=run_id,
        run_url=run_url,
        workflow_name=workflow_name,
        baseline_path=baseline_path,
    )

    rel_assets = os.path.relpath(assets_dir, report_path.parent if report_path.parent != Path("") else Path("."))
    if include_inline_assets:
        chart_entries = []
        if overview_chart.exists():
            chart_entries.append(("Performance Overview", overview_chart))
        if curve_chart.exists():
            chart_entries.append(("Best-Fold Learning Curves", curve_chart))
        if qdrop_chart.exists() and qdrop_chart_rendered:
            chart_entries.append(("Q-Drop Progress", qdrop_chart))

        if chart_entries:
            lines.append("")
            lines.append("## Visualizations")
            lines.append("")
            for title, chart_path in chart_entries:
                rel_chart = os.path.relpath(chart_path, report_path.parent if report_path.parent != Path("") else Path("."))
                lines.append(f"### {title}")
                lines.append("")
                lines.append(f"![{title}](./{rel_chart})")
                lines.append("")
    else:
        chart_entries = []
        if overview_chart.exists():
            chart_entries.append(("Performance Overview", overview_chart.name))
        if curve_chart.exists():
            chart_entries.append(("Best-Fold Learning Curves", curve_chart.name))
        if qdrop_chart.exists() and qdrop_chart_rendered:
            chart_entries.append(("Q-Drop Progress", qdrop_chart.name))

        if chart_entries:
            lines.append("")
            lines.append("### Visualizations")
            lines.append("")
            lines.append(
                f"Generated charts (download from the [workflow artifact]({run_url})):"
            )
            lines.append("")
            for title, filename in chart_entries:
                lines.append(f"- {title} (`{filename}`)")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path} ({len(lines)} lines).")
    if rel_assets:
        print(f"Charts in: {assets_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
