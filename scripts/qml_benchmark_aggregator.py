"""Aggregate per-cell metrics.json from QML benchmark matrix runs.

Reads artifacts named ``qml-benchmark-<technique>-<algorithm>-<dataset>-<run_id>``,
extracts each cell's ``metrics.json``, and renders four 4x4 Markdown tables
(one per metric: accuracy, f1, roc_auc, pr_auc) with mean +/- std values.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple


TECHNIQUES = ["baseline", "small_angle", "qng", "layerwise"]
ALGORITHMS = ["baseline", "pruning", "dropout", "both"]
METRICS = ["accuracy", "f1", "roc_auc", "pr_auc"]


CELL_RE = re.compile(
    r"^qml-benchmark-(?P<technique>[a-z_]+)-(?P<algorithm>[a-z]+)-"
    r"(?P<dataset>[a-z_]+)-\d+$"
)


def discover_cells(artifacts_dir: Path) -> Dict[Tuple[str, str], dict]:
    """Walk artifact subdirs, parse {technique, algorithm} from each name."""
    out: Dict[Tuple[str, str], dict] = {}
    if not artifacts_dir.exists():
        return out
    for sub in artifacts_dir.iterdir():
        if not sub.is_dir():
            continue
        m = CELL_RE.match(sub.name)
        if not m:
            continue
        metrics_files = list(sub.rglob("metrics.json"))
        if not metrics_files:
            print(f"WARN: no metrics.json in {sub}", flush=True)
            continue
        with open(metrics_files[0]) as f:
            payload = json.load(f)
        out[(m.group("technique"), m.group("algorithm"))] = payload
    return out


def _format_cell(payload: dict | None, metric: str) -> str:
    if payload is None:
        return "—"
    summary = payload.get("summary", {})
    mean = summary.get(f"mean_{metric}")
    std = summary.get(f"std_{metric}")
    if mean is None:
        return "—"
    return f"{float(mean):.4f}±{float(std or 0.0):.4f}"


def render_table(cells: Dict[Tuple[str, str], dict], metric: str) -> str:
    header = "| technique \\ algorithm | " + " | ".join(ALGORITHMS) + " |"
    sep = "|" + "---|" * (len(ALGORITHMS) + 1)
    lines = [f"### {metric}", "", header, sep]
    for tech in TECHNIQUES:
        row = [tech] + [_format_cell(cells.get((tech, algo)), metric) for algo in ALGORITHMS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_report(cells: Dict[Tuple[str, str], dict], dataset: str) -> str:
    total_cells = len(TECHNIQUES) * len(ALGORITHMS)
    lines = [f"# QML Benchmark — {dataset.upper()}", ""]
    lines.append(f"Cells collected: **{len(cells)} / {total_cells}**")
    lines.append("")
    if not cells:
        lines.append("_No cell artifacts found. Check matrix job statuses._")
        return "\n".join(lines)
    for metric in METRICS:
        lines.append(render_table(cells, metric))
    lines.append("---")
    lines.append("Legend: each cell shows ``mean±std`` over CV folds. ``—`` means cell did not produce a metric.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="mutag")
    args = parser.parse_args()

    cells = discover_cells(args.artifacts)
    report = render_report(cells, args.dataset)
    args.output.write_text(report)
    print(f"Wrote {args.output} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
