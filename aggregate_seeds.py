"""Aggregate multi-seed seed-strategy results into mean +/- std over seeds.

For every (variant, dataset) it collects each seed's 10-fold ``summary``
(mean_accuracy, mean_f1, ...) from its metrics.json and reports the mean and
std of those per-seed means ACROSS seeds. Writes a CSV + JSON and prints a
table.

Usage:
    python aggregate_seeds.py [--results-dir training_results/seed_strategy]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

METRICS = ["accuracy", "f1", "roc_auc", "precision", "recall"]


def find_metrics_files(results_dir: Path):
    # layout: <variant>/<dataset>/seed<seed>/**/metrics.json
    for path in results_dir.glob("*/*/seed*/**/metrics.json"):
        parts = path.relative_to(results_dir).parts
        variant, dataset, seed_dir = parts[0], parts[1], parts[2]
        seed = seed_dir.replace("seed", "")
        yield variant, dataset, seed, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="training_results/seed_strategy")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).resolve().parent / results_dir

    # (variant, dataset) -> {seed: summary_dict}
    grouped: dict = defaultdict(dict)
    for variant, dataset, seed, path in find_metrics_files(results_dir):
        try:
            payload = json.loads(path.read_text())
            grouped[(variant, dataset)][seed] = payload["summary"]
        except (json.JSONDecodeError, KeyError):
            print(f"  [warn] skipping unreadable {path}")

    rows = []
    for (variant, dataset), seed_map in sorted(grouped.items()):
        row = {"variant": variant, "dataset": dataset,
               "n_seeds": len(seed_map), "seeds": ",".join(sorted(seed_map))}
        for metric in METRICS:
            per_seed = [s[f"mean_{metric}"] for s in seed_map.values()
                        if f"mean_{metric}" in s]
            if per_seed:
                row[f"{metric}_mean"] = round(mean(per_seed), 4)
                row[f"{metric}_std"] = round(pstdev(per_seed) if len(per_seed) > 1 else 0.0, 4)
            else:
                row[f"{metric}_mean"] = ""
                row[f"{metric}_std"] = ""
        rows.append(row)

    out_csv = results_dir / "seed_summary.csv"
    out_json = results_dir / "seed_summary.json"
    fields = (["variant", "dataset", "n_seeds", "seeds"]
              + [f"{m}_{stat}" for m in METRICS for stat in ("mean", "std")])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2))

    print(f"\n{'variant':<16}{'dataset':<13}{'seeds':<7}{'acc (mean+/-std)':<22}{'f1 (mean+/-std)'}")
    print("-" * 78)
    for r in rows:
        acc = f"{r['accuracy_mean']}+/-{r['accuracy_std']}" if r["accuracy_mean"] != "" else "-"
        f1 = f"{r['f1_mean']}+/-{r['f1_std']}" if r["f1_mean"] != "" else "-"
        print(f"{r['variant']:<16}{r['dataset']:<13}{r['n_seeds']:<7}{acc:<22}{f1}")
    print(f"\nSaved: {out_csv}\n       {out_json}")


if __name__ == "__main__":
    main()
