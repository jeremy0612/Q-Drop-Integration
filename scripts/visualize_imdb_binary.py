"""
Visualize the IMDB-Binary dataset and save a summary PNG.

Produces: dataset_visualizations/imdb_binary_dataset_visualization.png
Updates:  dataset_visualizations/dataset_summary.json
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loader.load_imdb_binary import load_imdb_binary

OUT_DIR = ROOT / "dataset_visualizations"
OUT_PNG = OUT_DIR / "imdb_binary_dataset_visualization.png"
SUMMARY_JSON = OUT_DIR / "dataset_summary.json"
PYG_ROOT = "/tmp/pyg_data"


def _to_nx(g):
    G = nx.Graph()
    G.add_nodes_from(range(g.num_nodes))
    edges = g.edge_index.t().tolist()
    G.add_edges_from(edges)
    return G


def _density(g):
    n = g.num_nodes
    if n < 2:
        return 0.0
    max_edges = n * (n - 1) / 2
    actual = g.edge_index.shape[1] / 2
    return actual / max_edges


def main():
    print("Loading IMDB-Binary …")
    graphs = load_imdb_binary(PYG_ROOT)

    labels = [int(g.y.item()) for g in graphs]
    n_nodes = [g.num_nodes for g in graphs]
    n_edges = [g.edge_index.shape[1] // 2 for g in graphs]
    densities = [_density(g) for g in graphs]

    class_counts = {str(c): labels.count(c) for c in sorted(set(labels))}
    class_names = {0: "Action", 1: "Romance"}

    # Pick one representative sample per class
    samples = {}
    for c in [0, 1]:
        candidates = [g for g in graphs if int(g.y.item()) == c]
        # pick a medium-sized graph
        by_size = sorted(candidates, key=lambda g: g.num_nodes)
        samples[c] = by_size[len(by_size) // 2]

    # ── Layout ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        left=0.06, right=0.97,
        top=0.91, bottom=0.07,
    )

    DARK_BG = "#1a1d27"
    GRID_C  = "#2e3250"
    ACCENT  = ["#5b8dee", "#e05c6e"]     # blue / red
    TEXT_C  = "#dde1f0"
    TITLE_C = "#ffffff"

    def style_ax(ax, title=""):
        ax.set_facecolor(DARK_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
        ax.tick_params(colors=TEXT_C, labelsize=8)
        ax.xaxis.label.set_color(TEXT_C)
        ax.yaxis.label.set_color(TEXT_C)
        if title:
            ax.set_title(title, color=TITLE_C, fontsize=10, fontweight="bold", pad=6)
        ax.grid(color=GRID_C, linewidth=0.5, alpha=0.6)

    # ── 1. Class distribution ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, "Class Distribution")
    bars = ax1.bar(
        [class_names[0], class_names[1]],
        [class_counts["0"], class_counts["1"]],
        color=ACCENT, edgecolor=GRID_C, linewidth=0.8,
    )
    for bar, cnt in zip(bars, [class_counts["0"], class_counts["1"]]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                 str(cnt), ha="center", va="bottom", color=TEXT_C, fontsize=9)
    ax1.set_ylabel("Count")
    ax1.set_ylim(0, max(class_counts.values()) * 1.18)

    # ── 2. Node count histogram ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, "Node Count Distribution")
    ax2.hist(n_nodes, bins=30, color=ACCENT[0], edgecolor=DARK_BG, linewidth=0.4, alpha=0.9)
    ax2.axvline(np.mean(n_nodes), color="#f0c060", linewidth=1.5,
                linestyle="--", label=f"mean={np.mean(n_nodes):.1f}")
    ax2.legend(fontsize=7, labelcolor=TEXT_C, facecolor=DARK_BG, edgecolor=GRID_C)
    ax2.set_xlabel("Nodes")
    ax2.set_ylabel("Graphs")

    # ── 3. Edge count histogram ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, "Edge Count Distribution")
    ax3.hist(n_edges, bins=30, color=ACCENT[1], edgecolor=DARK_BG, linewidth=0.4, alpha=0.9)
    ax3.axvline(np.mean(n_edges), color="#f0c060", linewidth=1.5,
                linestyle="--", label=f"mean={np.mean(n_edges):.1f}")
    ax3.legend(fontsize=7, labelcolor=TEXT_C, facecolor=DARK_BG, edgecolor=GRID_C)
    ax3.set_xlabel("Edges")
    ax3.set_ylabel("Graphs")

    # ── 4. Density histogram ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    style_ax(ax4, "Graph Density Distribution")
    ax4.hist(densities, bins=30, color="#7ecfa8", edgecolor=DARK_BG, linewidth=0.4, alpha=0.9)
    ax4.axvline(np.mean(densities), color="#f0c060", linewidth=1.5,
                linestyle="--", label=f"mean={np.mean(densities):.3f}")
    ax4.legend(fontsize=7, labelcolor=TEXT_C, facecolor=DARK_BG, edgecolor=GRID_C)
    ax4.set_xlabel("Density")
    ax4.set_ylabel("Graphs")

    # ── 5. Degree distribution per class ──────────────────────────────────
    ax5 = fig.add_subplot(gs[1, :2])
    style_ax(ax5, "Node Degree Distribution by Class")
    for c, color in zip([0, 1], ACCENT):
        degs = []
        for g in graphs:
            if int(g.y.item()) == c:
                deg = g.edge_index[0].bincount(minlength=g.num_nodes).float()
                degs.extend(deg.tolist())
        ax5.hist(degs, bins=40, alpha=0.6, color=color,
                 label=class_names[c], edgecolor=DARK_BG, linewidth=0.3)
    ax5.set_xlabel("Degree")
    ax5.set_ylabel("Node Count")
    ax5.legend(fontsize=8, labelcolor=TEXT_C, facecolor=DARK_BG, edgecolor=GRID_C)

    # ── 6. Nodes vs Edges scatter ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2:])
    style_ax(ax6, "Nodes vs Edges per Graph")
    for c, color in zip([0, 1], ACCENT):
        nn = [n_nodes[i] for i, g in enumerate(graphs) if int(g.y.item()) == c]
        ne = [n_edges[i] for i, g in enumerate(graphs) if int(g.y.item()) == c]
        ax6.scatter(nn, ne, c=color, alpha=0.45, s=12, label=class_names[c])
    ax6.set_xlabel("Nodes")
    ax6.set_ylabel("Edges")
    ax6.legend(fontsize=8, labelcolor=TEXT_C, facecolor=DARK_BG, edgecolor=GRID_C)

    # ── 7 & 8. Sample graph drawings ──────────────────────────────────────
    for col, c in enumerate([0, 1]):
        ax = fig.add_subplot(gs[2, col * 2: col * 2 + 2])
        ax.set_facecolor(DARK_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
        ax.set_title(
            f"Sample Graph — {class_names[c]}  "
            f"(n={samples[c].num_nodes}, e={samples[c].edge_index.shape[1]//2})",
            color=TITLE_C, fontsize=10, fontweight="bold", pad=6,
        )

        G = _to_nx(samples[c])
        pos = nx.spring_layout(G, seed=42)
        deg = dict(G.degree())
        node_sizes = [20 + deg[v] * 15 for v in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.35,
                               edge_color=GRID_C, width=0.8)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                               node_color=ACCENT[c], alpha=0.85)
        ax.axis("off")

    # ── Title ─────────────────────────────────────────────────────────────
    fig.suptitle(
        "IMDB-Binary Dataset  ·  1 000 Social Ego-Networks  ·  Binary Classification",
        color=TITLE_C, fontsize=14, fontweight="bold", y=0.97,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {OUT_PNG}")

    # ── Update dataset_summary.json ───────────────────────────────────────
    summary = {}
    if SUMMARY_JSON.exists():
        summary = json.loads(SUMMARY_JSON.read_text())

    summary["imdb_binary"] = {
        "n_graphs": len(graphs),
        "feature_dim": int(graphs[0].x.shape[1]),
        "class_counts": class_counts,
        "nodes_mean": float(np.mean(n_nodes)),
        "nodes_std": float(np.std(n_nodes)),
        "edges_mean": float(np.mean(n_edges)),
        "edges_std": float(np.std(n_edges)),
        "density_mean": float(np.mean(densities)),
        "density_std": float(np.std(densities)),
        "figure": str(OUT_PNG.relative_to(ROOT)),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"Updated → {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
