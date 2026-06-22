"""Architecture visualization for QGCN and QGAT."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ── Colors ──────────────────────────────────────────────────────────────────
C_INPUT    = "#4A90D9"   # blue
C_CLASSIC  = "#7B8794"   # grey  – classical linear/norm layers
C_QUANTUM  = "#8E44AD"   # purple – quantum circuit
C_GRAPH    = "#27AE60"   # green  – graph aggregation
C_POOL     = "#E67E22"   # orange – pooling
C_CLF      = "#C0392B"   # red    – classifier
C_ATTN     = "#D35400"   # dark orange – attention
C_EDGE_CLR = "#2C3E50"

ALPHA_BOX  = 0.88
BOX_STYLE  = dict(boxstyle="round,pad=0.35", linewidth=1.4, alpha=ALPHA_BOX)


def draw_box(ax, cx, cy, w, h, label, color, fontsize=8.5, sublabel=None):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         facecolor=color, edgecolor="white", **BOX_STYLE)
    ax.add_patch(box)
    y_text = cy + (0.07 if sublabel else 0)
    ax.text(cx, y_text, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white")
    if sublabel:
        ax.text(cx, cy - 0.18, sublabel, ha="center", va="center",
                fontsize=6.8, color="white", alpha=0.9)


def arrow(ax, x0, y0, x1, y1, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE_CLR,
                                lw=1.3, mutation_scale=12))
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.07, my, label, fontsize=6.5, color="#555", va="center")


def bracket_label(ax, x, y_top, y_bot, text, color):
    """Vertical bracket on the right side to label a repeating block."""
    pad = 0.08
    ax.plot([x+pad, x+pad+0.08, x+pad+0.08, x+pad],
            [y_top, y_top, y_bot, y_bot],
            color=color, lw=1.5, solid_capstyle="round")
    ax.text(x + pad + 0.16, (y_top+y_bot)/2, text,
            fontsize=7.5, color=color, va="center", fontweight="bold")


# ═══════════════════════════════════════════════════════════════════════════
# QGCN panel
# ═══════════════════════════════════════════════════════════════════════════
def draw_qgcn(ax):
    ax.set_xlim(0, 3.2)
    ax.set_ylim(-0.5, 14.5)
    ax.axis("off")
    ax.set_title("QGCN", fontsize=14, fontweight="bold", pad=10, color="#2C3E50")

    W, H = 2.2, 0.52
    cx = 1.6
    gap = 0.88

    layers = [
        (14.0, "Input  X",              "[N, input_dim]",  C_INPUT),
        (13.1, "Feature Reduction",     "Linear(in→n_q)  orthogonal", C_CLASSIC),
        (12.2, "LayerNorm + tanh×π",    "→ angles ∈ [−π, π]", C_CLASSIC),
        (11.3, "AngleEmbedding",        "RY encoding per qubit", C_QUANTUM),
        (10.4, "BasicEntanglerLayers",  "RY + CNOT ring × L", C_QUANTUM),
        ( 9.5, "Measure ⟨Z⟩",           "[N, n_qubits]",   C_QUANTUM),
        ( 8.6, "GCN Aggregate",         "Σ (1/√di·dj) · hj + bias", C_GRAPH),
        ( 7.7, "LeakyReLU + Residual",  "skip if layer > 0", C_CLASSIC),
    ]

    for i, (y, lbl, sub, col) in enumerate(layers):
        draw_box(ax, cx, y, W, H, lbl, col, sublabel=sub)
        if i < len(layers) - 1:
            arrow(ax, cx, y - H/2, cx, layers[i+1][0] + H/2)

    # repeat bracket for both QGCNConv layers
    bracket_label(ax, cx + W/2, 12.55, 7.44, "× 2 conv layers", C_QUANTUM)

    # separator dashed box
    rect = plt.Rectangle((cx - W/2 - 0.1, 7.44), W + 0.2, 12.55 - 7.44,
                          linewidth=1.3, edgecolor=C_QUANTUM, facecolor="none",
                          linestyle="--", zorder=0, alpha=0.5)
    ax.add_patch(rect)

    # pooling
    pool_y = 6.8
    draw_box(ax, cx, pool_y, W, H, "Global Mean Pool", C_POOL,
             sublabel="[B, n_qubits]  (or multiscale ×3)")
    arrow(ax, cx, 7.7 - H/2, cx, pool_y + H/2)

    # classifier
    clf_y = 5.9
    draw_box(ax, cx, clf_y, W, H, "Linear Classifier", C_CLF,
             sublabel="[B, output_dim]")
    arrow(ax, cx, pool_y - H/2, cx, clf_y + H/2)

    # output
    out_y = 5.0
    draw_box(ax, cx, out_y, W, H, "Logit / Prediction", C_CLF, fontsize=8)
    arrow(ax, cx, clf_y - H/2, cx, out_y + H/2)

    # legend
    legend_items = [
        mpatches.Patch(color=C_INPUT,   label="Input"),
        mpatches.Patch(color=C_CLASSIC, label="Classical"),
        mpatches.Patch(color=C_QUANTUM, label="Quantum"),
        mpatches.Patch(color=C_GRAPH,   label="Graph Agg"),
        mpatches.Patch(color=C_POOL,    label="Pooling"),
        mpatches.Patch(color=C_CLF,     label="Classifier"),
    ]
    ax.legend(handles=legend_items, loc="lower center", fontsize=7,
              ncol=3, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))


# ═══════════════════════════════════════════════════════════════════════════
# QGAT panel
# ═══════════════════════════════════════════════════════════════════════════
def draw_qgat(ax):
    ax.set_xlim(0, 5.5)
    ax.set_ylim(-0.5, 14.5)
    ax.axis("off")
    ax.set_title("QGAT", fontsize=14, fontweight="bold", pad=10, color="#2C3E50")

    W, H = 2.0, 0.52
    gap  = 0.88

    # ── Node path (left column) ──────────────────────────────────────────
    cx_node = 1.5
    node_layers = [
        (14.0, "Input  X",           "[N, input_dim]",         C_INPUT),
        (13.1, "Feature Reduction",  "Linear(in → n_q)",       C_CLASSIC),
        (12.2, "VQC Norm + tanh×π", "angles ∈ [−π, π]",       C_CLASSIC),
        (11.3, "AngleEmbedding",     "RY per qubit",            C_QUANTUM),
        (10.4, "VQC (RY+RZ+CZ)",    "entangling ring × L",     C_QUANTUM),
        ( 9.5, "Measure ⟨Z⟩",        "[N, n_qubits]",           C_QUANTUM),
    ]
    for i, (y, lbl, sub, col) in enumerate(node_layers):
        draw_box(ax, cx_node, y, W, H, lbl, col, sublabel=sub)
        if i < len(node_layers) - 1:
            arrow(ax, cx_node, y - H/2, cx_node, node_layers[i+1][0] + H/2)

    # ── Attention path (right column) ───────────────────────────────────
    cx_attn = 3.9
    attn_layers = [
        (12.2, "Concat (xi ‖ xj)",      "per edge",              C_ATTN),
        (11.3, "Attn Reduction",         "Linear(2q → n_q)",      C_CLASSIC),
        (10.4, "Norm + tanh×π",          "angles ∈ [−π, π]",      C_CLASSIC),
        ( 9.5, "Attn Circuit",           "RY+RZ+CNOT → PauliZ[-1]", C_QUANTUM),
        ( 8.6, "LeakyReLU + Softmax",    "α_ij over neighbors",    C_ATTN),
    ]
    for i, (y, lbl, sub, col) in enumerate(attn_layers):
        draw_box(ax, cx_attn, y, W, H, lbl, col, sublabel=sub)
        if i < len(attn_layers) - 1:
            arrow(ax, cx_attn, y - H/2, cx_attn, attn_layers[i+1][0] + H/2)

    # branch arrow: node path → concat
    arrow(ax, cx_node + W/2, 12.2, cx_attn - W/2, 12.2)
    ax.text(2.7, 12.35, "edges (i,j)", fontsize=6.5, color="#555", ha="center")

    # ── Weighted aggregation ─────────────────────────────────────────────
    agg_y = 7.7
    cx_mid = (cx_node + cx_attn) / 2
    draw_box(ax, cx_mid, agg_y, 2.4, H,
             "Weighted Aggregation", C_GRAPH, sublabel="Σ α_ij · hj + bias")
    arrow(ax, cx_node, 9.5 - H/2, cx_node, agg_y + H/2 + 0.05)
    ax.annotate("", xy=(cx_mid - 1.2 + 0.05, agg_y + H/2),
                xytext=(cx_attn, 8.6 - H/2),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE_CLR,
                                lw=1.3, mutation_scale=12))
    ax.text(cx_attn - 0.1, 8.1, "α_ij", fontsize=7, color=C_ATTN, ha="center")

    # ── Residual + Norm ──────────────────────────────────────────────────
    res_y = 6.8
    draw_box(ax, cx_mid, res_y, 2.4, H,
             "LeakyReLU + Residual", C_CLASSIC, sublabel="skip connection")
    arrow(ax, cx_mid, agg_y - H/2, cx_mid, res_y + H/2)

    # repeat bracket
    bracket_label(ax, cx_attn + W/2, 13.36, 6.54, "× 2 QGATConv", C_QUANTUM)

    # dashed repeat box
    rect = plt.Rectangle((cx_node - W/2 - 0.1, 6.54),
                          (cx_attn + W/2 + 0.1) - (cx_node - W/2 - 0.1),
                          13.36 - 6.54,
                          linewidth=1.3, edgecolor=C_QUANTUM, facecolor="none",
                          linestyle="--", zorder=0, alpha=0.45)
    ax.add_patch(rect)

    # ── Output norm ──────────────────────────────────────────────────────
    norm_y = 5.9
    draw_box(ax, cx_mid, norm_y, 2.4, H,
             "Output LayerNorm", C_CLASSIC, sublabel="[N, n_qubits]")
    arrow(ax, cx_mid, res_y - H/2, cx_mid, norm_y + H/2)

    # ── Multiscale pool ──────────────────────────────────────────────────
    pool_y = 5.0
    draw_box(ax, cx_mid, pool_y, 2.4, H,
             "Multiscale Pool", C_POOL, sublabel="cat(mean, max, add) → [B, 3·n_q]")
    arrow(ax, cx_mid, norm_y - H/2, cx_mid, pool_y + H/2)

    # ── MLP classifier ───────────────────────────────────────────────────
    clf_y = 4.1
    draw_box(ax, cx_mid, clf_y, 2.4, H,
             "MLP Head", C_CLF, sublabel="Linear→LN→ReLU→Dropout × 2")
    arrow(ax, cx_mid, pool_y - H/2, cx_mid, clf_y + H/2)

    # ── Output ───────────────────────────────────────────────────────────
    out_y = 3.2
    draw_box(ax, cx_mid, out_y, 2.4, H,
             "Logit / Prediction", C_CLF, fontsize=8)
    arrow(ax, cx_mid, clf_y - H/2, cx_mid, out_y + H/2)

    # legend
    legend_items = [
        mpatches.Patch(color=C_INPUT,   label="Input"),
        mpatches.Patch(color=C_CLASSIC, label="Classical"),
        mpatches.Patch(color=C_QUANTUM, label="Quantum"),
        mpatches.Patch(color=C_ATTN,    label="Attention"),
        mpatches.Patch(color=C_GRAPH,   label="Graph Agg"),
        mpatches.Patch(color=C_POOL,    label="Pooling"),
        mpatches.Patch(color=C_CLF,     label="Classifier"),
    ]
    ax.legend(handles=legend_items, loc="lower center", fontsize=7,
              ncol=4, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))


# ═══════════════════════════════════════════════════════════════════════════
# Compose
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor("#F4F6F9")

ax_qgcn = fig.add_axes([0.02, 0.05, 0.38, 0.90])
ax_qgat = fig.add_axes([0.44, 0.05, 0.55, 0.90])

ax_qgcn.set_facecolor("#F4F6F9")
ax_qgat.set_facecolor("#F4F6F9")

draw_qgcn(ax_qgcn)
draw_qgat(ax_qgat)

fig.text(0.5, 0.98, "QGCN vs QGAT — Architecture Overview",
         ha="center", va="top", fontsize=16, fontweight="bold", color="#2C3E50")

out = "/home/cislab301b/Khanh/Q-Drop-Integration/qgcn_qgat_architecture.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.show()
