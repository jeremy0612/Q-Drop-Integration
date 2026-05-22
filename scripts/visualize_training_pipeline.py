"""
Visualize the full Q-Drop training pipeline for IMDB-Binary.

Produces: dataset_visualizations/training_pipeline_visualization.pdf
          dataset_visualizations/training_pipeline_visualization.png

Panels
------
A  Overall pipeline flow (data → model → loss → optimiser)
B  OneCycleLR: LR vs step for classical and quantum param groups
C  OneCycleLR phase breakdown (warmup / cos-anneal)
D  Warmup zoom-in for both groups
E  AdamW effective-update decomposition
F  Q-Drop accumulate/prune cycle & wire-drop interaction
G  K-Fold + Early-Stopping timeline
H  Per-epoch training-loop step diagram
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "dataset_visualizations"
OUT_PNG = OUT_DIR / "training_pipeline_visualization.png"
OUT_PDF = OUT_DIR / "training_pipeline_visualization.pdf"

# ── Palette ────────────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d27"
GRID    = "#2a2d3e"
TEXT    = "#dde1f0"
WHITE   = "#ffffff"
BLUE    = "#5b8dee"
RED     = "#e05c6e"
GREEN   = "#50c878"
YELLOW  = "#f0c060"
PURPLE  = "#a78bfa"
ORANGE  = "#fb923c"
CYAN    = "#22d3ee"
PINK    = "#f472b6"


def ax_style(ax, title="", xlabel="", ylabel="", grid=True):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
        sp.set_linewidth(0.8)
    ax.tick_params(colors=TEXT, labelsize=7.5)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if title:
        ax.set_title(title, color=WHITE, fontsize=9.5, fontweight="bold", pad=5)
    if grid:
        ax.grid(color=GRID, linewidth=0.5, alpha=0.7)


def arrow(ax, x0, y0, x1, y1, color=TEXT, lw=1.2, arrowsize=10):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=f"->,head_width=0.3,head_length=0.2",
                        color=color, lw=lw),
    )


# ── Simulate OneCycleLR ────────────────────────────────────────────────────
def simulate_onecycle(
    base_lr: float,
    max_lr: float,
    total_steps: int,
    pct_start: float = 0.1,
    div_factor: float = 10.0,
    final_div_factor: float = 50.0,
) -> np.ndarray:
    """Return per-step LR array by driving a real OneCycleLR instance."""
    param = torch.tensor([0.0], requires_grad=True)
    opt = AdamW([param], lr=base_lr)
    sched = OneCycleLR(
        opt,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )
    lrs = []
    for _ in range(total_steps):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    return np.array(lrs)


# ── Config (matches IMDB-Binary defaults) ─────────────────────────────────
BASE_LR         = 5e-3
QUANTUM_SCALE   = 0.1
CLASSICAL_MAX   = BASE_LR * 5.0         # 0.025
QUANTUM_MAX     = CLASSICAL_MAX * QUANTUM_SCALE  # 2.5e-3
DIV_FACTOR      = 10.0
FINAL_DIV       = 50.0
PCT_START       = 0.1
EPOCHS          = 100
STEPS_PER_EPOCH = 113       # ≈ 900 train graphs / batch_size 8
TOTAL_STEPS     = EPOCHS * STEPS_PER_EPOCH

CLASSICAL_INIT  = CLASSICAL_MAX / DIV_FACTOR       # 2.5e-3
QUANTUM_INIT    = QUANTUM_MAX   / DIV_FACTOR        # 2.5e-4
CLASSICAL_MIN   = CLASSICAL_INIT / FINAL_DIV        # 5e-5
QUANTUM_MIN     = QUANTUM_INIT   / FINAL_DIV        # 5e-6

WARMUP_STEPS    = int(TOTAL_STEPS * PCT_START)      # 10% → 1130 steps
ANNEAL_STEPS    = TOTAL_STEPS - WARMUP_STEPS

print("Simulating OneCycleLR …")
steps = np.arange(TOTAL_STEPS)
classical_lrs = simulate_onecycle(
    BASE_LR, CLASSICAL_MAX, TOTAL_STEPS, PCT_START, DIV_FACTOR, FINAL_DIV)
quantum_lrs = simulate_onecycle(
    BASE_LR * QUANTUM_SCALE, QUANTUM_MAX, TOTAL_STEPS, PCT_START, DIV_FACTOR, FINAL_DIV)

epochs_axis = steps / STEPS_PER_EPOCH

# ── Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 28), facecolor=BG)
gs_top = gridspec.GridSpec(
    4, 3, figure=fig,
    hspace=0.52, wspace=0.38,
    left=0.06, right=0.97,
    top=0.955, bottom=0.04,
)

# ══════════════════════════════════════════════════════════════════════════
# Panel A — Pipeline flow diagram
# ══════════════════════════════════════════════════════════════════════════
ax_A = fig.add_subplot(gs_top[0, :])
ax_A.set_facecolor(PANEL)
for sp in ax_A.spines.values():
    sp.set_edgecolor(GRID)
ax_A.set_xlim(0, 22)
ax_A.set_ylim(0, 3.6)
ax_A.axis("off")
ax_A.set_title(
    "Q-Drop Training Pipeline — IMDB-Binary",
    color=WHITE, fontsize=13, fontweight="bold", pad=8,
)

def box(ax, x, y, w, h, color, label, sub="", fontsize=8.5):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08", linewidth=1.2,
        edgecolor=color, facecolor=color + "30",
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.12 if sub else 0), label,
            color=color, fontsize=fontsize, fontweight="bold",
            ha="center", va="center")
    if sub:
        ax.text(x, y - 0.28, sub, color=TEXT, fontsize=6.8,
                ha="center", va="center", style="italic")

def harrow(ax, x0, x1, y, color=TEXT, label=""):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.15",
                                color=color, lw=1.3))
    if label:
        ax.text((x0+x1)/2, y+0.18, label, color=color, fontsize=6.5, ha="center")

# Row 1: data pipeline
box(ax_A, 1.6, 2.8, 2.5, 0.75, CYAN,    "IMDB-Binary",        "1 000 ego-graphs\n10-Fold CV")
box(ax_A, 4.8, 2.8, 2.5, 0.75, BLUE,    "Degree One-Hot",     "136-dim node features\n(clamped deg ≤ 135)")
box(ax_A, 8.0, 2.8, 2.5, 0.75, GREEN,   "DataLoader",         "batch_size=8\nshuffle=True (train)")
box(ax_A, 11.2,2.8, 2.5, 0.75, PURPLE,  "QGCN",               "QGCNConv × 2\nn_qubits=16")
box(ax_A, 14.4,2.8, 2.5, 0.75, YELLOW,  "Global Mean Pool",   "graph-level embedding\ndim=16")
box(ax_A, 17.6,2.8, 2.5, 0.75, RED,     "Linear Classifier",  "16 → 1\nBCEWithLogitsLoss")
box(ax_A, 20.4,2.8, 1.5, 0.75, ORANGE,  "Loss",               "class-weighted\npos_weight")

harrow(ax_A, 2.85, 3.55, 2.8, CYAN)
harrow(ax_A, 6.05, 6.75, 2.8, BLUE)
harrow(ax_A, 9.25, 9.95, 2.8, GREEN)
harrow(ax_A, 12.45,13.15,2.8, PURPLE)
harrow(ax_A, 15.65,16.35,2.8, YELLOW)
harrow(ax_A, 18.85,19.65,2.8, RED)

# Row 2: backward / optimiser
box(ax_A, 20.4,1.3, 1.5, 0.75, ORANGE,  "Backward",           "autograd")
box(ax_A, 17.0,1.3, 2.5, 0.75, PINK,    "Grad Clip",          "max_norm=1.0\nclip_grad_norm_")
box(ax_A, 13.4,1.3, 2.5, 0.75, BLUE,    "AdamW",              "2 param groups\nweight_decay=1e-3")
box(ax_A, 9.8, 1.3, 2.5, 0.75, GREEN,   "OneCycleLR",         "cos anneal\npct_start=0.1")
box(ax_A, 6.2, 1.3, 2.5, 0.75, PURPLE,  "Q-Drop",             "accumulate/prune\nwire masking")
box(ax_A, 2.6, 1.3, 2.5, 0.75, YELLOW,  "Early Stop",         "patience=15\nbest-state restore")

harrow(ax_A, 20.4, 20.4, 2.42, ORANGE)   # loss → backward (vertical implied by shared x)
ax_A.annotate("", xy=(20.4, 1.68), xytext=(20.4, 2.43),
              arrowprops=dict(arrowstyle="->,head_width=0.25", color=ORANGE, lw=1.3))
harrow(ax_A, 19.65, 18.25, 1.3, PINK)
harrow(ax_A, 16.75, 15.65, 1.3, BLUE)
harrow(ax_A, 12.15, 11.55, 1.3, GREEN)
harrow(ax_A, 8.55, 7.45, 1.3, PURPLE)
harrow(ax_A, 4.95, 3.85, 1.3, YELLOW)

# Labels
ax_A.text(11.0, 3.6, "FORWARD PASS", color=CYAN,   fontsize=7, ha="center", fontweight="bold")
ax_A.text(11.0, 0.78,"BACKWARD PASS", color=ORANGE, fontsize=7, ha="center", fontweight="bold")
ax_A.text(0.5,  2.8, "Data", color=TEXT, fontsize=6.5, rotation=90, va="center")
ax_A.text(0.5,  1.3, "Optim", color=TEXT, fontsize=6.5, rotation=90, va="center")
ax_A.axhline(2.0, color=GRID, lw=0.8, linestyle="--", alpha=0.6)

# ══════════════════════════════════════════════════════════════════════════
# Panel B — Full OneCycleLR curve (both groups)
# ══════════════════════════════════════════════════════════════════════════
ax_B = fig.add_subplot(gs_top[1, :2])
ax_style(ax_B,
    title="OneCycleLR Schedule — Classical vs Quantum Param Groups (100 Epochs)",
    xlabel="Epoch", ylabel="Learning Rate")

ax_B.plot(epochs_axis, classical_lrs, color=BLUE, lw=1.6, label="Classical params")
ax_B.plot(epochs_axis, quantum_lrs,   color=RED,  lw=1.6, label="Quantum params  (×0.1 scale)", linestyle="--")

# Phase boundary
warmup_epoch = WARMUP_STEPS / STEPS_PER_EPOCH
ax_B.axvline(warmup_epoch, color=YELLOW, lw=1.0, linestyle=":", alpha=0.8)
ax_B.text(warmup_epoch + 0.5, CLASSICAL_MAX * 1.01, "warmup\nend\n(10%)", color=YELLOW, fontsize=6.5)

# Key LR annotations — classical
for val, lbl, yoff in [
    (CLASSICAL_INIT, f"init={CLASSICAL_INIT:.2e}", -0.003),
    (CLASSICAL_MAX,  f"peak={CLASSICAL_MAX:.3f}",   0.0015),
    (CLASSICAL_MIN,  f"min={CLASSICAL_MIN:.1e}",    -0.003),
]:
    ax_B.axhline(val, color=BLUE, lw=0.6, linestyle="--", alpha=0.45)
    ax_B.text(101, val + yoff, lbl, color=BLUE, fontsize=6.3, va="center")

# Key LR annotations — quantum
for val, lbl, yoff in [
    (QUANTUM_MAX,  f"peak={QUANTUM_MAX:.2e}", 0.0002),
    (QUANTUM_MIN,  f"min={QUANTUM_MIN:.1e}", -0.0002),
]:
    ax_B.axhline(val, color=RED, lw=0.6, linestyle=":", alpha=0.45)
    ax_B.text(101, val + yoff, lbl, color=RED, fontsize=6.3, va="center")

ax_B.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID, loc="upper right")
ax_B.set_xlim(-1, 108)
ax_B.set_ylim(-0.001, CLASSICAL_MAX * 1.12)

# ══════════════════════════════════════════════════════════════════════════
# Panel C — LR Formula breakdown (text panel)
# ══════════════════════════════════════════════════════════════════════════
ax_C = fig.add_subplot(gs_top[1, 2])
ax_C.set_facecolor(PANEL)
for sp in ax_C.spines.values():
    sp.set_edgecolor(GRID)
ax_C.axis("off")
ax_C.set_title("LR Calculation — Key Values", color=WHITE, fontsize=9.5, fontweight="bold", pad=5)

lines = [
    ("OneCycleLR Parameters", WHITE, 10, True),
    ("", TEXT, 8, False),
    (f"base_lr          = {BASE_LR:.0e}", BLUE, 8.5, False),
    (f"pct_start        = {PCT_START}", YELLOW, 8.5, False),
    (f"div_factor       = {DIV_FACTOR:.0f}", CYAN, 8.5, False),
    (f"final_div_factor = {FINAL_DIV:.0f}", CYAN, 8.5, False),
    (f"anneal_strategy  = 'cos'", GREEN, 8.5, False),
    ("", TEXT, 8, False),
    ("Classical group", BLUE, 9, True),
    (f"  max_lr  = base_lr × 5  = {CLASSICAL_MAX:.4f}", BLUE, 8, False),
    (f"  init_lr = max_lr ÷ 10  = {CLASSICAL_INIT:.4f}", BLUE, 8, False),
    (f"  min_lr  = init_lr ÷ 50 = {CLASSICAL_MIN:.1e}", BLUE, 8, False),
    ("", TEXT, 8, False),
    ("Quantum group", RED, 9, True),
    (f"  scale   = quantum_lr_scale = {QUANTUM_SCALE}", RED, 8, False),
    (f"  max_lr  = {CLASSICAL_MAX:.4f} × {QUANTUM_SCALE} = {QUANTUM_MAX:.4e}", RED, 8, False),
    (f"  init_lr = {QUANTUM_MAX:.2e} ÷ 10  = {QUANTUM_INIT:.2e}", RED, 8, False),
    (f"  min_lr  = {QUANTUM_INIT:.2e} ÷ 50 = {QUANTUM_MIN:.2e}", RED, 8, False),
    ("", TEXT, 8, False),
    ("total_steps", WHITE, 9, True),
    (f"  = epochs × steps_per_epoch", TEXT, 8, False),
    (f"  = {EPOCHS} × ~{STEPS_PER_EPOCH} = {TOTAL_STEPS:,}", YELLOW, 8, False),
    (f"  warmup = {WARMUP_STEPS:,} steps (10%)", YELLOW, 8, False),
    (f"  anneal = {ANNEAL_STEPS:,} steps (90%)", GREEN, 8, False),
]

y = 0.97
for text, color, size, bold in lines:
    ax_C.text(0.03, y, text, color=color, fontsize=size,
              fontweight="bold" if bold else "normal",
              transform=ax_C.transAxes, va="top",
              fontfamily="monospace" if not bold else "sans-serif")
    y -= 0.048 if text else 0.025

# ══════════════════════════════════════════════════════════════════════════
# Panel D — Warmup zoom (step-level, first 15 epochs)
# ══════════════════════════════════════════════════════════════════════════
ax_D = fig.add_subplot(gs_top[2, 0])
ax_style(ax_D,
    title="Warmup Phase (0 → 10%)",
    xlabel="Step", ylabel="Learning Rate")

zoom = int(WARMUP_STEPS * 1.5)
ax_D.plot(steps[:zoom], classical_lrs[:zoom], color=BLUE, lw=1.5, label="Classical")
ax_D.plot(steps[:zoom], quantum_lrs[:zoom],   color=RED,  lw=1.5, linestyle="--", label="Quantum")
ax_D.axvline(WARMUP_STEPS, color=YELLOW, lw=1.0, linestyle=":")
ax_D.fill_betweenx(
    [0, CLASSICAL_MAX * 1.05], 0, WARMUP_STEPS,
    color=YELLOW, alpha=0.07, label="warmup zone",
)

# Annotate warmup linear ramp
mid_warm = WARMUP_STEPS // 2
ax_D.annotate(
    "linear\nwarm-up",
    xy=(mid_warm, classical_lrs[mid_warm]),
    xytext=(mid_warm + 300, classical_lrs[mid_warm] * 0.6),
    color=YELLOW, fontsize=7,
    arrowprops=dict(arrowstyle="->", color=YELLOW, lw=0.8),
)
ax_D.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID)

# ══════════════════════════════════════════════════════════════════════════
# Panel E — Cosine annealing formula visualised
# ══════════════════════════════════════════════════════════════════════════
ax_E = fig.add_subplot(gs_top[2, 1])
ax_style(ax_E,
    title="Cosine Annealing Formula",
    xlabel="Progress in anneal phase (t / T_anneal)", ylabel="Learning Rate")

t = np.linspace(0, 1, 500)
# PyTorch OneCycleLR cos formula: lr = min_lr + 0.5*(max_lr - min_lr)*(1 + cos(π·t))
cos_lr_c = CLASSICAL_MIN + 0.5 * (CLASSICAL_MAX - CLASSICAL_MIN) * (1 + np.cos(np.pi * t))
cos_lr_q = QUANTUM_MIN   + 0.5 * (QUANTUM_MAX   - QUANTUM_MIN)   * (1 + np.cos(np.pi * t))

ax_E.plot(t, cos_lr_c, color=BLUE, lw=2.0, label="Classical")
ax_E.plot(t, cos_lr_q, color=RED,  lw=2.0, linestyle="--", label="Quantum")
ax_E.fill_between(t, cos_lr_c, CLASSICAL_MIN, color=BLUE, alpha=0.12)

# Annotate formula
ax_E.text(0.5, CLASSICAL_MAX * 0.6,
    r"$lr(t) = lr_{min} + \frac{1}{2}(lr_{max}-lr_{min})(1+\cos(\pi t))$",
    color=WHITE, fontsize=7.5, ha="center", va="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL, edgecolor=GRID, alpha=0.9),
)

ax_E.axhline(CLASSICAL_MAX, color=BLUE, lw=0.6, linestyle="--", alpha=0.5)
ax_E.axhline(CLASSICAL_MIN, color=BLUE, lw=0.6, linestyle="--", alpha=0.5)
ax_E.text(1.01, CLASSICAL_MAX, f"{CLASSICAL_MAX:.3f}", color=BLUE, fontsize=6.5, va="center")
ax_E.text(1.01, CLASSICAL_MIN, f"{CLASSICAL_MIN:.1e}", color=BLUE, fontsize=6.5, va="center")
ax_E.set_xlim(-0.02, 1.12)
ax_E.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID)

# ══════════════════════════════════════════════════════════════════════════
# Panel F — AdamW effective update
# ══════════════════════════════════════════════════════════════════════════
ax_F = fig.add_subplot(gs_top[2, 2])
ax_F.set_facecolor(PANEL)
for sp in ax_F.spines.values():
    sp.set_edgecolor(GRID)
ax_F.axis("off")
ax_F.set_title("AdamW Update Rule", color=WHITE, fontsize=9.5, fontweight="bold", pad=5)

update_lines = [
    ("Step t  (per param group):", WHITE, 9, True),
    ("", TEXT, 8, False),
    ("  g_t  = ∇L (+ grad_clip ≤ 1.0)", ORANGE, 8.2, False),
    ("  m_t  = β₁·m_{t-1} + (1-β₁)·g_t", BLUE, 8.2, False),
    ("  v_t  = β₂·v_{t-1} + (1-β₂)·g_t²", BLUE, 8.2, False),
    ("", TEXT, 8, False),
    ("  m̂_t  = m_t / (1 - β₁ᵗ)   [bias corr]", CYAN, 8.2, False),
    ("  v̂_t  = v_t / (1 - β₂ᵗ)   [bias corr]", CYAN, 8.2, False),
    ("", TEXT, 8, False),
    ("  θ_t = θ_{t-1}", WHITE, 8.2, False),
    ("       − lr(t) · m̂_t / (√v̂_t + ε)", YELLOW, 8.2, False),
    ("       − lr(t) · λ · θ_{t-1}", RED, 8.2, False),
    ("          ↑ weight_decay=1e-3", RED, 7.5, False),
    ("", TEXT, 8, False),
    ("Defaults:  β₁=0.9  β₂=0.999  ε=1e-8", TEXT, 7.8, False),
    ("", TEXT, 8, False),
    ("Two lr values from OneCycleLR:", WHITE, 8.5, True),
    (f"  Classical: {CLASSICAL_INIT:.4f} → {CLASSICAL_MAX:.4f} → {CLASSICAL_MIN:.1e}", BLUE, 8, False),
    (f"  Quantum:   {QUANTUM_INIT:.2e} → {QUANTUM_MAX:.2e} → {QUANTUM_MIN:.1e}", RED, 8, False),
]

y = 0.97
for text, color, size, bold in update_lines:
    ax_F.text(0.03, y, text, color=color, fontsize=size,
              fontweight="bold" if bold else "normal",
              transform=ax_F.transAxes, va="top",
              fontfamily="monospace" if not bold else "sans-serif")
    y -= 0.05 if text else 0.025

# ══════════════════════════════════════════════════════════════════════════
# Panel G — Q-Drop accumulate/prune cycle
# ══════════════════════════════════════════════════════════════════════════
ax_G = fig.add_subplot(gs_top[3, :2])
ax_style(ax_G,
    title="Q-Drop Phase Cycle (accumulate_window=10 | prune_window=8 | prune_ratio=0.8)",
    xlabel="Training Step (batches)", ylabel="")

accum_w = 10
prune_w = 8
cycle_len = accum_w + prune_w
n_cycles = 8
total_show = n_cycles * cycle_len

accum_color = GREEN + "40"
prune_color = RED + "40"

for c in range(n_cycles):
    start = c * cycle_len
    # accumulate phase
    ax_G.axvspan(start, start + accum_w, ymin=0, ymax=1,
                 color=accum_color, lw=0)
    ax_G.text(start + accum_w / 2, 0.5, "ACCUM",
              color=GREEN, fontsize=6.5, ha="center", va="center", fontweight="bold")
    # prune phase
    ax_G.axvspan(start + accum_w, start + cycle_len, ymin=0, ymax=1,
                 color=prune_color, lw=0)
    ax_G.text(start + accum_w + prune_w / 2, 0.5, "PRUNE",
              color=RED, fontsize=6.5, ha="center", va="center", fontweight="bold")

# LR overlay (scaled to 0-1 range for visibility, first n_cycles*cycle_len steps)
disp_steps = min(total_show, len(classical_lrs))
lr_norm = classical_lrs[:disp_steps] / CLASSICAL_MAX
ax_G.plot(np.arange(disp_steps), lr_norm, color=BLUE, lw=1.3, alpha=0.85, label="Classical LR (norm)")
ax_G.plot(np.arange(disp_steps), quantum_lrs[:disp_steps] / QUANTUM_MAX,
          color=RED, lw=1.3, alpha=0.7, linestyle="--", label="Quantum LR (norm)")

# Wire-drop indicator (random for illustration)
rng = np.random.default_rng(42)
for c in range(n_cycles):
    start = c * cycle_len + accum_w
    ax_G.scatter(
        [start + rng.integers(0, prune_w)],
        [0.85],
        marker="v", color=ORANGE, s=40, zorder=5,
    )

ax_G.text(total_show * 0.5, 0.92, "▼ = wire-drop event",
          color=ORANGE, fontsize=6.5, ha="center")
ax_G.set_xlim(0, total_show)
ax_G.set_ylim(0, 1.05)
ax_G.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_G.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.0 (norm)"])
ax_G.legend(fontsize=7, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID, loc="lower right")

# ══════════════════════════════════════════════════════════════════════════
# Panel H — K-Fold + Early Stopping timeline
# ══════════════════════════════════════════════════════════════════════════
ax_H = fig.add_subplot(gs_top[3, 2])
ax_H.set_facecolor(PANEL)
for sp in ax_H.spines.values():
    sp.set_edgecolor(GRID)
ax_H.axis("off")
ax_H.set_title("10-Fold CV + Early Stopping", color=WHITE, fontsize=9.5, fontweight="bold", pad=5)

n_folds = 10
bar_h = 0.06
gap = 0.025
fold_h = bar_h + gap

rng2 = np.random.default_rng(7)
stop_epochs = rng2.integers(40, 101, size=n_folds)

for i in range(n_folds):
    y_base = 0.95 - i * (fold_h + 0.01)
    total_w = 0.85
    stop_frac = stop_epochs[i] / 100
    train_w = total_w * stop_frac
    remain_w = total_w - train_w

    # train bar
    rect_t = mpatches.FancyBboxPatch(
        (0.05, y_base), train_w, bar_h,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=BLUE + "80",
    )
    ax_H.add_patch(rect_t)

    # early-stop remainder (if < 100)
    if stop_epochs[i] < 100:
        rect_r = mpatches.FancyBboxPatch(
            (0.05 + train_w, y_base), remain_w, bar_h,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=GRID,
        )
        ax_H.add_patch(rect_r)
        ax_H.scatter([0.05 + train_w], [y_base + bar_h / 2],
                     marker="|", s=60, color=YELLOW, zorder=5, linewidths=2)

    ax_H.text(0.02, y_base + bar_h / 2, f"F{i+1}",
              color=TEXT, fontsize=6.5, va="center", ha="right")
    ax_H.text(0.05 + train_w + 0.02, y_base + bar_h / 2,
              f"ep{stop_epochs[i]}",
              color=YELLOW if stop_epochs[i] < 100 else GREEN,
              fontsize=6, va="center")

ax_H.set_xlim(0, 1)
ax_H.set_ylim(0, 1)

# Legend
from matplotlib.patches import Patch as _Patch
ax_H.legend(
    handles=[
        _Patch(facecolor=BLUE+"80", edgecolor="none", label="Training"),
        _Patch(facecolor=GRID,      edgecolor="none", label="Skipped (early stop)"),
        plt.Line2D([0], [0], marker="|", color=YELLOW, ms=8, lw=0, label="Best val stop"),
    ],
    fontsize=6.5, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID,
            loc="lower right", framealpha=0.8)
ax_H.text(0.5, 0.02,
    "val_frequency=5 | patience=15 | best-state restored",
    color=TEXT, fontsize=6.5, ha="center", style="italic",
    transform=ax_H.transAxes)

# ── Main title ─────────────────────────────────────────────────────────────
fig.suptitle(
    "Q-Drop Quantum GCN — Full Training Pipeline & Learning Rate Schedule",
    color=WHITE, fontsize=15, fontweight="bold", y=0.975,
)

# ── Save ───────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
for path in [OUT_PNG, OUT_PDF]:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved → {path}")

plt.close(fig)
