"""
Clear explainer: AdamW + OneCycleLR with two param groups (classical vs quantum).

Produces: dataset_visualizations/lr_explainer.png
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "dataset_visualizations"
OUT     = OUT_DIR / "lr_explainer.png"

# ── palette ────────────────────────────────────────────────────────────────
BG    = "#0d0f1a"
PANEL = "#13172a"
CARD  = "#1c2038"
GRID  = "#252945"
TEXT  = "#c8cde8"
DIM   = "#6b7194"
WHITE = "#ffffff"
BLUE  = "#4f9eff"   # classical
RED   = "#ff5f7e"   # quantum
YELL  = "#ffd166"
GREEN = "#06d6a0"
PURP  = "#b388ff"
ORNG  = "#ff9f43"

# ── helpers ────────────────────────────────────────────────────────────────
def card(ax, x, y, w, h, color, title, lines, fontsize=8):
    rect = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.04", lw=1.4,
        edgecolor=color, facecolor=color+"18")
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.04, title,
            color=color, fontsize=fontsize+0.5, fontweight="bold",
            ha="center", va="top")
    for i, (txt, col) in enumerate(lines):
        ax.text(x + 0.04, y + h - 0.14 - i*0.115, txt,
                color=col, fontsize=fontsize - 0.5,
                va="top", fontfamily="monospace")

def hline(ax, x0, x1, y, color=TEXT, lw=1.5):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle="->,head_width=0.22,head_length=0.015",
                        color=color, lw=lw))

def ax_base(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID); sp.set_linewidth(0.8)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if title:  ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5)
    ax.grid(color=GRID, lw=0.5, alpha=0.7)

def simulate_lr(base, peak, steps, pct=0.1, div=10., fdiv=50.):
    p = torch.tensor([0.], requires_grad=True)
    opt = AdamW([p], lr=base)
    sch = OneCycleLR(opt, max_lr=peak, total_steps=steps,
                     pct_start=pct, anneal_strategy="cos",
                     div_factor=div, final_div_factor=fdiv)
    lrs = []
    for _ in range(steps):
        lrs.append(sch.get_last_lr()[0])
        opt.step(); sch.step()
    return np.array(lrs)

# ── config ─────────────────────────────────────────────────────────────────
BASE_C = 5e-3;  PEAK_C = 2.5e-2;  INIT_C = 2.5e-3;  MIN_C = 5e-5
BASE_Q = 5e-4;  PEAK_Q = 2.5e-3;  INIT_Q = 2.5e-4;  MIN_Q = 5e-6
EPOCHS = 100;   SPE    = 113;      STEPS  = EPOCHS * SPE
WARM   = int(STEPS * 0.1)

print("Simulating …")
steps_arr  = np.arange(STEPS)
epochs_arr = steps_arr / SPE
lrs_c = simulate_lr(BASE_C, PEAK_C, STEPS)
lrs_q = simulate_lr(BASE_Q, PEAK_Q, STEPS)

# ══════════════════════════════════════════════════════════════════════════
# Figure layout  (4 rows × 3 cols)
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 22), facecolor=BG)
gs  = gridspec.GridSpec(4, 3, figure=fig,
        hspace=0.52, wspace=0.38,
        left=0.05, right=0.97, top=0.95, bottom=0.04)

# ──────────────────────────────────────────────────────────────────────────
# ROW 0 — Conceptual explainer: "what are the two groups?"
# ──────────────────────────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.set_facecolor(PANEL)
for sp in ax0.spines.values(): sp.set_edgecolor(GRID)
ax0.set_xlim(0, 20); ax0.set_ylim(0, 4.2)
ax0.axis("off")
ax0.set_title("Tại sao cần 2 param group?  —  Vấn đề về noise trên quantum loss surface",
              color=WHITE, fontsize=12, fontweight="bold", pad=8)

# QGCN model box
model_rect = FancyBboxPatch((0.3, 0.5), 3.5, 3.1,
    boxstyle="round,pad=0.1", lw=1.5, edgecolor=DIM, facecolor=CARD)
ax0.add_patch(model_rect)
ax0.text(2.05, 3.45, "QGCN Model", color=WHITE, fontsize=9.5,
         fontweight="bold", ha="center")

# layers inside model
for i, (name, color, y_) in enumerate([
    ("Linear encoder", BLUE, 2.8),
    ("QGCNConv layer 1", PURP, 2.1),
    ("  quantum_layer.weights ← 16 scalars", RED, 1.6),
    ("QGCNConv layer 2", PURP, 1.0),
    ("  quantum_layer.weights ← 16 scalars", RED, 0.55),
]):
    ax0.text(0.55, y_, name, color=color, fontsize=8,
             va="center", fontfamily="monospace")

ax0.text(2.05, 0.18, "Linear classifier", color=BLUE, fontsize=8, ha="center")

# arrow → split
hline(ax0, 3.9, 5.1, 2.0, YELL, lw=1.8)
ax0.text(4.5, 2.2, "split_quantum_\nclassical_params()", color=YELL,
         fontsize=7.5, ha="center", fontweight="bold")

# Classical group card
card(ax0, 5.3, 0.5, 4.5, 3.1, BLUE, "Classical Params",
     lines=[
         ("• Linear encoder weights",    TEXT),
         ("• Linear classifier weights", TEXT),
         ("• All non-quantum params",    TEXT),
         ("",                            TEXT),
         (f"base_lr  = {BASE_C:.0e}",    BLUE),
         (f"max_lr   = {PEAK_C:.4f}",    BLUE),
         (f"init_lr  = {INIT_C:.4f}",    BLUE),
         (f"min_lr   = {MIN_C:.0e}",     BLUE),
     ], fontsize=8)
ax0.text(7.55, 3.75, "Gradient: stable & dense", color=GREEN,
         fontsize=7.5, ha="center", style="italic")

# Quantum group card
card(ax0, 10.5, 0.5, 4.5, 3.1, RED, "Quantum Params  (Q-Drop noisy!)",
     lines=[
         ("• quantum_layer.weights", TEXT),
         ("  (PennyLane rotation angles)", DIM),
         ("• Only 32 scalars total", TEXT),
         ("",  TEXT),
         (f"base_lr  = {BASE_Q:.0e}  (×0.1)",   RED),
         (f"max_lr   = {PEAK_Q:.4f}  (×0.1)",   RED),
         (f"init_lr  = {INIT_Q:.4f}  (×0.1)",   RED),
         (f"min_lr   = {MIN_Q:.0e}   (×0.1)",   RED),
     ], fontsize=8)
ax0.text(12.75, 3.75, "Gradient: NOISY  (Q-Drop masks wires)", color=ORNG,
         fontsize=7.5, ha="center", style="italic")

hline(ax0, 3.9, 5.1, 2.0, YELL, lw=1.8)
hline(ax0, 5.15, 5.25, 2.0, BLUE, lw=0)  # dummy

# arrows to both groups
ax0.annotate("", xy=(5.3, 2.1), xytext=(5.15, 2.1),
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
ax0.annotate("", xy=(10.5, 2.1), xytext=(9.75, 2.1),
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
ax0.plot([5.15, 5.15], [2.1, 2.1], color=YELL)  # branch point
ax0.annotate("", xy=(10.5, 2.0), xytext=(5.15, 2.0),
    arrowprops=dict(arrowstyle="-", color=YELL, lw=1.5))
ax0.annotate("", xy=(10.5, 2.1), xytext=(9.75, 2.1),
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

# Both → AdamW
ax0.annotate("", xy=(15.4, 2.1), xytext=(15.0, 2.1),
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
ax0.plot([15.0, 15.0], [1.0, 3.15], color=YELL, lw=1.5)
ax0.plot([9.0, 15.0], [3.15, 3.15], color=BLUE, lw=1.5)
ax0.plot([9.0, 15.0], [1.0, 1.0],   color=RED,  lw=1.5)

card(ax0, 15.4, 0.5, 4.1, 3.1, GREEN, "AdamW Optimizer",
     lines=[
         ("param_groups = [",      TEXT),
         ("  {classical, lr=5e-3},", BLUE),
         ("  {quantum,   lr=5e-4},", RED),
         ("]",                      TEXT),
         ("weight_decay = 1e-3",    YELL),
         ("",                       TEXT),
         ("→ OneCycleLR schedules", GREEN),
         ("  each group separately", GREEN),
     ], fontsize=8)

# "is this new?" annotation
note_rect = FancyBboxPatch((15.4, 3.85), 4.1, 0.3,
    boxstyle="round,pad=0.05", lw=1, edgecolor=YELL, facecolor=YELL+"20")
ax0.add_patch(note_rect)
ax0.text(17.45, 4.0,
    "Kỹ thuật: Layer-wise LR  (không mới, nhưng hợp lý cho quantum noise)",
    color=YELL, fontsize=7, ha="center", va="center")

# ──────────────────────────────────────────────────────────────────────────
# ROW 1 — Full LR curves side-by-side, same y-axis scale (absolute)
# ──────────────────────────────────────────────────────────────────────────
ax1a = fig.add_subplot(gs[1, :2])
ax_base(ax1a,
    title="OneCycleLR — Classical vs Quantum  (absolute LR, same y-axis)",
    xlabel="Epoch", ylabel="Learning Rate")

ax1a.plot(epochs_arr, lrs_c, color=BLUE, lw=2.0, label="Classical  (max=0.025)")
ax1a.plot(epochs_arr, lrs_q, color=RED,  lw=2.0, linestyle="--", label="Quantum  (max=0.0025)")

warmup_ep = WARM / SPE
ax1a.axvline(warmup_ep, color=YELL, lw=1.0, ls=":", alpha=0.8)

# annotate key values on the right
for val, lbl, col in [
    (PEAK_C, f"peak  {PEAK_C:.4f}", BLUE),
    (INIT_C, f"init  {INIT_C:.4f}", BLUE),
    (MIN_C,  f"min   {MIN_C:.0e}",  BLUE),
    (PEAK_Q, f"peak  {PEAK_Q:.4f}", RED),
    (INIT_Q, f"init  {INIT_Q:.4f}", RED),
    (MIN_Q,  f"min   {MIN_Q:.0e}",  RED),
]:
    ax1a.axhline(val, color=col, lw=0.5, ls="--", alpha=0.4)
    ax1a.text(101.5, val, lbl, color=col, fontsize=6.5, va="center")

ax1a.text(warmup_ep + 0.5, PEAK_C * 1.02, "warmup ends\n(step 1 130)", color=YELL, fontsize=6.5)
ax1a.set_xlim(-1, 111)
ax1a.legend(fontsize=8.5, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID)

# ── ratio panel
ax1b = fig.add_subplot(gs[1, 2])
ax_base(ax1b, title="Ratio  quantum / classical  (always = 0.1)",
        xlabel="Epoch", ylabel="LR ratio")
ratio = lrs_q / np.maximum(lrs_c, 1e-15)
ax1b.plot(epochs_arr, ratio, color=ORNG, lw=1.8)
ax1b.axhline(0.1, color=YELL, lw=1.0, ls="--")
ax1b.text(2, 0.101, "quantum_lr_scale = 0.1  (constant)", color=YELL, fontsize=8)
ax1b.set_ylim(0, 0.15)
ax1b.set_xlim(-1, 101)
ax1b.text(50, 0.05,
    "Ratio hằng số = 0.1\n→ KHÔNG phải adaptive\n→ Đây chỉ là scale cố định",
    color=TEXT, fontsize=8, ha="center",
    bbox=dict(boxstyle="round,pad=0.3", fc=CARD, ec=GRID))

# ──────────────────────────────────────────────────────────────────────────
# ROW 2 — Warmup zoom  |  Cosine formula  |  Noise motivation
# ──────────────────────────────────────────────────────────────────────────
ax2a = fig.add_subplot(gs[2, 0])
ax_base(ax2a, title="Zoom: Warmup Phase (epoch 0 → 11)",
        xlabel="Step", ylabel="Learning Rate")
z = int(WARM * 1.6)
ax2a.plot(steps_arr[:z], lrs_c[:z], color=BLUE, lw=2.0, label="Classical")
ax2a.plot(steps_arr[:z], lrs_q[:z], color=RED,  lw=2.0, ls="--", label="Quantum")
ax2a.axvline(WARM, color=YELL, lw=1.0, ls=":")
ax2a.fill_betweenx([0, PEAK_C*1.05], 0, WARM, color=YELL, alpha=0.06)
ax2a.text(WARM*0.45, PEAK_C*0.55, "Linear ramp\ninit → peak", color=YELL, fontsize=7.5, ha="center")
ax2a.text(WARM*1.15, PEAK_C*0.85, "Cosine\ndecay starts", color=GREEN, fontsize=7.5)
ax2a.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID)

ax2b = fig.add_subplot(gs[2, 1])
ax_base(ax2b, title="Cosine Annealing  (90% of training)",
        xlabel="t / T_anneal  (0→1)", ylabel="Learning Rate")
t = np.linspace(0, 1, 500)
c_cos = MIN_C + 0.5*(PEAK_C - MIN_C)*(1 + np.cos(np.pi*t))
q_cos = MIN_Q + 0.5*(PEAK_Q - MIN_Q)*(1 + np.cos(np.pi*t))
ax2b.plot(t, c_cos, color=BLUE, lw=2.2, label="Classical")
ax2b.plot(t, q_cos, color=RED,  lw=2.2, ls="--", label="Quantum")
ax2b.fill_between(t, c_cos, MIN_C, color=BLUE, alpha=0.1)
ax2b.fill_between(t, q_cos, MIN_Q, color=RED,  alpha=0.1)
ax2b.text(0.5, PEAK_C*0.55,
    r"$lr(t)=lr_{min}+\frac{1}{2}(lr_{max}-lr_{min})(1+\cos\pi t)$",
    color=WHITE, fontsize=8, ha="center",
    bbox=dict(boxstyle="round,pad=0.3", fc=CARD, ec=GRID))
ax2b.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID)

# noise motivation panel
ax2c = fig.add_subplot(gs[2, 2])
ax2c.set_facecolor(PANEL)
for sp in ax2c.spines.values(): sp.set_edgecolor(GRID)
ax2c.axis("off")
ax2c.set_title("Tại sao LR quantum thấp hơn?", color=WHITE, fontsize=10, fontweight="bold", pad=6)

sections = [
    ("Vấn đề:", YELL, 9, True),
    ("Q-Drop mask một số quantum", TEXT, 8.2, False),
    ("wires mỗi step → gradient", TEXT, 8.2, False),
    ("của rotation angles bị nhiễu.", TEXT, 8.2, False),
    ("", TEXT, 8, False),
    ("Hậu quả nếu LR lớn:", RED, 9, True),
    ("• Overshooting trên loss", TEXT, 8.2, False),
    ("  surface nhiễu", TEXT, 8.2, False),
    ("• Quantum circuit weights", TEXT, 8.2, False),
    ("  không hội tụ", TEXT, 8.2, False),
    ("", TEXT, 8, False),
    ("Giải pháp:", GREEN, 9, True),
    ("LR_quantum = LR_classical × 0.1", GREEN, 8.2, False),
    ("(quantum_lr_scale = 0.1)", DIM, 7.5, False),
    ("", TEXT, 8, False),
    ("Tên kỹ thuật:", PURP, 9, True),
    ("Layer-wise Learning Rate", PURP, 8.5, False),
    ("(LLRD — không phải mới,", DIM, 7.5, False),
    (" nhưng áp dụng đúng chỗ)", DIM, 7.5, False),
]
y = 0.96
for txt, col, sz, bold in sections:
    ax2c.text(0.05, y, txt, color=col, fontsize=sz,
              fontweight="bold" if bold else "normal",
              transform=ax2c.transAxes, va="top")
    y -= 0.052 if txt else 0.022

# ──────────────────────────────────────────────────────────────────────────
# ROW 3 — Novelty verdict + AdamW update breakdown
# ──────────────────────────────────────────────────────────────────────────
ax3a = fig.add_subplot(gs[3, 0])
ax3a.set_facecolor(PANEL)
for sp in ax3a.spines.values(): sp.set_edgecolor(GRID)
ax3a.axis("off")
ax3a.set_title("Kết luận: Có mới không?", color=WHITE, fontsize=10, fontweight="bold", pad=6)

verdict = [
    ("AdamW",          "Standard ✓",  DIM,   BLUE),
    ("OneCycleLR",     "Standard ✓",  DIM,   BLUE),
    ("2 param groups", "Standard ✓",  DIM,   BLUE),
    ("LLRD scale×0.1", "Standard ✓",  DIM,   BLUE),
    ("Q-Drop masking", "NOVEL ★",     YELL,  GREEN),
    ("Quantum GCN",    "NOVEL ★",     YELL,  GREEN),
]

for i, (comp, status, sc, vc) in enumerate(verdict):
    y_  = 0.88 - i * 0.135
    box = FancyBboxPatch((0.03, y_ - 0.055), 0.94, 0.1,
        boxstyle="round,pad=0.02", lw=0.8,
        edgecolor=vc+"60", facecolor=vc+"15",
        transform=ax3a.transAxes)
    ax3a.add_patch(box)
    ax3a.text(0.08,  y_, comp,   color=sc, fontsize=8.5,
              transform=ax3a.transAxes, va="center")
    ax3a.text(0.92,  y_, status, color=vc, fontsize=8.5,
              transform=ax3a.transAxes, va="center", ha="right", fontweight="bold")

ax3a.text(0.5, 0.04,
    "LR strategy: hợp lý & đúng, nhưng không phải đóng góp novel.",
    color=DIM, fontsize=7, ha="center", style="italic",
    transform=ax3a.transAxes)

# AdamW update breakdown
ax3b = fig.add_subplot(gs[3, 1:])
ax_base(ax3b, title="AdamW Step — hai group nhận lr(t) khác nhau từ OneCycleLR",
        xlabel="Epoch", ylabel="Learning Rate")

ax3b.plot(epochs_arr, lrs_c, color=BLUE, lw=2.0, alpha=0.9)
ax3b.plot(epochs_arr, lrs_q, color=RED,  lw=2.0, alpha=0.9, ls="--")

# shade the 3 zones
wep  = WARM / SPE
ax3b.axvspan(0,   wep, alpha=0.10, color=YELL,  label="Warmup: linear ↑")
ax3b.axvspan(wep, 100, alpha=0.06, color=GREEN, label="Anneal: cosine ↓")

# annotate AdamW formula inline
mid = 55
lr_mid_c = lrs_c[int(mid * SPE)]
lr_mid_q = lrs_q[int(mid * SPE)]
ax3b.annotate(
    f"θ ← θ  −  lr_c(t)·AdamGrad\n   lr_c({mid})={lr_mid_c:.5f}",
    xy=(mid, lr_mid_c), xytext=(mid - 22, lr_mid_c + 0.005),
    color=BLUE, fontsize=7.5,
    arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9),
    bbox=dict(boxstyle="round,pad=0.25", fc=CARD, ec=BLUE+"60"),
)
ax3b.annotate(
    f"θ_q ← θ_q  −  lr_q(t)·AdamGrad\n   lr_q({mid})={lr_mid_q:.6f}",
    xy=(mid, lr_mid_q), xytext=(mid + 8, lr_mid_q + 0.004),
    color=RED, fontsize=7.5,
    arrowprops=dict(arrowstyle="->", color=RED, lw=0.9),
    bbox=dict(boxstyle="round,pad=0.25", fc=CARD, ec=RED+"60"),
)

ax3b.legend(
    handles=[
        plt.Line2D([0],[0], color=BLUE, lw=2, label="Classical LR  (32K+ params)"),
        plt.Line2D([0],[0], color=RED,  lw=2, ls="--", label="Quantum LR  (32 rotation angles)"),
        mpatches.Patch(fc=YELL+"30", ec="none", label="Warmup zone"),
        mpatches.Patch(fc=GREEN+"20", ec="none", label="Cosine anneal zone"),
    ],
    fontsize=8, labelcolor=TEXT, facecolor=PANEL, edgecolor=GRID, loc="upper right",
)
ax3b.set_xlim(-1, 104)

# ── suptitle ───────────────────────────────────────────────────────────────
fig.suptitle(
    "Giải thích: AdamW + OneCycleLR với 2 Param Group (Classical vs Quantum)",
    color=WHITE, fontsize=14, fontweight="bold", y=0.975,
)

OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT}")
plt.close(fig)
