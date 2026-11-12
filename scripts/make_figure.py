"""
COMPASS / multi_objective_framework architecture diagram.

Regenerates the three-layer (dataset / modality analysis / COMPASS core)
architecture figure from the README, fixing the layout bugs in the
original export:
  - "Level 2" extension tag no longer overlaps the "Objectives" box
  - "Level 3: BYO CSVs" label no longer collides with the
    "Artefact contract" title
  - The Layer 3 subtitle no longer touches the row of boxes below it

Run:
    python scripts/make_figure.py
Produces:
    mof_architecture.png (and .pdf) in the repository root
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------
GREEN = "#55C596"        # fixed core
GREEN_EDGE = "#2E9A72"
ORANGE = "#F5B942"       # user-configurable
ORANGE_EDGE = "#D99A1F"
GRAY = "#C7C2B5"         # swappable layer
GRAY_EDGE = "#9C9788"
WHITE = "#FFFFFF"
WHITE_EDGE = "#8A8579"

LAYER3_BG = "#DEF5EC"
LAYER3_EDGE = "#3FAE84"
LAYER2_BG = "#EDE8DC"
LAYER2_EDGE = "#A39C89"
LAYER1_BG = "#F6F4EC"
LAYER1_EDGE = "#B7B19E"

TEXT_DARK = "#2B2B26"
TEXT_MUTED = "#6B6658"
ARROW_COLOR = "#6B6658"


def rbox(ax, x, y, w, h, facecolor, edgecolor, lw=1.4, radius=0.35, z=2):
    """Rounded rectangle in data coordinates."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw,
        facecolor=facecolor,
        edgecolor=edgecolor,
        zorder=z,
    )
    ax.add_patch(box)
    return box


def label(ax, x, y, text, *, size=11, weight="normal", color=TEXT_DARK,
          ha="left", va="center", style="normal", z=3):
    ax.text(x, y, text, fontsize=size, fontweight=weight, color=color,
             ha=ha, va=va, fontstyle=style, zorder=z)


def h_arrow(ax, x1, x2, y, color=ARROW_COLOR, lw=1.6):
    arr = FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>",
                           mutation_scale=14, linewidth=lw,
                           color=color, zorder=4)
    ax.add_patch(arr)


def v_arrow(ax, x, y1, y2, color=ARROW_COLOR, lw=1.6, style="-|>"):
    arr = FancyArrowPatch((x, y1), (x, y2), arrowstyle=style,
                           mutation_scale=14, linewidth=lw,
                           color=color, zorder=4)
    ax.add_patch(arr)


# ----------------------------------------------------------------------
# Figure / axis setup
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 14.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 139)
ax.axis("off")

# ----------------------------------------------------------------------
# Legend
# ----------------------------------------------------------------------
legend_y = 135.5
swatch_w, swatch_h = 2.6, 2.6
items = [
    ("Fixed core", GREEN, GREEN_EDGE),
    ("User-configurable", ORANGE, ORANGE_EDGE),
    ("Swappable layer", GRAY, GRAY_EDGE),
]
lx = 4
for text, fc, ec in items:
    rbox(ax, lx, legend_y - swatch_h / 2, swatch_w, swatch_h, fc, ec,
         lw=1.2, radius=0.15)
    label(ax, lx + swatch_w + 1.2, legend_y, text, size=17, color=TEXT_DARK)
    lx += swatch_w + 1.2 + len(text) * 0.95 + 5

# ========================================================================
# LAYER 3 -- COMPASS core
# ========================================================================
L3_Y0, L3_Y1 = 79, 128
rbox(ax, 3, L3_Y0, 94, L3_Y1 - L3_Y0, LAYER3_BG, LAYER3_EDGE, lw=1.8,
     radius=1.0, z=1)

label(ax, 6.5, 124.3, "Layer 3 — COMPASS core", size=22, weight="bold",
      color="#1F7A57")
label(ax, 6.5, 120.4,
      "modality-agnostic \u00b7 Lagrangian shell is fixed",
      size=16.5, style="italic", color="#3FAE84")

# --- Row 1: Bootstrap / Objectives / Lagrangian core -------------------
row1_y0, row1_h = 106.5, 10.5
box_w = 26.5
gap = 3.7
x_boot = 6.5
x_obj = x_boot + box_w + gap
x_lag = x_obj + box_w + gap

rbox(ax, x_boot, row1_y0, box_w, row1_h, GREEN, GREEN_EDGE)
label(ax, x_boot + box_w / 2, row1_y0 + row1_h - 3.4, "Bootstrap",
      size=17, weight="bold", ha="center", color=TEXT_DARK)
label(ax, x_boot + box_w / 2, row1_y0 + 3.4, "stability metrics \u2192 f_P, f_S",
      size=12.5, ha="center", color=TEXT_DARK)

rbox(ax, x_obj, row1_y0, box_w, row1_h, ORANGE, ORANGE_EDGE)
label(ax, x_obj + box_w / 2, row1_y0 + row1_h - 3.4, "Objectives",
      size=17, weight="bold", ha="center", color=TEXT_DARK)
label(ax, x_obj + box_w / 2, row1_y0 + 3.4, "f_P, f_S, f_A  (YAML \u00b7 L1)",
      size=12.5, ha="center", color=TEXT_DARK)

rbox(ax, x_lag, row1_y0, box_w, row1_h, GREEN, GREEN_EDGE)
label(ax, x_lag + box_w / 2, row1_y0 + row1_h - 3.4, "Lagrangian core",
      size=17, weight="bold", ha="center", color=TEXT_DARK)
label(ax, x_lag + box_w / 2, row1_y0 + 3.4, "frontier \u00b7 shadow price",
      size=12.5, ha="center", color=TEXT_DARK)

h_arrow(ax, x_boot + box_w + 0.3, x_obj - 0.3, row1_y0 + row1_h / 2)
h_arrow(ax, x_obj + box_w + 0.3, x_lag - 0.3, row1_y0 + row1_h / 2)

# --- Level 2 extension tag (own row, clear of everything else) --------
lvl2_y0, lvl2_h = 98.3, 5.6
lvl2_w = 40
lvl2_x = x_obj + box_w / 2 - lvl2_w / 2
v_arrow(ax, x_obj + box_w / 2, row1_y0 - 0.15, lvl2_y0 + lvl2_h + 0.15)

rbox(ax, lvl2_x, lvl2_y0, lvl2_w, lvl2_h, ORANGE, ORANGE_EDGE, lw=1.2,
     radius=0.3)
label(ax, lvl2_x + lvl2_w / 2, lvl2_y0 + lvl2_h - 1.9,
      "Level 2", size=17, weight="bold", ha="center",
      color=TEXT_DARK)
label(ax, lvl2_x + lvl2_w / 2, lvl2_y0 + 1.7,
      "pandas on objective_inputs.csv", size=13,
      ha="center", color=TEXT_DARK)

v_arrow(ax, x_obj + box_w / 2, lvl2_y0 - 0.15, lvl2_y0 - 1.9)

# --- Artefact contract row ---------------------------------------------
art_y0, art_h = 88.7, 8.4
rbox(ax, 6.5, art_y0, 87, art_h, WHITE, WHITE_EDGE, lw=1.4, radius=0.4)
label(ax, 50, art_y0 + art_h - 2.9, "Artefact contract (CSV / NPZ)",
      size=16.5, weight="bold", ha="center", color=TEXT_DARK)
label(ax, 50, art_y0 + 2.3,
      "join: task_id \u00b7 architecture \u00b7 lr \u00b7 dr   |   "
      "bootstrap_results \u00b7 sweep_fa_*",
      size=12, ha="center", color=TEXT_MUTED)

# "Level 3: BYO CSVs" -- placed below the artefact-contract box, fully
# clear of its border, with a short arrow pointing up into the corner.
lvl3_label_y = art_y0 - 3.6
v_arrow(ax, 10.5, lvl3_label_y + 1.0, art_y0 - 0.15, lw=1.3,
        style="-|>")
label(ax, 12.5, lvl3_label_y, "Level 3: BYO CSVs",
      size=12, style="italic", color=TEXT_MUTED, ha="left")

# ========================================================================
# Connector: Layer 3 -> Layer 2
# ========================================================================
v_arrow(ax, 50, L3_Y0 - 0.2, 76.5)

# ========================================================================
# LAYER 2 -- modality analysis
# ========================================================================
L2_Y0, L2_Y1 = 43, 74
rbox(ax, 3, L2_Y0, 94, L2_Y1 - L2_Y0, LAYER2_BG, LAYER2_EDGE, lw=1.8,
     radius=1.0, z=1)

label(ax, 6.5, L2_Y1 - 4.2, "Layer 2 — modality analysis", size=22,
      weight="bold", color="#736C58")
label(ax, 6.5, L2_Y1 - 8.0, "ECG reference \u00b7 swappable or skipped (Level 3)",
      size=16.5, style="italic", color="#8B8570")

row2_y0, row2_h = L2_Y0 + 3.5, 10.5
box_w2 = 26.5
x_a = 6.5
x_b = x_a + box_w2 + gap
x_c = x_b + box_w2 + gap

rbox(ax, x_a, row2_y0, box_w2, row2_h, GRAY, GRAY_EDGE)
label(ax, x_a + box_w2 / 2, row2_y0 + row2_h - 3.4, "Train + sweep",
      size=16.5, weight="bold", ha="center", color=TEXT_DARK)
label(ax, x_a + box_w2 / 2, row2_y0 + 3.4, "train_multi_f_array.py",
      size=12, ha="center", color=TEXT_DARK)

rbox(ax, x_b, row2_y0, box_w2, row2_h, GRAY, GRAY_EDGE)
label(ax, x_b + box_w2 / 2, row2_y0 + row2_h - 3.4, "Saliency scoring",
      size=16.5, weight="bold", ha="center", color=TEXT_DARK)
label(ax, x_b + box_w2 / 2, row2_y0 + 3.4, "component columns \u2192 f_A",
      size=12, ha="center", color=TEXT_DARK)

rbox(ax, x_c, row2_y0, box_w2, row2_h, GRAY, GRAY_EDGE)
label(ax, x_c + box_w2 / 2, row2_y0 + row2_h - 3.4, "Artefacts",
      size=16.5, weight="bold", ha="center", color=TEXT_DARK)
label(ax, x_c + box_w2 / 2, row2_y0 + 3.4, "npz + per-run CSVs",
      size=12, ha="center", color=TEXT_DARK)

h_arrow(ax, x_a + box_w2 + 0.3, x_b - 0.3, row2_y0 + row2_h / 2)
h_arrow(ax, x_b + box_w2 + 0.3, x_c - 0.3, row2_y0 + row2_h / 2)

# ========================================================================
# Connector: Layer 1 -> Layer 2
# ========================================================================
v_arrow(ax, 50, 27.5, L2_Y0 - 0.2)

# ========================================================================
# LAYER 1 -- dataset
# ========================================================================
L1_Y0, L1_Y1 = 4, 27
rbox(ax, 3, L1_Y0, 94, L1_Y1 - L1_Y0, LAYER1_BG, LAYER1_EDGE, lw=1.8,
     radius=1.0, z=1)

label(ax, 6.5, L1_Y1 - 4.2, "Layer 1 — dataset", size=22, weight="bold",
      color="#847E68")
label(ax, 6.5, L1_Y1 - 8.0, "user-supplied or built cohort pickles",
      size=16.5, style="italic", color="#9A937B")

row3_y0, row3_h = L1_Y0 + 4.5, 10.5
box_w3 = 40
gap3 = 6.0
x_d = 10
x_e = x_d + box_w3 + gap3

rbox(ax, x_d, row3_y0, box_w3, row3_h, GRAY, GRAY_EDGE)
label(ax, x_d + box_w3 / 2, row3_y0 + row3_h - 3.4, "Dataset builder",
      size=16.5, weight="bold", ha="center", color=TEXT_DARK)
label(ax, x_d + box_w3 / 2, row3_y0 + 3.4,
      "external \u00b7 build_datasets.py \u00b7 examples/", size=12,
      ha="center", color=TEXT_DARK)

rbox(ax, x_e, row3_y0, box_w3, row3_h, GRAY, GRAY_EDGE)
label(ax, x_e + box_w3 / 2, row3_y0 + row3_h - 3.4, "Pickle contract",
      size=16.5, weight="bold", ha="center", color=TEXT_DARK)
label(ax, x_e + box_w3 / 2, row3_y0 + 3.4, "signal \u00b7 label \u00b7 demographics",
      size=12, ha="center", color=TEXT_DARK)

h_arrow(ax, x_d + box_w3 + 0.3, x_e - 0.3, row3_y0 + row3_h / 2)

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
label(ax, 4, 1.6,
      "Objective extension:   L1 = edit YAML   \u00b7   "
      "L2 = post-process CSV   \u00b7   L3 = supply artefact tables",
      size=17, color=TEXT_MUTED)

plt.tight_layout()
repo_root = Path(__file__).resolve().parents[1]
out_png = repo_root / "mof_architecture.png"
plt.savefig(out_png, dpi=220, bbox_inches="tight")
plt.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
manuscript_png = repo_root / "mof_architecture_fixed_v2.png"
manuscript_png.write_bytes(out_png.read_bytes())
print(f"wrote {out_png}")
print(f"wrote {manuscript_png}")