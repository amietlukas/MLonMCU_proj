"""
High-level (slide 1) pipeline flow chart for the hand-gesture classifier:

    grayscale image  ->  [ BLOCKS ]  ->  MLP head  ->  6 drive intents
                          Conv3x3 -> BN -> ReLU -> MaxPool2x2   (GAP -> Linear)

This is the conceptual overview that accompanies the detailed per-layer
diagrams produced by _make_arch_diagrams.py. Palette is shared with that
script so the deck stays visually consistent.

Outputs (PNG + PDF + SVG) are written next to this script as
"pipeline_overview.*".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- shared palette (matches _make_arch_diagrams.py) -----------------------
C_INPUT = "#d9d9d9"
C_CONV = "#4c78a8"
C_BN = "#9ecae9"
C_RELU = "#f58518"
C_POOL = "#54a24b"
C_GAP = "#b279a2"
C_FC = "#eeca3b"
C_OUTPUT = "#bab0ac"
C_BLOCK_BAND = "#eef3f8"
EDGE = "#222222"


def _box(ax, x, y, w, h, text, face, fontsize=12, weight="bold",
         edge=EDGE, lw=1.1, rounding=0.12, txtcolor="#111111"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=lw, facecolor=face, edgecolor=edge,
    ))
    ax.text(x + w / 2.0, y + h / 2.0, text,
            ha="center", va="center", fontsize=fontsize, weight=weight,
            color=txtcolor)


# one shared arrow style for the whole diagram
ARROW_LW = 1.5
ARROW_SCALE = 14
ARROW_COLOR = "#333333"


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>",
        mutation_scale=ARROW_SCALE, linewidth=ARROW_LW, color=ARROW_COLOR,
    ))


def render(out_dir: Path) -> None:
    fig_w, fig_h = 15.0, 3.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # --- main row geometry --------------------------------------------------
    row_y = 1.2           # bottom of the main boxes
    box_h = 1.2
    cy = row_y + box_h / 2.0

    # Input
    in_x, in_w = 0.4, 2.6
    _box(ax, in_x, row_y, in_w, box_h,
         "Grayscale image\n1 x 120 x 160\n(QQVGA)", C_INPUT, fontsize=12)

    # BLOCKS band (the big middle box)
    blk_x, blk_w = 3.8, 5.6
    _box(ax, blk_x, row_y - 0.35, blk_w, box_h + 0.7,
         "", C_BLOCK_BAND, lw=1.3, rounding=0.14, edge="#9bbcdc")
    ax.text(blk_x + blk_w / 2.0, row_y + box_h + 0.18,
            "BLOCKS   ( x N stages )", ha="center", va="center",
            fontsize=12.5, weight="bold", color="#234")

    # MLP head
    mlp_x, mlp_w = 10.1, 2.1
    _box(ax, mlp_x, row_y, mlp_w, box_h,
         "MLP head\nGlobal Avg Pool\n-> Linear", C_FC, fontsize=12)

    # Output
    out_x, out_w = 12.9, 1.7
    _box(ax, out_x, row_y, out_w, box_h,
         "6 logits", C_OUTPUT, fontsize=12)

    # main arrows (one shared style everywhere - see ARROW_* constants)
    _arrow(ax, in_x + in_w + 0.05, cy, blk_x - 0.4 + 0.05, cy)
    _arrow(ax, blk_x + blk_w + 0.4 - 0.05, cy, mlp_x - 0.05, cy)
    _arrow(ax, mlp_x + mlp_w + 0.05, cy, out_x - 0.05, cy)

    # --- inner sub-blocks inside BLOCKS band --------------------------------
    sub_specs = [
        ("Conv 3x3", C_CONV, "white"),
        ("BN", C_BN, "#111111"),
        ("ReLU", C_RELU, "#111111"),
        ("MaxPool 2x2", C_POOL, "white"),
    ]
    n = len(sub_specs)
    pad = 0.18
    inner_left = blk_x + 0.30
    inner_right = blk_x + blk_w - 0.30
    avail = inner_right - inner_left
    sub_w = (avail - (n - 1) * pad) / n
    sub_h = 0.92
    sub_y = row_y + (box_h - sub_h) / 2.0
    x = inner_left
    centers = []
    for text, face, tc in sub_specs:
        _box(ax, x, sub_y, sub_w, sub_h, text, face, fontsize=10.5,
             txtcolor=tc, rounding=0.10)
        centers.append((x, x + sub_w))
        x += sub_w + pad
    # small arrows between sub-blocks
    ymid = sub_y + sub_h / 2.0
    for i in range(n - 1):
        _arrow(ax, centers[i][1] + 0.01, ymid, centers[i + 1][0] - 0.01, ymid)

    # caption under the blocks band: what changes per stage
    ax.text(blk_x + blk_w / 2.0, row_y - 0.55,
            "each stage: doubles channels, halves H x W",
            ha="center", va="center", fontsize=12.5, style="italic",
            color="#444444")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"pipeline_overview.{ext}"
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    render(out_dir)


if __name__ == "__main__":
    main()
