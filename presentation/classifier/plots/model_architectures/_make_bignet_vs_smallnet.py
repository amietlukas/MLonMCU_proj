"""
Slide-2 comparison diagram: BigNet vs SmallNet (BEFORE pruning / quant).

Each network is drawn as a row of collapsed stage-"blocks" (one box per stage,
labelled with out-channels + feature-map size), matching the high-level style of
pipeline_overview.py. One shared block = Conv3x3 -> BN -> ReLU -> MaxPool2x2.

Numbers are taken from the two BASELINE runs:
  bignet-20260505-163101_BASELINE          channels [16,32,64,96,128]
  smallnet_greyscale-20260504-184641_...   channels [32,64,128]

Outputs (PNG + PDF + SVG) -> "bignet_vs_smallnet.*" next to this script.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- shared palette (matches the other diagrams) ---------------------------
C_INPUT = "#d9d9d9"
C_BLOCK = "#4c78a8"
C_HEAD = "#eeca3b"
C_OUTPUT = "#bab0ac"
EDGE = "#222222"

# one shared arrow style everywhere
ARROW_LW = 1.5
ARROW_SCALE = 14
ARROW_COLOR = "#333333"

INPUT_HW = (120, 160)   # H, W


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


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>",
        mutation_scale=ARROW_SCALE, linewidth=ARROW_LW, color=ARROW_COLOR,
    ))


def _draw_row(ax, y, label, channels: List[int], params: str, macc: str,
              dim_dy: float = -0.22):
    """Draw one network as: label | input | block.. | head | 6 logits | stats."""
    box_h = 0.95
    cy = y + box_h / 2.0
    gap = 0.28

    # row label on far left
    ax.text(0.15, cy, label, ha="left", va="center",
            fontsize=15, weight="bold", color="#222222")

    x = 2.0

    # input
    in_w = 1.5
    _box(ax, x, y, in_w, box_h, f"input\n1x{INPUT_HW[0]}x{INPUT_HW[1]}",
         C_INPUT, fontsize=9.5)
    prev_right = x + in_w
    x = prev_right + gap

    # stage blocks
    blk_w = 1.15
    H, W = INPUT_HW
    for i, ch in enumerate(channels, start=1):
        H //= 2
        W //= 2
        _arrow(ax, prev_right + 0.02, cy, x - 0.02, cy)
        _box(ax, x, y, blk_w, box_h, f"{ch}", C_BLOCK, fontsize=15,
             txtcolor="white")
        # feature-map size under the block
        ax.text(x + blk_w / 2.0, y + dim_dy, f"{H}x{W}",
                ha="center", va="center", fontsize=12, weight="bold",
                color="#111111")
        prev_right = x + blk_w
        x = prev_right + gap

    # head
    head_w = 1.7
    _arrow(ax, prev_right + 0.02, cy, x - 0.02, cy)
    _box(ax, x, y, head_w, box_h, "GAP\n-> Linear", C_HEAD, fontsize=10)
    prev_right = x + head_w
    x = prev_right + gap

    # output
    out_w = 1.25
    _arrow(ax, prev_right + 0.02, cy, x - 0.02, cy)
    _box(ax, x, y, out_w, box_h, "6\nlogits", C_OUTPUT, fontsize=10)
    prev_right = x + out_w

    # stats on the right
    ax.text(prev_right + 0.45, cy,
            f"{len(channels)} stages\n{params} params\n{macc} MACCs",
            ha="left", va="center", fontsize=14, weight="bold", color="#000000")


def render(out_dir: Path) -> None:
    fig_w, fig_h = 17.0, 3.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # BigNet on top, SmallNet below
    _draw_row(ax, y=2.05, label="BigNet",
              channels=[16, 32, 64, 96, 128],
              params="190 k", macc="71 M", dim_dy=-0.20)
    _draw_row(ax, y=0.55, label="SmallNet",
              channels=[32, 64, 128],
              params="94 k", macc="182 M", dim_dy=-0.20)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02)
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"bignet_vs_smallnet.{ext}"
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    render(Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
