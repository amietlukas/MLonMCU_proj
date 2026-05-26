"""
Generate VGG-style architecture block diagrams for the three BASELINE models.

Each diagram visualises the BaselineCNN topology defined in
software/classifier/src/model.py: a sequence of
[Conv3x3 -> BN -> ReLU -> Dropout2d -> MaxPool2x2] stages,
followed by Global Average Pooling and a single Linear classifier.

Outputs (PNG + PDF + SVG) are written next to this script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D


# --------------------------------------------------------------------------- #
# Configuration: one entry per BASELINE model.
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    name: str
    pretty: str
    in_channels: int
    input_hw: Tuple[int, int]      # (H, W)
    channels: List[int]            # per-stage out channels
    kernel_size: int
    dropout_rate: float
    num_classes: int
    total_params: int
    mults_megs: float


MODELS: List[ModelSpec] = [
    ModelSpec(
        name="bignet_baseline",
        pretty="bignet  -  BASELINE",
        in_channels=1,
        input_hw=(120, 160),
        channels=[16, 32, 64, 96, 128],
        kernel_size=3,
        dropout_rate=0.05,
        num_classes=6,
        total_params=190_518,
        mults_megs=71.33,
    ),
    ModelSpec(
        name="smallnet_greyscale_baseline_small",
        pretty="smallnet greyscale  -  BASELINE_SMALL",
        in_channels=1,
        input_hw=(120, 160),
        channels=[32, 64, 128],
        kernel_size=3,
        dropout_rate=0.05,
        num_classes=6,
        total_params=93_670,
        mults_megs=182.48,
    ),
    ModelSpec(
        name="bignet_pruned_baseline_pruned",
        pretty="bignet_pruned  -  BASELINE_PRUNED  (~70% pruned)",
        in_channels=1,
        input_hw=(120, 160),
        channels=[5, 10, 19, 29, 38],
        kernel_size=3,
        dropout_rate=0.05,
        num_classes=6,
        total_params=17_518,
        mults_megs=7.26,
    ),
]


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
# Colour palette (colour-blind friendly).
C_INPUT = "#d9d9d9"
C_CONV = "#4c78a8"
C_BN = "#9ecae9"
C_RELU = "#f58518"
C_DROP = "#e45756"
C_POOL = "#54a24b"
C_GAP = "#b279a2"
C_FC = "#eeca3b"
C_OUTPUT = "#bab0ac"


def _conv_params(in_ch: int, out_ch: int, k: int) -> int:
    # bias=False in BaselineCNN; BN has its own params.
    return in_ch * out_ch * k * k


def _bn_params(out_ch: int) -> int:
    return 2 * out_ch  # gamma + beta


def _add_box(ax, x, y, w, h, text, face, edge="#222222", fontsize=8, weight="normal"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=0.8,
        facecolor=face,
        edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(x + w / 2.0, y + h / 2.0, text,
            ha="center", va="center", fontsize=fontsize, weight=weight, color="#111111")


def _add_arrow(ax, x0, y0, x1, y1):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=0.9,
        color="#444444",
    )
    ax.add_patch(arr)


def _shape_str(c: int, h: int, w: int) -> str:
    return f"[{c} x {h} x {w}]"


def render(spec: ModelSpec, out_dir: Path) -> None:
    """Draw a top-to-bottom architecture diagram for one model."""
    # ----- pre-compute per-stage shapes & param counts -----
    H, W = spec.input_hw
    in_ch = spec.in_channels
    rows = []  # one per drawn row

    # Header row 0: Input tensor
    rows.append({
        "kind": "input",
        "shape": _shape_str(in_ch, H, W),
        "text": f"Input  ( grayscale )\n{_shape_str(in_ch, H, W)}",
    })

    # Stages
    for stage_idx, out_ch in enumerate(spec.channels, start=1):
        # conv preserves spatial size, then pool halves it
        conv_p = _conv_params(in_ch, out_ch, spec.kernel_size)
        bn_p = _bn_params(out_ch)
        rows.append({
            "kind": "stage",
            "stage_idx": stage_idx,
            "in_ch": in_ch,
            "out_ch": out_ch,
            "H_in": H, "W_in": W,
            "conv_params": conv_p,
            "bn_params": bn_p,
        })
        # update spatial dims after pooling
        H //= 2
        W //= 2
        in_ch = out_ch
        rows.append({
            "kind": "shape_after_stage",
            "shape": _shape_str(in_ch, H, W),
        })

    # GAP
    rows.append({"kind": "gap", "channels": in_ch, "H": H, "W": W})
    rows.append({"kind": "shape_after_gap", "shape": f"[{in_ch}]"})

    # Dropout (FC dropout)
    if spec.dropout_rate > 0:
        rows.append({"kind": "drop_fc", "rate": spec.dropout_rate})

    # Linear
    fc_p = in_ch * spec.num_classes + spec.num_classes
    rows.append({
        "kind": "fc",
        "in_dim": in_ch,
        "out_dim": spec.num_classes,
        "params": fc_p,
    })
    rows.append({"kind": "output", "shape": f"[{spec.num_classes}]  (logits)"})

    # ----- figure layout -----
    # Sub-blocks per stage: Conv | BN | ReLU | (Dropout2d) | MaxPool
    sub_w = 2.4
    sub_h = 0.55
    gap_x = 0.2
    stage_total_w = 5 * sub_w + 4 * gap_x  # always reserve room for 5 sub-blocks

    n_stages = len(spec.channels)
    # vertical sizing: header + 2 lines per stage (stage row + shape arrow row) + GAP + shape + drop + FC + out
    # extra +1.2 at the bottom to make room for the legend below the Output box.
    fig_w = max(11.0, stage_total_w + 4.5)
    fig_h = 2.0 + n_stages * 1.4 + 3.5 + 1.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # title
    ax.text(fig_w / 2.0, fig_h - 0.35,
            spec.pretty,
            ha="center", va="center", fontsize=14, weight="bold")
    ax.text(fig_w / 2.0, fig_h - 0.85,
            f"VGG-style plain CNN  -  {spec.total_params:,} params  -  "
            f"{spec.mults_megs:.1f} MMACs  -  input {_shape_str(spec.in_channels, *spec.input_hw)}",
            ha="center", va="center", fontsize=10, color="#444444")

    # cursor y starts under the title
    y = fig_h - 1.4
    cx = fig_w / 2.0  # horizontal centre

    # ---- draw Input ----
    in_w = 4.5
    _add_box(ax, cx - in_w / 2.0, y - 0.6, in_w, 0.6,
             rows[0]["text"], C_INPUT, fontsize=9, weight="bold")
    prev_bottom = (cx, y - 0.6)
    y -= 0.85

    idx = 1
    while idx < len(rows):
        r = rows[idx]

        if r["kind"] == "stage":
            stage_idx = r["stage_idx"]
            in_ch = r["in_ch"]
            out_ch = r["out_ch"]
            conv_p = r["conv_params"]
            bn_p = r["bn_params"]

            # arrow into stage band
            _add_arrow(ax, prev_bottom[0], prev_bottom[1], cx, y + 0.05)

            # stage label on the left
            ax.text(
                cx - stage_total_w / 2.0 - 0.25, y - sub_h / 2.0,
                f"Stage {stage_idx}",
                ha="right", va="center", fontsize=10, weight="bold", color="#222222",
            )

            # 5 sub-blocks horizontally
            sub_blocks = [
                (f"Conv {spec.kernel_size}x{spec.kernel_size}\n{in_ch} -> {out_ch}\n{conv_p:,} p", C_CONV),
                (f"BatchNorm2d\n{bn_p} p", C_BN),
                ("ReLU", C_RELU),
            ]
            if spec.dropout_rate > 0:
                sub_blocks.append((f"Dropout2d\np={spec.dropout_rate}", C_DROP))
            else:
                sub_blocks.append(("", None))  # blank placeholder
            sub_blocks.append(("MaxPool 2x2\nstride 2", C_POOL))

            x_start = cx - stage_total_w / 2.0
            x_cursor = x_start
            sub_centers = []
            for text, face in sub_blocks:
                if face is None:
                    x_cursor += sub_w + gap_x
                    continue
                _add_box(ax, x_cursor, y - sub_h, sub_w, sub_h, text, face,
                         fontsize=7.5, weight="bold")
                sub_centers.append((x_cursor + sub_w / 2.0, y - sub_h, x_cursor + sub_w, y - sub_h / 2.0))
                # connect to previous sub-block (small arrow)
                if len(sub_centers) >= 2:
                    x_prev_right = sub_centers[-2][2]
                    y_mid = sub_centers[-2][3]
                    _add_arrow(ax, x_prev_right + 0.02, y_mid,
                               x_cursor - 0.02, y_mid)
                x_cursor += sub_w + gap_x

            prev_bottom = (cx, y - sub_h)
            y -= sub_h + 0.35

        elif r["kind"] == "shape_after_stage":
            ax.text(cx, y - 0.18, r["shape"],
                    ha="center", va="center",
                    fontsize=9, color="#222222", style="italic",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="#bbbbbb", linewidth=0.6))
            _add_arrow(ax, prev_bottom[0], prev_bottom[1], cx, y - 0.05)
            prev_bottom = (cx, y - 0.35)
            y -= 0.55

        elif r["kind"] == "gap":
            _add_arrow(ax, prev_bottom[0], prev_bottom[1], cx, y + 0.05)
            w_box = 5.0
            _add_box(ax, cx - w_box / 2.0, y - 0.55, w_box, 0.55,
                     f"Global Average Pool   ( mean over H,W )", C_GAP,
                     fontsize=9, weight="bold")
            prev_bottom = (cx, y - 0.55)
            y -= 0.8

        elif r["kind"] == "shape_after_gap":
            ax.text(cx, y - 0.18, r["shape"],
                    ha="center", va="center",
                    fontsize=9, color="#222222", style="italic",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="#bbbbbb", linewidth=0.6))
            _add_arrow(ax, prev_bottom[0], prev_bottom[1], cx, y - 0.05)
            prev_bottom = (cx, y - 0.35)
            y -= 0.55

        elif r["kind"] == "drop_fc":
            _add_arrow(ax, prev_bottom[0], prev_bottom[1], cx, y + 0.05)
            w_box = 3.4
            _add_box(ax, cx - w_box / 2.0, y - 0.5, w_box, 0.5,
                     f"Dropout  p={r['rate']}", C_DROP, fontsize=9, weight="bold")
            prev_bottom = (cx, y - 0.5)
            y -= 0.7

        elif r["kind"] == "fc":
            _add_arrow(ax, prev_bottom[0], prev_bottom[1], cx, y + 0.05)
            w_box = 5.0
            _add_box(ax, cx - w_box / 2.0, y - 0.55, w_box, 0.55,
                     f"Linear   {r['in_dim']} -> {r['out_dim']}   ({r['params']:,} p)",
                     C_FC, fontsize=9, weight="bold")
            prev_bottom = (cx, y - 0.55)
            y -= 0.8

        elif r["kind"] == "output":
            _add_arrow(ax, prev_bottom[0], prev_bottom[1], cx, y + 0.05)
            w_box = 4.5
            _add_box(ax, cx - w_box / 2.0, y - 0.55, w_box, 0.55,
                     f"Output   {r['shape']}", C_OUTPUT, fontsize=9, weight="bold")
            prev_bottom = (cx, y - 0.55)
            y -= 0.8

        idx += 1

    # legend (bottom-right)
    legend_handles = [
        Line2D([0], [0], marker="s", color="w", label="Conv 3x3",
               markerfacecolor=C_CONV, markersize=10),
        Line2D([0], [0], marker="s", color="w", label="BatchNorm2d",
               markerfacecolor=C_BN, markersize=10),
        Line2D([0], [0], marker="s", color="w", label="ReLU",
               markerfacecolor=C_RELU, markersize=10),
        Line2D([0], [0], marker="s", color="w", label="Dropout2d",
               markerfacecolor=C_DROP, markersize=10),
        Line2D([0], [0], marker="s", color="w", label="MaxPool 2x2",
               markerfacecolor=C_POOL, markersize=10),
        Line2D([0], [0], marker="s", color="w", label="Global Avg Pool",
               markerfacecolor=C_GAP, markersize=10),
        Line2D([0], [0], marker="s", color="w", label="Linear (FC)",
               markerfacecolor=C_FC, markersize=10),
    ]
    # Reserve a stripe at the bottom of the data-coordinate canvas and put the
    # legend there - this avoids any clash with the Output box.
    legend_y = 0.4   # data coords - well below the Output box (which sits near y=1.0)
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4,
        frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, 0.015),    # figure coords - just above the bottom edge
    )
    # leave some bottom padding in figure coords for the legend
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.07)
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"{spec.name}_architecture.{ext}"
        # do NOT use bbox_inches="tight" - it would crop away the reserved
        # bottom margin and put the legend back on top of the Output box.
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in MODELS:
        render(spec, out_dir)


if __name__ == "__main__":
    main()
