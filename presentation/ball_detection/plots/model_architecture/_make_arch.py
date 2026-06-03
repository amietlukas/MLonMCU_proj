"""
Ball-STYOLO-Nano architecture diagrams.

Produces:
  ball_styolo_nano_architecture.{png,pdf,svg}   -- high-level main chart (slide 2)
  detail_backbone.{png,pdf,svg}                 -- appendix: one stage + bottleneck
  detail_neck.{png,pdf,svg}                     -- appendix: FPN+PAN lanes
  detail_head.{png,pdf,svg}                     -- appendix: detection head

Channels are the FULL (un-pruned) design: stem 32 -> 48 -> 64 -> 128 -> 192,
neck 96. Input 3 x 288 x 384.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent

# palette
C_INPUT = "#d9d9d9"
C_BACK_BG = "#dce9f5"; C_BACK = "#4c78a8"
C_NECK_BG = "#e3f1de"; C_NECK = "#54a24b"
C_HEAD_BG = "#fde7d0"; C_HEAD = "#f58518"
C_OUT = "#eeca3b"
C_POST_BG = "#efdcec"; C_POST = "#b279a2"
C_BNECK = "#9ecae9"
EDGE = "#333333"; ARC = "#555555"


def rbox(ax, x, y, w, h, text, face, fs=9, weight="normal", edge=EDGE, tcol="#111", pad=0.02, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={pad},rounding_size=0.08",
                 linewidth=lw, facecolor=face, edgecolor=edge))
    if text:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, weight=weight, color=tcol)


def arr(ax, x0, y0, x1, y1, color=ARC, lw=1.3, ms=12):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                 linewidth=lw, color=color, shrinkA=0, shrinkB=0))


def grid_icon(ax, cx, cy, ncol, nrow, cell, color="#5b5b5b", lw=0.7, fill="#fbf2cf"):
    w = ncol*cell; h = nrow*cell
    x0 = cx - w/2; y0 = cy - h/2
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=fill, edgecolor=color, linewidth=lw))
    for i in range(1, ncol):
        ax.plot([x0+i*cell, x0+i*cell], [y0, y0+h], color=color, lw=lw*0.7)
    for j in range(1, nrow):
        ax.plot([x0, x0+w], [y0+j*cell, y0+j*cell], color=color, lw=lw*0.7)


# ===================================================================== #
# MAIN  high-level chart
# ===================================================================== #
def main_chart():
    fw, fh = 10.0, 12.6
    fig, ax = plt.subplots(figsize=(fw, fh)); ax.set_xlim(0, fw); ax.set_ylim(0, fh); ax.axis("off")
    cx = fw/2
    bx0, bw = 1.0, fw-2.0           # full-width band geometry
    L, M, R = bx0+bw*0.22, cx, bx0+bw*0.78   # 3 scale lanes

    # ---- Input ----
    y = fh-0.4
    iw = 4.8
    rbox(ax, cx-iw/2, y-0.62, iw, 0.62, "Input image   3 x 288 x 384", C_INPUT, fs=12, weight="bold")
    arr(ax, cx, y-0.62, cx, y-1.02)
    y -= 1.02

    # ---- Backbone ----
    bb_h = 1.5
    rbox(ax, bx0, y-bb_h, bw, bb_h, "", C_BACK_BG, edge=C_BACK, lw=1.2)
    ax.text(bx0+0.2, y-0.3, "BACKBONE", ha="left", va="center", fontsize=13, weight="bold", color="#2a4d6e")
    chips = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]
    n = len(chips); cw = 1.35; gap = (bw-0.5 - n*cw)/(n-1); x0 = bx0+0.25; ccy = y-bb_h+0.38
    for i, t in enumerate(chips):
        x = x0+i*(cw+gap)
        rbox(ax, x, ccy, cw, 0.72, t, C_BACK, fs=11, weight="bold", tcol="white")
        if i > 0:
            arr(ax, x-gap+0.03, ccy+0.36, x-0.03, ccy+0.36, lw=1.0, ms=9)
    y -= bb_h

    # ---- 3 scale arrows into neck ----
    lane_lbl = [(L, "stride 8"), (M, "stride 16"), (R, "stride 32")]
    for lx, lbl in lane_lbl:
        arr(ax, lx, y-0.05, lx, y-0.9, lw=1.6)
        ax.text(lx+0.14, y-0.47, lbl, ha="left", va="center", fontsize=10.5, style="italic", color="#444")
    y -= 0.95

    # ---- Neck ----
    nk_h = 1.05
    rbox(ax, bx0, y-nk_h, bw, nk_h, "", C_NECK_BG, edge=C_NECK, lw=1.2)
    ax.text(bx0+0.2, y-0.3, "NECK", ha="left", va="center", fontsize=13, weight="bold", color="#2f5d28")
    ax.text(cx, y-nk_h*0.42, "fuses the 3 scales", ha="center", va="center",
            fontsize=10, style="italic", color="#3a6b32")
    ax.text(cx, y-nk_h*0.76, "FPN-lite  +  PAN-lite", ha="center", va="center",
            fontsize=15, weight="bold", color="#2f5d28")
    y -= nk_h
    # neck outputs are STILL 3 scales -> label them again
    for lx, lbl in lane_lbl:
        arr(ax, lx, y-0.05, lx, y-0.9, lw=1.6)
        ax.text(lx+0.14, y-0.47, lbl + "  (96 ch)", ha="left", va="center",
                fontsize=9.5, style="italic", color="#444")
    y -= 0.95

    # ---- Head ----
    hd_h = 1.2
    rbox(ax, bx0, y-hd_h, bw, hd_h, "", C_HEAD_BG, edge=C_HEAD, lw=1.2)
    ax.text(bx0+0.2, y-0.3, "DETECTION HEAD", ha="left", va="center", fontsize=13, weight="bold", color="#9a5410")
    for lx, _ in lane_lbl:
        rbox(ax, lx-0.78, y-hd_h+0.18, 1.56, 0.62, "2x Conv 3x3\n->  Conv 1x1 -> 5",
             C_HEAD, fs=9.5, weight="bold", tcol="white")
    y -= hd_h
    for lx, _ in lane_lbl:
        arr(ax, lx, y-0.05, lx, y-0.75, lw=1.6)
    y -= 0.8

    # ---- raw outputs: grids of different density (schematic) ----
    grids = [(L, 9, 7, "stride 8\n36 x 48"), (M, 5, 4, "stride 16\n18 x 24"), (R, 2, 2, "stride 32\n9 x 12")]
    for lx, nc, nr, lbl in grids:
        grid_icon(ax, lx, y-0.4, nc, nr, 0.1)
        ax.text(lx, y-0.95, lbl, ha="center", va="center", fontsize=10, color="#7a6410", weight="bold")
    ax.text(cx, y-1.35, "2268 bboxes  ( [ x, y, w, h, obj ] )",
            ha="center", va="center", fontsize=11, weight="bold", color="#7a6410")
    y -= 2.0

    # ---- NPU | CPU divider ----
    ax.plot([bx0, bx0+bw], [y, y], ls=(0, (5, 3)), color="#999", lw=1.1)
    ax.text(bx0+0.05, y+0.13, "NPU  (Neural-ART)", ha="left", va="bottom", fontsize=10, style="italic", color="#777")
    ax.text(bx0+bw-0.05, y-0.13, "CPU", ha="right", va="top", fontsize=10, style="italic", color="#777")
    arr(ax, cx, y-0.02, cx, y-0.45, lw=1.6)
    y -= 0.53

    # ---- Post-processing ----
    pp_h = 0.88
    rbox(ax, bx0, y-pp_h, bw, pp_h, "", C_POST_BG, edge=C_POST, lw=1.2)
    ax.text(cx, y-pp_h*0.38, "POST-PROCESSING", ha="center", va="center", fontsize=12.5, weight="bold", color="#6e3d63")
    ax.text(cx, y-pp_h*0.78, "decode  ->  conf >= 0.50  ->  NMS @ IoU 0.25  ->  ball boxes",
            ha="center", va="center", fontsize=10.5, color="#6e3d63")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, "ball_styolo_nano_architecture")


# ===================================================================== #
# DETAIL  backbone (one stage + residual bottleneck)
# ===================================================================== #
def detail_backbone():
    fw, fh = 12.5, 7.0
    fig, ax = plt.subplots(figsize=(fw, fh)); ax.set_xlim(0, fw); ax.set_ylim(0, fh); ax.axis("off")
    ax.text(fw/2, fh-0.35, "Backbone — full stack & residual bottleneck", ha="center", fontsize=14, weight="bold")

    # full backbone strip (top)
    stages = [
        ("Stage 1\nStem", "Conv3x3 /2", "3 -> 32", "s2"),
        ("Stage 2", "Conv3x3 /2 + 1x bneck", "32 -> 48", "s4"),
        ("Stage 3", "Conv3x3 /2 + 2x bneck", "48 -> 64", "s8  -> tap"),
        ("Stage 4", "Conv3x3 /2 + 2x bneck", "64 -> 128", "s16 -> tap"),
        ("Stage 5", "Conv3x3 /2 + 2x bneck", "128 -> 192", "s32 -> tap"),
    ]
    n = len(stages); cw = 2.0; gap = (fw-1.2 - n*cw)/(n-1); x0 = 0.6; yb = 5.0
    for i, (nm, op, ch, st) in enumerate(stages):
        x = x0+i*(cw+gap)
        tap = "-> tap" in st
        rbox(ax, x, yb, cw, 1.1, f"{nm}\n\n{op}\n{ch}", C_BACK, fs=8.5, weight="bold", tcol="white",
             edge="#e45756" if tap else EDGE, lw=2.0 if tap else 1.0)
        ax.text(x+cw/2, yb-0.25, st, ha="center", fontsize=8.5, color="#b03030" if tap else "#444",
                weight="bold" if tap else "normal")
        if i > 0:
            arr(ax, x-gap+0.03, yb+0.55, x-0.03, yb+0.55, lw=1.1, ms=10)
    ax.text(0.6, yb+1.45, "spatial size halves each stage  ·  channels grow  ·  Stages 3/4/5 feed the neck",
            ha="left", fontsize=9.5, style="italic", color="#444")

    # zoom: residual bottleneck
    ax.text(fw/2, 3.7, "Residual bottleneck  (repeated inside each stage)", ha="center", fontsize=12, weight="bold")
    seq = [("Conv 1x1\nC -> C/2", C_BACK, "white"), ("Conv 3x3\nC/2 -> C/2", C_BACK, "white"),
           ("Conv 1x1\nC/2 -> C", C_BNECK, "#111"), ("(+) add\nskip", C_NECK, "white"), ("ReLU", C_HEAD, "white")]
    bw_ = 1.9; bgap = 0.55; tot = len(seq)*bw_ + (len(seq)-1)*bgap; sx = (fw-tot)/2; yy = 1.7
    centers = []
    for i, (t, f, tc) in enumerate(seq):
        x = sx+i*(bw_+bgap)
        rbox(ax, x, yy, bw_, 0.95, t, f, fs=8.5, weight="bold", tcol=tc)
        centers.append((x, x+bw_))
        if i > 0:
            arr(ax, centers[i-1][1]+0.03, yy+0.47, x-0.03, yy+0.47, lw=1.1, ms=10)
    # skip arc from input to add
    x_in = sx; x_add = sx+3*(bw_+bgap)
    ax.add_patch(FancyArrowPatch((x_in+0.2, yy+0.95), (x_add+bw_/2, yy+0.95),
                 connectionstyle="arc3,rad=-0.45", arrowstyle="-|>", mutation_scale=12,
                 color="#888", lw=1.3, linestyle="--"))
    ax.text((x_in+x_add)/2, yy+1.75, "skip connection", ha="center", fontsize=9, style="italic", color="#777")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02)
    save(fig, "detail_backbone")


# ===================================================================== #
# DETAIL  neck  (FPN top-down + PAN bottom-up)
# ===================================================================== #
def detail_neck():
    fw, fh = 11.0, 7.2
    fig, ax = plt.subplots(figsize=(fw, fh)); ax.set_xlim(0, fw); ax.set_ylim(0, fh); ax.axis("off")
    ax.text(fw/2, fh-0.35, "Neck — FPN-lite (top-down)  +  PAN-lite (bottom-up)", ha="center", fontsize=14, weight="bold")

    # rows = scales (top = coarse s32, bottom = fine s8); cols = stage
    yC, yM, yF = 4.7, 3.4, 2.1            # s32 / s16 / s8 rows
    xin, xlat, xfpn, xpan = 1.7, 3.9, 6.6, 9.3
    def node(x, y, txt, f, fs=8):
        rbox(ax, x-0.85, y-0.42, 1.7, 0.84, txt, f, fs=fs, weight="bold",
             tcol="white" if f != C_OUT else "#111")
    # column labels
    for x, t in [(xin, "backbone\ntaps"), (xlat, "1x1 lateral\n-> 96"), (xfpn, "FPN top-down"), (xpan, "PAN bottom-up\n(outputs)")]:
        ax.text(x, 5.7, t, ha="center", fontsize=9.5, weight="bold", color="#333")
    for y, lab in [(yC, "s32"), (yM, "s16"), (yF, "s8")]:
        ax.text(0.45, y, lab, ha="center", va="center", fontsize=9, weight="bold", color="#555")

    # inputs
    node(xin, yC, "C5  192", C_BACK); node(xin, yM, "C4  128", C_BACK); node(xin, yF, "C3  64", C_BACK)
    # laterals
    for y in (yC, yM, yF):
        node(xlat, y, "96", C_NECK); arr(ax, xin+0.85, y, xlat-0.85, y, lw=1.2)
    # FPN top-down
    node(xfpn, yC, "P5  96", C_NECK)
    node(xfpn, yM, "P4  96", C_NECK)
    node(xfpn, yF, "P3  96", C_NECK)
    arr(ax, xlat+0.85, yC, xfpn-0.85, yC, lw=1.2)
    # down arrows (upsample+concat)
    arr(ax, xfpn, yC-0.42, xfpn, yM+0.42, color="#c0392b", lw=1.6)
    arr(ax, xfpn, yM-0.42, xfpn, yF+0.42, color="#c0392b", lw=1.6)
    arr(ax, xlat+0.85, yM, xfpn-0.85, yM, lw=1.0, color="#999")
    arr(ax, xlat+0.85, yF, xfpn-0.85, yF, lw=1.0, color="#999")
    ax.text(xfpn+0.95, (yC+yM)/2, "up x2\n+ concat", ha="left", va="center", fontsize=7.5, color="#c0392b")
    # PAN bottom-up
    node(xpan, yC, "N5  96", C_OUT); node(xpan, yM, "N4  96", C_OUT); node(xpan, yF, "P3  96", C_OUT)
    arr(ax, xfpn+0.85, yF, xpan-0.85, yF, lw=1.2)
    arr(ax, xpan, yF+0.42, xpan, yM-0.42, color="#1f6f3f", lw=1.6)
    arr(ax, xpan, yM+0.42, xpan, yC-0.42, color="#1f6f3f", lw=1.6)
    arr(ax, xfpn+0.85, yM, xpan-0.85, yM, lw=1.0, color="#999")
    arr(ax, xfpn+0.85, yC, xpan-0.85, yC, lw=1.0, color="#999")
    ax.text(xpan+0.95, (yF+yM)/2, "down /2\n+ concat", ha="left", va="center", fontsize=7.5, color="#1f6f3f")
    ax.text(fw/2, 0.9, "all scales unified to 96 channels  ->  P3 / N4 / N5  go to the detection head",
            ha="center", fontsize=10, style="italic", color="#444")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02)
    save(fig, "detail_neck")


# ===================================================================== #
# DETAIL  head
# ===================================================================== #
def detail_head():
    fw, fh = 11.0, 6.4
    fig, ax = plt.subplots(figsize=(fw, fh)); ax.set_xlim(0, fw); ax.set_ylim(0, fh); ax.axis("off")
    ax.text(fw/2, fh-0.35, "Detection head — one per scale (shared structure), anchor-free, single class",
            ha="center", fontsize=13.5, weight="bold")

    lanes = [(2.0, "stride 8", "96 x 36 x 48", "5 x 36 x 48", 6, 5),
             (5.5, "stride 16", "96 x 18 x 24", "5 x 18 x 24", 4, 3),
             (9.0, "stride 32", "96 x 9 x 12", "5 x 9 x 12", 2, 2)]
    for x, st, fin, fout, nc, nr in lanes:
        ax.text(x, 5.5, st, ha="center", fontsize=10.5, weight="bold", color="#9a5410")
        rbox(ax, x-1.0, 4.7, 2.0, 0.45, f"in  {fin}", C_NECK, fs=8, weight="bold", tcol="white")
        arr(ax, x, 4.7, x, 4.35)
        rbox(ax, x-1.0, 3.75, 2.0, 0.55, "Conv3x3  96->96\nx 2  (stem)", C_HEAD, fs=8, weight="bold", tcol="white")
        arr(ax, x, 3.75, x, 3.4)
        rbox(ax, x-1.0, 2.95, 2.0, 0.45, "Conv1x1  96->5", C_BACK, fs=8.5, weight="bold", tcol="white")
        arr(ax, x, 2.95, x, 2.6)
        grid_icon(ax, x, 2.0, nc, nr, 0.13)
        ax.text(x, 1.25, f"out {fout}", ha="center", fontsize=8.5, weight="bold", color="#7a6410")
    ax.text(fw/2, 0.5, "5 channels per cell = [ tx, ty, tw, th, objectness ]   ·   2 268 cells total",
            ha="center", fontsize=10, style="italic", color="#444")
    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.03)
    save(fig, "detail_head")


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    main_chart()
    detail_backbone()
    detail_neck()
    detail_head()
