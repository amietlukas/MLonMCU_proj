"""
Visualization helpers for transform previews.
Supports both grayscale (1-channel) and RGB (3-channel) images.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def _tensor_to_img01(x: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized image tensor back to a [0,1] numpy array for plotting.
      - [1, H, W] → [H, W]      (grayscale)
      - [3, H, W] → [H, W, 3]   (RGB)
      - [H, W]    → [H, W]      (already 2-D)
    """
    x = x.detach().cpu().float()
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]                          # [H, W]
    elif x.ndim == 3 and x.shape[0] == 3:
        x = x.permute(1, 2, 0)            # [H, W, 3]

    # rescale to [0, 1] for visualization (undo normalization)
    x_min, x_max = float(x.min()), float(x.max())
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = torch.zeros_like(x)
    return x.numpy()


def _imshow_auto(ax, img_np: np.ndarray) -> None:
    """Show a numpy image on an axis, choosing gray cmap for 2-D arrays."""
    if img_np.ndim == 2:
        ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img_np)


# ── existing preview: raw | transformed pairs per class ──────────────────────
def save_transform_preview(
    *,
    train_samples: List[Tuple[Path, int]],
    class_names: List[str],
    transform,
    out_path: Path,
    n_per_class: int = 5,
    seed: int = 0,
) -> None:
    """
    Grid: for each class → n images, each shown as (raw | transformed).
    Works for both grayscale and RGB transforms.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # group samples by class idx
    by_cls = {i: [] for i in range(len(class_names))}
    for p, y in train_samples:
        by_cls[y].append(p)

    rows = len(class_names)
    cols = n_per_class * 2  # raw + transformed per example

    fig_w = max(12, cols * 1.5)
    fig_h = max(10, rows * 1.8)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for r, cls_name in enumerate(class_names):
        paths = list(by_cls[r])
        if len(paths) == 0:
            for c in range(cols):
                axes[r, c].axis("off")
            continue

        rng.shuffle(paths)
        paths = paths[:n_per_class]

        for i, img_path in enumerate(paths):
            raw = Image.open(img_path).convert("RGB")

            # ---- raw ----
            ax_raw = axes[r, 2 * i]
            ax_raw.imshow(raw)
            ax_raw.axis("off")
            if i == 0:
                ax_raw.set_title(f"{cls_name} (raw)", fontsize=10)

            # ---- transformed ----
            x = transform(raw)
            ax_tf = axes[r, 2 * i + 1]
            _imshow_auto(ax_tf, _tensor_to_img01(x))
            ax_tf.axis("off")
            if i == 0:
                ax_tf.set_title(f"{cls_name} (tf)", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── new: augmentation variety preview ────────────────────────────────────────
def save_augmentation_preview(
    *,
    train_samples: List[Tuple[Path, int]],
    class_names: List[str],
    transform,
    out_path: Path,
    n_augmentations: int = 8,
    seed: int = 0,
) -> None:
    """
    For each class pick ONE image, show it raw in column 0, then
    *n_augmentations* different random augmented versions in the remaining
    columns.  This lets you see the variety produced by a stochastic transform.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    by_cls = {i: [] for i in range(len(class_names))}
    for p, y in train_samples:
        by_cls[y].append(p)

    rows = len(class_names)
    cols = 1 + n_augmentations  # 1 raw + N augmented

    fig_w = max(12, cols * 1.6)
    fig_h = max(8, rows * 1.8)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for r, cls_name in enumerate(class_names):
        paths = list(by_cls[r])
        if len(paths) == 0:
            for c in range(cols):
                axes[r, c].axis("off")
            continue

        # pick one representative image
        img_path = rng.choice(paths)
        raw = Image.open(img_path).convert("RGB")

        # column 0: raw
        axes[r, 0].imshow(raw)
        axes[r, 0].axis("off")
        axes[r, 0].set_title(f"{cls_name} (raw)", fontsize=10)

        # columns 1..N: same image, different random augmentations
        for j in range(1, cols):
            x = transform(raw)  # stochastic → different each call
            axes[r, j].axis("off")
            _imshow_auto(axes[r, j], _tensor_to_img01(x))
            if r == 0:
                axes[r, j].set_title(f"aug #{j}", fontsize=9)

    plt.suptitle("Augmentation variety (same source image per row)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)