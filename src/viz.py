"""
visualizes the transform. currenty just if "in_channel=1 (grayscale)"
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
    x: [1,H,W] or [H,W] tensor, float
    returns HxW float in [0,1] for plotting
    """
    if x.ndim == 3:
        x = x[0]
    x = x.detach().cpu().float()
    # if normalized, it might be outside [0,1]; rescale for visualization
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max > x_min:
        x = (x - x_min) / (x_max - x_min)
    else:
        x = torch.zeros_like(x)
    return x.numpy()


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
    Saves a grid: for each class: n images, each with (raw | transformed).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # group samples by class idx
    by_cls = {i: [] for i in range(len(class_names))}
    for p, y in train_samples:
        by_cls[y].append(p)

    # choose examples
    chosen = []
    for y in range(len(class_names)):
        paths = by_cls[y]
        if len(paths) == 0:
            continue
        rng.shuffle(paths)
        chosen_paths = paths[:n_per_class]
        for p in chosen_paths:
            chosen.append((p, y))

    rows = len(class_names)
    cols = n_per_class * 2  # raw + transformed per example

    fig_w = max(12, cols * 1.5)
    fig_h = max(10, rows * 1.8)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    # make axes always 2D
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for r, cls_name in enumerate(class_names):
        # take the paths again in the same order we chose
        paths = by_cls[r]
        if len(paths) == 0:
            # empty row
            for c in range(cols):
                axes[r, c].axis("off")
            continue

        rng.shuffle(paths)
        paths = paths[:n_per_class]

        for i, img_path in enumerate(paths):
            # ---- raw ----
            raw = Image.open(img_path).convert("RGB")
            ax_raw = axes[r, 2 * i]
            ax_raw.imshow(raw)
            ax_raw.axis("off")
            if i == 0:
                ax_raw.set_title(f"{cls_name} (raw)", fontsize=10)

            # ---- transformed ----
            x = transform(raw)  # [1,H,W] after your pipeline
            x_img = _tensor_to_img01(x)
            ax_tf = axes[r, 2 * i + 1]
            ax_tf.imshow(x_img, cmap="gray", vmin=0, vmax=1)
            ax_tf.axis("off")
            if i == 0:
                ax_tf.set_title(f"{cls_name} (tf)", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)  