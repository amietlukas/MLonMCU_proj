"""
Helper utility functions for the project.
"""

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image


def compute_rgb_mean_std(
    samples: List[Tuple[Path, int]],
    size: int = 128,
    fill: int = 0,
) -> Tuple[List[float], List[float]]:
    """Compute per-channel mean and std over a list of (path, label) samples.

    Images are loaded as RGB, letterbox-resized to *size*×*size*, and converted
    to [0, 1] tensors (no normalisation).  The statistics are accumulated with
    Welford-style running sums so that only one pass is needed.

    Returns
    -------
    mean : list of 3 floats  [R, G, B]
    std  : list of 3 floats  [R, G, B]
    """
    from src.data import LetterboxSquare          # avoid circular import

    tf = T.Compose([LetterboxSquare(size=size, fill=fill), T.ToTensor()])

    n_pixels = 0
    ch_sum = torch.zeros(3, dtype=torch.float64)
    ch_sq  = torch.zeros(3, dtype=torch.float64)

    for i, (path, _) in enumerate(samples):
        img = Image.open(path).convert("RGB")
        x = tf(img).to(torch.float64)             # (3, H, W)
        npx = x.shape[1] * x.shape[2]
        ch_sum += x.sum(dim=(1, 2))
        ch_sq  += (x ** 2).sum(dim=(1, 2))
        n_pixels += npx
        if (i + 1) % 20_000 == 0:
            print(f"  [mean/std] processed {i + 1}/{len(samples)} images …")

    mean = ch_sum / n_pixels
    std  = ((ch_sq / n_pixels) - mean ** 2).sqrt()

    mean_list = [round(v, 6) for v in mean.tolist()]
    std_list  = [round(v, 6) for v in std.tolist()]

    print(f"  Dataset mean: {mean_list}")
    print(f"  Dataset std:  {std_list}")
    return mean_list, std_list


def vprint(msg: str, cfg: Dict[str, Any]) -> None:
    """Print if debug.verbose is True, otherwise silent."""
    if cfg.get("debug", {}).get("verbose", False):
        print(msg)


def param_norm(model) -> float:
    """L2 norm of all model parameters (useful for sanity-checking training)."""
    s = 0.0
    for p in model.parameters():
        s += p.detach().float().norm().item() ** 2
    return math.sqrt(s)


def infer_mapping_from_samples(samples, n: int = 5000, seed: int = 0) -> dict:
    """Sample up to *n* entries and return {label: [(folder, count), ...]} top-3."""
    rng = random.Random(seed)
    idxs = rng.sample(range(len(samples)), min(n, len(samples)))
    m: Dict[int, Counter] = defaultdict(Counter)
    for i in idxs:
        p, y = samples[i]
        cls = Path(p).parent.name
        m[int(y)][cls] += 1
    return {y: m[y].most_common(3) for y in sorted(m.keys())}
