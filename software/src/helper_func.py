"""
Helper utility functions for the project.
"""

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict


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
