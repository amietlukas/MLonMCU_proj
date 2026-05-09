from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg: str) -> torch.device:
    device_arg = device_arg.lower().strip()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError("--device must be one of: auto, cpu, cuda")


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
