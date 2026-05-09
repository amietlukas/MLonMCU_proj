from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from ball_detection.src.config import config_to_serializable


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    cfg: Dict[str, Any],
    best_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": config_to_serializable(cfg),
        "best_metric": float(best_metric),
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()

    torch.save(state, path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return ckpt
