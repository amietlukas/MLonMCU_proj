"""
File dedicated for all checkpointing related stuff.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import torch
import torch.nn as nn


def save_checkpoint(
    path: str | Path,
    *,
    run_id: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    cfg: Dict[str, Any],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # convert Path objects in cfg so torch.save doesn't choke
    import copy
    def _sanitize(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    payload = {
        "run_id": run_id,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": _sanitize(copy.deepcopy(cfg)),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, path)