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
    cfg: Dict[str, Any],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "run_id": run_id,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            #"cfg": cfg,            # NOTE: isse with pytorch
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )