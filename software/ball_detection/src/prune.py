"""
Structured channel pruning for BallSTYOLONano via torch-pruning.

The library traces the model's forward graph and propagates each pruning
decision through coupled layers — residual skip connections, neck concats,
and the parallel head branches — so we don't have to hand-track them.

The head's three `pred*` convs have out_channels=5 (4 box deltas + objectness)
and must NOT be pruned; they are added to `ignored_layers`.

The result is a smaller, dense model — no masks, no wrappers — so existing
training/export code paths continue to work unchanged.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

import torch_pruning as tp

from ball_detection.src.models.styolo.model import BallSTYOLONano


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _ignored_pred_convs(model: BallSTYOLONano) -> list[nn.Module]:
    """Pred convs (out_channels=5) must keep their output dim."""
    return [m for m in model.head.modules()
            if isinstance(m, nn.Conv2d) and m.out_channels == 5]


def prune_styolo(
    model: BallSTYOLONano,
    prune_ratio: float,
    example_input: torch.Tensor,
    p_norm: int = 1,
) -> Tuple[BallSTYOLONano, int, int]:
    """
    Prune `prune_ratio` (in (0, 1)) of channels per layer using L`p_norm`-norm
    magnitude importance. `example_input` shape must match the model's
    expected (B, C, H, W); strides require H, W divisible by 32.

    Returns (model, params_before, params_after). The model is modified
    in place — same object is returned for convenience.
    """
    if not (0.0 < prune_ratio < 1.0):
        raise ValueError(f"prune_ratio must be in (0, 1), got {prune_ratio}")

    n_before = count_params(model)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_input,
        importance=tp.importance.MagnitudeImportance(p=p_norm),
        pruning_ratio=prune_ratio,
        ignored_layers=_ignored_pred_convs(model),
    )
    pruner.step()
    n_after = count_params(model)
    return model, n_before, n_after
