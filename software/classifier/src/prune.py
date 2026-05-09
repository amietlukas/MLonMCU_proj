"""
Structured L1-norm channel pruning for BaselineCNN.

For each stage, the output channels with the smallest L1 norm of the last conv's
weights are dropped. The previous stage's output channels and the next stage's
input channels are filtered consistently, and the final Linear layer's input
features are matched.

The result is a smaller BaselineCNN (no masks, no wrappers) — so existing code
paths (training loop, ONNX export, summary) work unchanged.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from classifier.src.model import BaselineCNN


def _stage_components(stage: nn.Sequential) -> Tuple[List[nn.Conv2d], List[nn.BatchNorm2d]]:
    convs, bns = [], []
    for m in stage:
        if isinstance(m, nn.Conv2d):
            convs.append(m)
        elif isinstance(m, nn.BatchNorm2d):
            bns.append(m)
    return convs, bns


def _select_keep_indices(conv: nn.Conv2d, n_keep: int) -> torch.Tensor:
    l1 = conv.weight.data.abs().sum(dim=(1, 2, 3))
    return torch.topk(l1, n_keep).indices.sort().values


def _copy_pruned_conv(src: nn.Conv2d, dst: nn.Conv2d, keep_out: torch.Tensor, keep_in: torch.Tensor | None) -> None:
    w = src.weight.data[keep_out]
    if keep_in is not None:
        w = w[:, keep_in]
    dst.weight.data = w.clone()
    if src.bias is not None and dst.bias is not None:
        dst.bias.data = src.bias.data[keep_out].clone()


def _copy_pruned_bn(src: nn.BatchNorm2d, dst: nn.BatchNorm2d, keep: torch.Tensor) -> None:
    dst.weight.data = src.weight.data[keep].clone()
    dst.bias.data = src.bias.data[keep].clone()
    dst.running_mean = src.running_mean[keep].clone()
    dst.running_var = src.running_var[keep].clone()
    dst.num_batches_tracked = src.num_batches_tracked.clone()


def prune_baseline_cnn(model: BaselineCNN, prune_ratio: float, cfg: Dict[str, Any]) -> Tuple[BaselineCNN, Dict[str, Any]]:
    """
    Prune `prune_ratio` (in (0,1)) of output channels from each stage.
    Returns (pruned_model, new_cfg) — new_cfg has updated `model.channels`.
    """
    if not (0.0 < prune_ratio < 1.0):
        raise ValueError(f"prune_ratio must be in (0, 1), got {prune_ratio}")

    keep_per_stage: List[torch.Tensor] = []
    for stage in model.features:
        convs, _ = _stage_components(stage)
        # Importance ranked by the LAST conv (its output is what feeds the next stage)
        last = convs[-1]
        n_old = last.out_channels
        n_new = max(1, int(round(n_old * (1.0 - prune_ratio))))
        keep_per_stage.append(_select_keep_indices(last, n_new))

    new_cfg = copy.deepcopy(cfg)
    new_cfg["model"]["channels"] = [k.numel() for k in keep_per_stage]

    new_model = BaselineCNN(new_cfg)
    prev_keep: torch.Tensor | None = None  # None = use original input channels for stage 0

    for i, (stage_old, stage_new) in enumerate(zip(model.features, new_model.features)):
        convs_old, bns_old = _stage_components(stage_old)
        convs_new, bns_new = _stage_components(stage_new)
        keep_out = keep_per_stage[i]

        for j, (c_old, c_new) in enumerate(zip(convs_old, convs_new)):
            in_keep = prev_keep if j == 0 else keep_out
            _copy_pruned_conv(c_old, c_new, keep_out, in_keep)

        for b_old, b_new in zip(bns_old, bns_new):
            _copy_pruned_bn(b_old, b_new, keep_out)

        prev_keep = keep_out

    last_keep = keep_per_stage[-1]
    new_model.classifier.weight.data = model.classifier.weight.data[:, last_keep].clone()
    new_model.classifier.bias.data = model.classifier.bias.data.clone()

    return new_model, new_cfg


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
