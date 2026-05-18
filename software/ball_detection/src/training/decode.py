from __future__ import annotations

from typing import Sequence, Tuple

import torch

from ball_detection.src.utils.boxes import clamp_xyxy


def _make_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return xs, ys


def decode_single_scale(pred: torch.Tensor, stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode one head output into absolute xyxy boxes in resized image space.

    pred shape: [B, 5, H, W]
    returns:
      boxes [B, 4, H, W]
      scores [B, 1, H, W]
    """
    return decode_single_scale_with_params(
        pred,
        stride,
        decode_twth_clamp_min=-4.0,
        decode_twth_clamp_max=4.0,
    )


def decode_outputs(
    raw_outputs: Tuple[torch.Tensor, ...],
    strides: Sequence[int],
    input_size: tuple[int, int] | None = None,
    decode_twth_clamp_min: float = -4.0,
    decode_twth_clamp_max: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode raw outputs from all scales and concatenate.

    returns:
      boxes: [B, N, 4]
      scores: [B, N]
    """
    boxes_all: list[torch.Tensor] = []
    scores_all: list[torch.Tensor] = []

    for pred, stride in zip(raw_outputs, strides):
        boxes_s, scores_s = decode_single_scale_with_params(
            pred,
            stride,
            decode_twth_clamp_min=decode_twth_clamp_min,
            decode_twth_clamp_max=decode_twth_clamp_max,
        )
        b = boxes_s.shape[0]
        boxes_all.append(boxes_s.permute(0, 2, 3, 1).reshape(b, -1, 4))
        scores_all.append(scores_s.reshape(b, -1))

    boxes = torch.cat(boxes_all, dim=1)
    scores = torch.cat(scores_all, dim=1)

    if input_size is not None:
        in_h, in_w = input_size
        boxes = clamp_xyxy(boxes, width=in_w, height=in_h)

    return boxes, scores


def decode_single_scale_with_params(
    pred: torch.Tensor,
    stride: int,
    decode_twth_clamp_min: float,
    decode_twth_clamp_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, _, h, w = pred.shape
    device = pred.device
    dtype = pred.dtype

    tx = pred[:, 0, :, :]
    ty = pred[:, 1, :, :]
    tw = pred[:, 2, :, :]
    th = pred[:, 3, :, :]
    tobj = pred[:, 4, :, :]

    gx, gy = _make_grid(h, w, device=device, dtype=dtype)

    cx = (torch.sigmoid(tx) + gx.unsqueeze(0)) * float(stride)
    cy = (torch.sigmoid(ty) + gy.unsqueeze(0)) * float(stride)

    pw = torch.exp(torch.clamp(tw, min=float(decode_twth_clamp_min), max=float(decode_twth_clamp_max))) * float(stride)
    ph = torch.exp(torch.clamp(th, min=float(decode_twth_clamp_min), max=float(decode_twth_clamp_max))) * float(stride)

    x1 = cx - 0.5 * pw
    y1 = cy - 0.5 * ph
    x2 = cx + 0.5 * pw
    y2 = cy + 0.5 * ph

    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    scores = torch.sigmoid(tobj).unsqueeze(1)
    return boxes, scores
