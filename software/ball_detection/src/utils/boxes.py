from __future__ import annotations

from typing import Tuple

import torch


def xywh_to_xyxy(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return x, y, x + w, y + h


def xyxy_to_xywh_topleft(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.clone()
    out = boxes.clone()
    out[..., 2] = (out[..., 2] - out[..., 0]).clamp(min=0)
    out[..., 3] = (out[..., 3] - out[..., 1]).clamp(min=0)
    return out


def clamp_xyxy(box: torch.Tensor, width: int, height: int) -> torch.Tensor:
    out = box.clone()
    out[..., 0] = out[..., 0].clamp(0, width - 1)
    out[..., 1] = out[..., 1].clamp(0, height - 1)
    out[..., 2] = out[..., 2].clamp(0, width - 1)
    out[..., 3] = out[..., 3].clamp(0, height - 1)
    return out


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[..., 2:4] - boxes[..., 0:2]).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=boxes1.dtype, device=boxes1.device)

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def pairwise_iou_diag(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((0,), dtype=boxes1.dtype, device=boxes1.device)

    x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    a2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = a1 + a2 - inter
    return inter / union.clamp(min=1e-6)


def pairwise_diou_diag(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Diagonal-paired DIoU: DIoU[i] for boxes1[i] vs boxes2[i] (xyxy).

    Formula (Zheng et al. 2020, "Distance-IoU Loss"):
        DIoU = IoU - rho^2(b, b_gt) / c^2
    where rho is the L2 distance between box centers and c is the diagonal length of
    the smallest enclosing axis-aligned box covering both boxes.

    Output range is [-1, 1]: 1 = identical boxes, -> -1 as boxes get far apart.
    Use `(1 - pairwise_diou_diag(...)).mean()` as a regression loss (range [0, 2]).
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((0,), dtype=boxes1.dtype, device=boxes1.device)

    # IoU term (same as pairwise_iou_diag, recomputed here to keep this function standalone).
    inter_x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    a1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    a2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = (a1 + a2 - inter).clamp(min=eps)
    iou = inter / union

    # Center distance squared.
    cx1 = 0.5 * (boxes1[:, 0] + boxes1[:, 2])
    cy1 = 0.5 * (boxes1[:, 1] + boxes1[:, 3])
    cx2 = 0.5 * (boxes2[:, 0] + boxes2[:, 2])
    cy2 = 0.5 * (boxes2[:, 1] + boxes2[:, 3])
    center_dist_sq = (cx1 - cx2).pow(2) + (cy1 - cy2).pow(2)

    # Enclosing box diagonal squared.
    enc_x1 = torch.minimum(boxes1[:, 0], boxes2[:, 0])
    enc_y1 = torch.minimum(boxes1[:, 1], boxes2[:, 1])
    enc_x2 = torch.maximum(boxes1[:, 2], boxes2[:, 2])
    enc_y2 = torch.maximum(boxes1[:, 3], boxes2[:, 3])
    enc_w = (enc_x2 - enc_x1).clamp(min=0)
    enc_h = (enc_y2 - enc_y1).clamp(min=0)
    enc_diag_sq = enc_w.pow(2) + enc_h.pow(2) + eps

    return iou - center_dist_sq / enc_diag_sq


def centers_of_boxes(boxes: torch.Tensor) -> torch.Tensor:
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    return torch.stack([cx, cy], dim=1)


def letterbox_to_original(
    boxes: torch.Tensor,
    scale: float,
    pad_left: int,
    pad_top: int,
    orig_size: Tuple[int, int],
) -> torch.Tensor:
    return resized_to_original(
        boxes=boxes,
        scale_x=float(scale),
        scale_y=float(scale),
        pad_left=pad_left,
        pad_top=pad_top,
        orig_size=orig_size,
    )


def original_to_letterbox(
    boxes: torch.Tensor,
    scale: float,
    pad_left: int,
    pad_top: int,
) -> torch.Tensor:
    return original_to_resized(
        boxes=boxes,
        scale_x=float(scale),
        scale_y=float(scale),
        pad_left=pad_left,
        pad_top=pad_top,
    )


def resized_to_original(
    boxes: torch.Tensor,
    scale_x: float,
    scale_y: float,
    pad_left: int,
    pad_top: int,
    orig_size: Tuple[int, int],
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.clone()

    sx = max(float(scale_x), 1e-9)
    sy = max(float(scale_y), 1e-9)
    orig_h, orig_w = orig_size

    out = boxes.clone()
    out[:, [0, 2]] = (out[:, [0, 2]] - float(pad_left)) / sx
    out[:, [1, 3]] = (out[:, [1, 3]] - float(pad_top)) / sy
    out = clamp_xyxy(out, width=orig_w, height=orig_h)
    return out


def original_to_resized(
    boxes: torch.Tensor,
    scale_x: float,
    scale_y: float,
    pad_left: int,
    pad_top: int,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.clone()

    out = boxes.clone()
    out[:, [0, 2]] = out[:, [0, 2]] * float(scale_x) + float(pad_left)
    out[:, [1, 3]] = out[:, [1, 3]] * float(scale_y) + float(pad_top)
    return out
