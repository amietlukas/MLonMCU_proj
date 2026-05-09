from __future__ import annotations

import torch

from ball_detection.src.utils.boxes import box_iou


try:
    from torchvision.ops import nms as tv_nms
except Exception:  # pragma: no cover
    tv_nms = None


def nms_pure_torch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: list[int] = []

    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[i : i + 1], boxes[rest]).squeeze(0)
        order = rest[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if tv_nms is not None:
        return tv_nms(boxes, scores, iou_threshold)
    return nms_pure_torch(boxes, scores, iou_threshold)
