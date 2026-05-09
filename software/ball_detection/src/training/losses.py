from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ball_detection.src.training.assigners import build_targets
from ball_detection.src.utils.boxes import clamp_xyxy, pairwise_iou_diag


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


class BallDetectionLoss(nn.Module):
    def __init__(
        self,
        strides: Sequence[int],
        obj_weight: float = 1.0,
        box_weight: float = 5.0,
        obj_loss_mode: str = "mean",
        obj_pos_weight: float = 1.0,
        obj_neg_weight: float = 1.0,
        obj_bce_pos_weight: float = 1.0,
        focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        assign_scale_target: float = 3.0,
        assign_conflict_policy: str = "largest_area",
        assign_center_radius: int = 0,
        decode_twth_clamp_min: float = -4.0,
        decode_twth_clamp_max: float = 4.0,
    ):
        super().__init__()
        self.strides = tuple(int(s) for s in strides)
        self.obj_weight = float(obj_weight)
        self.box_weight = float(box_weight)
        self.obj_loss_mode = str(obj_loss_mode).lower()
        self.obj_pos_weight = float(obj_pos_weight)
        self.obj_neg_weight = float(obj_neg_weight)
        self.obj_bce_pos_weight = float(obj_bce_pos_weight)
        self.focal_loss = bool(focal_loss)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.assign_scale_target = float(assign_scale_target)
        self.assign_conflict_policy = str(assign_conflict_policy).lower()
        self.assign_center_radius = int(assign_center_radius)
        self.decode_twth_clamp_min = float(decode_twth_clamp_min)
        self.decode_twth_clamp_max = float(decode_twth_clamp_max)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _objectness_loss(self, pred_obj: torch.Tensor, target_obj: torch.Tensor) -> torch.Tensor:
        bce = self.bce(pred_obj, target_obj)
        if self.obj_bce_pos_weight != 1.0:
            bce = torch.where(target_obj > 0.5, bce * self.obj_bce_pos_weight, bce)
        if self.focal_loss:
            prob = torch.sigmoid(pred_obj)
            p_t = prob * target_obj + (1.0 - prob) * (1.0 - target_obj)
            gamma = self.focal_gamma
            alpha = self.focal_alpha
            alpha_t = alpha * target_obj + (1 - alpha) * (1 - target_obj)
            focal_weight = alpha_t * ((1.0 - p_t) ** gamma)
            bce = bce * focal_weight

        if self.obj_loss_mode == "mean":
            return bce.mean()

        if self.obj_loss_mode == "balanced":
            pos_mask = target_obj > 0.5
            neg_mask = ~pos_mask

            zero = bce.new_tensor(0.0)
            pos_loss = bce[pos_mask].mean() if pos_mask.any() else zero
            neg_loss = bce[neg_mask].mean() if neg_mask.any() else zero
            return self.obj_pos_weight * pos_loss + self.obj_neg_weight * neg_loss

        raise ValueError(
            f"Unsupported obj_loss_mode: {self.obj_loss_mode}. "
            "Expected one of ['mean', 'balanced']."
        )

    def forward(self, raw_outputs: Tuple[torch.Tensor, ...], targets: list[dict]) -> Dict[str, torch.Tensor]:
        device = raw_outputs[0].device

        output_shapes = [(int(o.shape[2]), int(o.shape[3])) for o in raw_outputs]
        obj_targets, box_targets_abs, pos_masks, num_pos = build_targets(
            targets=targets,
            output_shapes=output_shapes,
            strides=self.strides,
            device=device,
            assign_scale_target=self.assign_scale_target,
            conflict_policy=self.assign_conflict_policy,
            center_radius=self.assign_center_radius,
        )

        obj_loss = torch.tensor(0.0, device=device)
        box_loss = torch.tensor(0.0, device=device)

        for scale_idx, (pred, stride) in enumerate(zip(raw_outputs, self.strides)):
            pred_obj = pred[:, 4:5, :, :]
            obj_t = obj_targets[scale_idx]
            obj_loss = obj_loss + self._objectness_loss(pred_obj, obj_t)

            pos_mask = pos_masks[scale_idx].squeeze(1)
            if pos_mask.any():
                decoded_boxes, _ = decode_single_scale_with_params(
                    pred,
                    stride=stride,
                    decode_twth_clamp_min=self.decode_twth_clamp_min,
                    decode_twth_clamp_max=self.decode_twth_clamp_max,
                )
                pred_boxes_flat = decoded_boxes.permute(0, 2, 3, 1)[pos_mask]
                tgt_boxes_flat = box_targets_abs[scale_idx].permute(0, 2, 3, 1)[pos_mask]
                iou = pairwise_iou_diag(pred_boxes_flat, tgt_boxes_flat)
                box_loss = box_loss + (1.0 - iou).mean()

        n_scales = max(len(raw_outputs), 1)
        obj_loss = obj_loss / n_scales
        box_loss = box_loss / n_scales

        total_loss = self.obj_weight * obj_loss + self.box_weight * box_loss

        return {
            "total_loss": total_loss,
            "obj_loss": obj_loss,
            "box_loss": box_loss,
            "num_pos": torch.tensor(float(num_pos), device=device),
        }
