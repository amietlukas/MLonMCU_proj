from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn

from ball_detection.src.training.assigners import SimOTALiteAssigner, build_targets
# Re-export decode helpers from this module for existing train/eval imports.
from ball_detection.src.training.decode import decode_outputs, decode_single_scale, decode_single_scale_with_params
from ball_detection.src.utils.boxes import pairwise_diou_diag, pairwise_iou_diag


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
        box_loss_type: str = "iou",
        assigner_cfg: Dict[str, Any] | None = None,
        input_size: tuple[int, int] | None = None,
        logger=None,
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
        self.box_loss_type = str(box_loss_type).lower().strip()
        if self.box_loss_type not in {"iou", "diou"}:
            raise ValueError(
                f"Unsupported box_loss_type: {self.box_loss_type}. "
                "Expected one of ['iou', 'diou']."
            )
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.input_size = input_size
        self.logger = logger
        self.assigner_cfg = dict(assigner_cfg or {})
        self.assigner_type = str(self.assigner_cfg.get("type", "center")).lower().strip()
        self._assigner_calls = 0
        self.assigner_debug_context = "train"

        if self.assigner_type == "center":
            self.simota_assigner = None
        elif self.assigner_type == "simota_lite":
            self.simota_assigner = SimOTALiteAssigner(
                strides=self.strides,
                input_size=self.input_size,
                simota_cfg=self.assigner_cfg.get("simota", {}),
                decode_twth_clamp_min=self.decode_twth_clamp_min,
                decode_twth_clamp_max=self.decode_twth_clamp_max,
                assign_scale_target=self.assign_scale_target,
            )
        else:
            raise ValueError(
                f"Unsupported assigner.type: {self.assigner_type}. "
                "Expected one of ['center', 'simota_lite']."
            )

        if self.logger is not None:
            self.logger.info(f"[INFO] assigner: {self.assigner_type}")
            self.logger.info(f"[INFO] box_loss_type: {self.box_loss_type}")

    def set_assigner_debug_context(self, context: str) -> None:
        self.assigner_debug_context = str(context).strip().lower() or "unknown"

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
        self._assigner_calls += 1
        if self.assigner_type == "center":
            obj_targets, box_targets_abs, pos_masks, num_pos = build_targets(
                targets=targets,
                output_shapes=output_shapes,
                strides=self.strides,
                device=device,
                assign_scale_target=self.assign_scale_target,
                conflict_policy=self.assign_conflict_policy,
                center_radius=self.assign_center_radius,
            )
        else:
            assert self.simota_assigner is not None
            obj_targets, box_targets_abs, pos_masks, num_pos, stats = self.simota_assigner.assign(
                raw_outputs=raw_outputs,
                targets=targets,
                output_shapes=output_shapes,
                device=device,
            )
            if self.logger is not None and self.simota_assigner.should_log_debug(self._assigner_calls):
                self.logger.info(stats.format_debug(context=self.assigner_debug_context))

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
                if self.box_loss_type == "diou":
                    score = pairwise_diou_diag(pred_boxes_flat, tgt_boxes_flat)
                else:
                    score = pairwise_iou_diag(pred_boxes_flat, tgt_boxes_flat)
                box_loss = box_loss + (1.0 - score).mean()

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
