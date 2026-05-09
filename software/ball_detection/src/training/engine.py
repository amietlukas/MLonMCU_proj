from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from torch.utils.data import DataLoader

from ball_detection.src.datasets.visualization import save_prediction_preview
from ball_detection.src.training.losses import decode_outputs
from ball_detection.src.training.metrics import build_image_predictions, compute_detection_metrics



def _move_targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    out = []
    for t in targets:
        t_new = dict(t)
        t_new["boxes"] = t["boxes"].to(device)
        t_new["labels"] = t["labels"].to(device)
        out.append(t_new)
    return out


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    amp_enabled: bool,
    grad_clip_norm: float,
    max_batches: int | None,
) -> Dict[str, float]:
    model.train()

    loss_sum = 0.0
    obj_sum = 0.0
    box_sum = 0.0
    pos_sum = 0.0
    n_steps = 0

    t0 = time.time()

    for b_idx, (images, targets) in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets_dev = _move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            losses = criterion(outputs, targets_dev)
            loss = losses["total_loss"]

        scaler.scale(loss).backward()

        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        loss_sum += float(losses["total_loss"].detach().item())
        obj_sum += float(losses["obj_loss"].detach().item())
        box_sum += float(losses["box_loss"].detach().item())
        pos_sum += float(losses["num_pos"].detach().item())
        n_steps += 1

    dt = time.time() - t0
    denom = max(n_steps, 1)

    return {
        "total_loss": loss_sum / denom,
        "obj_loss": obj_sum / denom,
        "box_loss": box_sum / denom,
        "num_pos": pos_sum / denom,
        "time_sec": dt,
    }


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    *,
    strides: Sequence[int],
    input_size: tuple[int, int],
    conf_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    use_nms: bool,
    single_object_mode: bool,
    ap_conf_threshold: float,
    ap_max_detections: int,
    ap_use_nms: bool,
    iou_thresholds: Sequence[float],
    matching_iou_threshold: float,
    small_area_threshold: float,
    medium_area_threshold: float,
    decode_twth_clamp_min: float,
    decode_twth_clamp_max: float,
    max_batches: int | None,
    preview_dir: Path | None,
) -> Dict[str, float]:
    model.eval()

    loss_sum = 0.0
    obj_sum = 0.0
    box_sum = 0.0
    pos_sum = 0.0
    n_steps = 0

    map_preds_all = []
    op_preds_all = []
    targets_all = []

    for b_idx, (images, targets) in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets_dev = _move_targets_to_device(targets, device)

        outputs = model(images)
        losses = criterion(outputs, targets_dev)

        loss_sum += float(losses["total_loss"].detach().item())
        obj_sum += float(losses["obj_loss"].detach().item())
        box_sum += float(losses["box_loss"].detach().item())
        pos_sum += float(losses["num_pos"].detach().item())
        n_steps += 1

        decoded_boxes, decoded_scores = decode_outputs(
            outputs,
            strides=strides,
            input_size=input_size,
            decode_twth_clamp_min=decode_twth_clamp_min,
            decode_twth_clamp_max=decode_twth_clamp_max,
        )

        # AP set uses low confidence threshold to build score ranking.
        map_preds = build_image_predictions(
            decoded_boxes=decoded_boxes,
            decoded_scores=decoded_scores,
            targets=targets,
            conf_threshold=ap_conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=ap_max_detections,
            use_nms=ap_use_nms,
            single_object_mode=False,
            return_in_original=True,
        )
        op_preds = build_image_predictions(
            decoded_boxes=decoded_boxes,
            decoded_scores=decoded_scores,
            targets=targets,
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            use_nms=use_nms,
            single_object_mode=single_object_mode,
            return_in_original=True,
        )

        map_preds_all.extend(map_preds)
        op_preds_all.extend(op_preds)
        targets_all.extend(targets)

        if preview_dir is not None and b_idx == 0:
            op_preds_preview = build_image_predictions(
                decoded_boxes=decoded_boxes,
                decoded_scores=decoded_scores,
                targets=targets,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections,
                use_nms=use_nms,
                single_object_mode=single_object_mode,
                return_in_original=False,
            )
            save_prediction_preview(
                images=images.detach().cpu(),
                pred_boxes=[p.boxes for p in op_preds_preview],
                targets=targets,
                out_dir=preview_dir,
                prefix="val_pred",
                max_images=min(8, images.shape[0]),
            )

    metric_dict = compute_detection_metrics(
        map_predictions=map_preds_all,
        operating_predictions=op_preds_all,
        targets_all=targets_all,
        iou_thresholds=iou_thresholds,
        matching_iou_threshold=matching_iou_threshold,
        small_area_threshold=small_area_threshold,
        medium_area_threshold=medium_area_threshold,
    )

    denom = max(n_steps, 1)
    metric_dict.update(
        {
            "total_loss": loss_sum / denom,
            "obj_loss": obj_sum / denom,
            "box_loss": box_sum / denom,
            "num_pos": pos_sum / denom,
        }
    )

    return metric_dict
