from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from torch.utils.data import DataLoader

from ball_detection.src.datasets.visualization import save_prediction_preview
from ball_detection.src.training.losses import decode_outputs
from ball_detection.src.training.metrics import (
    build_image_predictions,
    build_threshold_grid,
    compute_detection_metrics,
    pick_best_sweep_row,
    run_conf_threshold_sweep,
    write_sweep_csv,
)



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

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.bfloat16):
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
    epoch: int | None = None,
    threshold_sweep_cfg: Dict[str, Any] | None = None,
    sweep_csv_dir: Path | None = None,
    n_preview: int = 8,
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

    # Sweep is enabled when the config opts in AND the current epoch matches every_n_epochs.
    sweep_enabled = False
    if threshold_sweep_cfg is not None and bool(threshold_sweep_cfg.get("enabled", False)):
        every_n = int(threshold_sweep_cfg.get("every_n_epochs", 1))
        every_n = max(1, every_n)
        sweep_enabled = epoch is None or (epoch % every_n == 0)

    decoded_boxes_chunks: list[torch.Tensor] = []
    decoded_scores_chunks: list[torch.Tensor] = []

    # Per-epoch RNG keeps preview selection deterministic for that epoch but varied across epochs.
    preview_rng = random.Random((int(epoch) if epoch is not None else 0) * 1000003 + 7)

    # Decide which batch indices will contribute a preview image. len(loader) is the total batch count;
    # if max_batches caps it, we only sample from the capped range. One random image is saved per chosen batch.
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, int(max_batches))
    if preview_dir is not None and total_batches > 0 and n_preview > 0:
        k = min(n_preview, total_batches)
        preview_batch_indices = set(preview_rng.sample(range(total_batches), k))
    else:
        preview_batch_indices = set()

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

        if sweep_enabled:
            decoded_boxes_chunks.append(decoded_boxes.detach().cpu())
            decoded_scores_chunks.append(decoded_scores.detach().cpu())

        if preview_dir is not None and b_idx in preview_batch_indices:
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
            n_in_batch = int(images.shape[0])
            offset = preview_rng.randint(0, max(0, n_in_batch - 1))
            save_prediction_preview(
                images=images[offset:offset + 1].detach().cpu(),
                pred_boxes=[op_preds_preview[offset].boxes],
                targets=[targets[offset]],
                out_dir=preview_dir,
                prefix=f"val_pred_b{b_idx:03d}",
                max_images=1,
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

    # Threshold sweep -- reuses the decoded outputs from this epoch's validation pass.
    # Adds five keys: best_conf_threshold, best_conf_{precision,recall,f1,fp_per_image}.
    if sweep_enabled and decoded_boxes_chunks:
        decoded_boxes_all = torch.cat(decoded_boxes_chunks, dim=0)
        decoded_scores_all = torch.cat(decoded_scores_chunks, dim=0)

        conf_grid = build_threshold_grid(
            start=float(threshold_sweep_cfg.get("start", 0.05)),
            end=float(threshold_sweep_cfg.get("end", 0.95)),
            step=float(threshold_sweep_cfg.get("step", 0.05)),
        )
        sweep_rows = run_conf_threshold_sweep(
            decoded_boxes_all=decoded_boxes_all,
            decoded_scores_all=decoded_scores_all,
            targets_all=targets_all,
            conf_grid=conf_grid,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            use_nms=use_nms,
            single_object_mode=single_object_mode,
            iou_thresholds=iou_thresholds,
            matching_iou_threshold=matching_iou_threshold,
            small_area_threshold=small_area_threshold,
            medium_area_threshold=medium_area_threshold,
            ap_conf_threshold=ap_conf_threshold,
            ap_max_detections=ap_max_detections,
            ap_use_nms=ap_use_nms,
        )
        objective = str(threshold_sweep_cfg.get("objective", "f1")).lower()
        best_row = pick_best_sweep_row(sweep_rows, objective)
        metric_dict["best_conf_threshold"] = float(best_row["conf_threshold"])
        metric_dict["best_conf_precision"] = float(best_row["precision"])
        metric_dict["best_conf_recall"] = float(best_row["recall"])
        metric_dict["best_conf_f1"] = float(best_row["f1"])
        metric_dict["best_conf_fp_per_image"] = float(best_row["fp_per_image"])

        if sweep_csv_dir is not None and bool(threshold_sweep_cfg.get("save_csv", True)):
            tag = f"epoch_{int(epoch):03d}" if epoch is not None else "latest"
            write_sweep_csv(sweep_csv_dir / f"{tag}.csv", sweep_rows)

    return metric_dict
