from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    software_root = Path(__file__).resolve().parent.parent
    if str(software_root) not in sys.path:
        sys.path.insert(0, str(software_root))

from ball_detection.src.config import load_config
from ball_detection.src.datasets.spl_ball_dataset import build_dataloaders
from ball_detection.src.logging_utils import setup_logger
from ball_detection.src.models import build_model
from ball_detection.src.reproducibility import resolve_device
from ball_detection.src.training.checkpoint import load_checkpoint
from ball_detection.src.training.engine import validate_one_epoch
from ball_detection.src.training.losses import BallDetectionLoss, decode_outputs
from ball_detection.src.training.metrics import build_image_predictions, compute_detection_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ball detector checkpoint")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    split_group = p.add_mutually_exclusive_group()
    split_group.add_argument(
        "--reuse-splits",
        dest="reuse_splits",
        action="store_true",
        help="Reuse split files if valid for the current dataset",
    )
    split_group.add_argument(
        "--regenerate-splits",
        dest="reuse_splits",
        action="store_false",
        help="Force regeneration of split files",
    )
    p.set_defaults(reuse_splits=None)
    p.add_argument("--sweep-conf", action="store_true", help="Sweep confidence thresholds on val set")
    p.add_argument(
        "--sweep-conf-values",
        type=str,
        default="",
        help="Comma-separated confidence values for sweep (overrides start/end/step)",
    )
    p.add_argument("--sweep-conf-start", type=float, default=0.01)
    p.add_argument("--sweep-conf-end", type=float, default=0.95)
    p.add_argument("--sweep-conf-step", type=float, default=0.01)
    p.add_argument(
        "--sweep-objective",
        type=str,
        default="f1",
        choices=["f1", "precision", "recall"],
        help="Objective used to pick recommended confidence threshold",
    )
    p.add_argument(
        "--sweep-out",
        type=str,
        default="",
        help="Optional CSV output path for threshold sweep report",
    )
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _move_targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    out = []
    for t in targets:
        t_new = dict(t)
        t_new["boxes"] = t["boxes"].to(device)
        t_new["labels"] = t["labels"].to(device)
        out.append(t_new)
    return out


def _build_threshold_grid(args: argparse.Namespace) -> list[float]:
    if args.sweep_conf_values.strip():
        vals = []
        for raw in args.sweep_conf_values.split(","):
            v = float(raw.strip())
            if 0.0 <= v <= 1.0:
                vals.append(v)
        uniq = sorted(set(vals))
        if not uniq:
            raise ValueError("No valid values parsed from --sweep-conf-values")
        return uniq

    start = float(args.sweep_conf_start)
    end = float(args.sweep_conf_end)
    step = float(args.sweep_conf_step)
    if step <= 0:
        raise ValueError("--sweep-conf-step must be > 0")
    if end < start:
        raise ValueError("--sweep-conf-end must be >= --sweep-conf-start")

    vals: list[float] = []
    cur = start
    guard = 0
    while cur <= end + 1e-12 and guard < 10000:
        vals.append(round(cur, 6))
        cur += step
        guard += 1
    uniq = sorted(set(v for v in vals if 0.0 <= v <= 1.0))
    if not uniq:
        raise ValueError("Threshold sweep grid is empty after bounds filtering")
    return uniq


@torch.no_grad()
def _collect_decoded_validation(
    *,
    model: torch.nn.Module,
    loader,
    criterion: BallDetectionLoss,
    device: torch.device,
    strides: list[int],
    input_size: tuple[int, int],
    decode_twth_clamp_min: float,
    decode_twth_clamp_max: float,
    max_batches: int | None,
):
    model.eval()
    loss_sum = 0.0
    obj_sum = 0.0
    box_sum = 0.0
    pos_sum = 0.0
    n_steps = 0

    decoded_boxes_chunks: list[torch.Tensor] = []
    decoded_scores_chunks: list[torch.Tensor] = []
    targets_all: list[dict] = []

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

        boxes, scores = decode_outputs(
            outputs,
            strides=strides,
            input_size=input_size,
            decode_twth_clamp_min=decode_twth_clamp_min,
            decode_twth_clamp_max=decode_twth_clamp_max,
        )
        decoded_boxes_chunks.append(boxes.detach().cpu())
        decoded_scores_chunks.append(scores.detach().cpu())
        targets_all.extend(targets)

    if not decoded_boxes_chunks:
        raise RuntimeError("No validation batches decoded. Check --max-val-batches / dataloader.")

    decoded_boxes_all = torch.cat(decoded_boxes_chunks, dim=0)
    decoded_scores_all = torch.cat(decoded_scores_chunks, dim=0)
    if decoded_boxes_all.shape[0] != len(targets_all):
        raise RuntimeError(
            f"Decoded sample count ({decoded_boxes_all.shape[0]}) != targets count ({len(targets_all)})"
        )

    denom = max(n_steps, 1)
    loss_metrics = {
        "total_loss": loss_sum / denom,
        "obj_loss": obj_sum / denom,
        "box_loss": box_sum / denom,
        "num_pos": pos_sum / denom,
    }
    return decoded_boxes_all, decoded_scores_all, targets_all, loss_metrics


def _write_sweep_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "conf_threshold",
        "precision",
        "recall",
        "f1",
        "fp_per_image",
        "mean_iou",
        "recall_small",
        "recall_medium",
        "recall_large",
        "map50",
        "map5095",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.reuse_splits is not None:
        cfg["dataset"]["reuse_splits"] = bool(args.reuse_splits)

    device = resolve_device(args.device)
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger = setup_logger(ckpt_path.parent.parent / "evaluate.log", debug=args.debug)

    train_loader, val_loader, info = build_dataloaders(cfg, logger=logger, num_workers_override=args.num_workers)
    _ = train_loader  # only val is used here

    model = build_model(cfg).to(device)
    load_checkpoint(ckpt_path, model=model, map_location=device)

    criterion = BallDetectionLoss(
        strides=cfg["model"]["strides"],
        obj_weight=float(cfg["loss"]["obj_weight"]),
        box_weight=float(cfg["loss"]["box_weight"]),
        obj_loss_mode=str(cfg["loss"].get("obj_loss_mode", "mean")),
        obj_pos_weight=float(cfg["loss"].get("obj_pos_weight", 1.0)),
        obj_neg_weight=float(cfg["loss"].get("obj_neg_weight", 1.0)),
        obj_bce_pos_weight=float(cfg["loss"].get("obj_bce_pos_weight", 1.0)),
        focal_loss=bool(cfg["loss"].get("focal_loss", False)),
        focal_alpha=float(cfg["loss"].get("focal_alpha", 0.25)),
        focal_gamma=float(cfg["loss"].get("focal_gamma", 2.0)),
        assign_scale_target=float(cfg["loss"].get("assign_scale_target", 3.0)),
        assign_conflict_policy=str(cfg["loss"].get("assign_conflict_policy", "largest_area")),
        assign_center_radius=int(cfg["loss"].get("assign_center_radius", 0)),
        decode_twth_clamp_min=float(cfg["loss"].get("decode_twth_clamp_min", -4.0)),
        decode_twth_clamp_max=float(cfg["loss"].get("decode_twth_clamp_max", 4.0)),
        box_loss_type=str(cfg["loss"].get("box_loss_type", "iou")),
        assigner_cfg=cfg.get("assigner", {"type": "center"}),
        input_size=(int(cfg["input"]["height"]), int(cfg["input"]["width"])),
        logger=logger,
    )

    metrics = validate_one_epoch(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        strides=cfg["model"]["strides"],
        input_size=(int(cfg["input"]["height"]), int(cfg["input"]["width"])),
        conf_threshold=float(cfg["eval"]["conf_threshold"]),
        nms_iou_threshold=float(cfg["eval"]["nms_iou_threshold"]),
        max_detections=int(cfg["eval"]["max_detections"]),
        use_nms=bool(cfg["eval"].get("use_nms", True)),
        single_object_mode=bool(cfg["eval"].get("single_object_mode", False)),
        ap_conf_threshold=float(cfg["eval"].get("ap_conf_threshold", 0.001)),
        ap_max_detections=int(cfg["eval"].get("ap_max_detections", max(200, int(cfg["eval"]["max_detections"])))),
        ap_use_nms=bool(cfg["eval"].get("ap_use_nms", True)),
        iou_thresholds=list(cfg["eval"]["iou_thresholds"]),
        matching_iou_threshold=float(cfg["eval"].get("matching_iou_threshold", 0.5)),
        small_area_threshold=float(cfg["eval"].get("small_area_threshold", 0.005)),
        medium_area_threshold=float(cfg["eval"].get("medium_area_threshold", 0.03)),
        decode_twth_clamp_min=float(cfg["loss"].get("decode_twth_clamp_min", -4.0)),
        decode_twth_clamp_max=float(cfg["loss"].get("decode_twth_clamp_max", 4.0)),
        max_batches=args.max_val_batches,
        preview_dir=ckpt_path.parent.parent / "predictions_preview_eval",
    )

    logger.info(f"[INFO] Evaluated images: {info['val_images']}")
    logger.info(
        "[INFO] Val metrics | "
        f"loss={metrics['total_loss']:.4f} map50={metrics['map50']:.4f} map5095={metrics['map5095']:.4f} "
        f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
        f"mean_iou={metrics['mean_iou']:.4f} "
        f"center_err_px(mean/med/p95)="
        f"{metrics['center_error_px']:.2f}/{metrics['center_error_px_median']:.2f}/{metrics['center_error_px_p95']:.2f} "
        f"center_err_norm_diag(mean/med/p95)="
        f"{metrics['center_error_norm_diag']:.4f}/{metrics['center_error_norm_diag_median']:.4f}/{metrics['center_error_norm_diag_p95']:.4f} "
        f"fp/img={metrics['fp_per_image']:.4f}"
    )

    if not args.sweep_conf:
        return

    logger.info("[INFO] Starting confidence threshold sweep on validation set")
    conf_grid = _build_threshold_grid(args)
    decoded_boxes_all, decoded_scores_all, targets_all, loss_metrics = _collect_decoded_validation(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        strides=list(cfg["model"]["strides"]),
        input_size=(int(cfg["input"]["height"]), int(cfg["input"]["width"])),
        decode_twth_clamp_min=float(cfg["loss"].get("decode_twth_clamp_min", -4.0)),
        decode_twth_clamp_max=float(cfg["loss"].get("decode_twth_clamp_max", 4.0)),
        max_batches=args.max_val_batches,
    )

    map_preds_all = build_image_predictions(
        decoded_boxes=decoded_boxes_all,
        decoded_scores=decoded_scores_all,
        targets=targets_all,
        conf_threshold=float(cfg["eval"].get("ap_conf_threshold", 0.001)),
        nms_iou_threshold=float(cfg["eval"]["nms_iou_threshold"]),
        max_detections=int(cfg["eval"].get("ap_max_detections", 200)),
        use_nms=bool(cfg["eval"].get("ap_use_nms", True)),
        single_object_mode=False,
        return_in_original=True,
    )

    rows: list[dict] = []
    for conf in conf_grid:
        op_preds_all = build_image_predictions(
            decoded_boxes=decoded_boxes_all,
            decoded_scores=decoded_scores_all,
            targets=targets_all,
            conf_threshold=float(conf),
            nms_iou_threshold=float(cfg["eval"]["nms_iou_threshold"]),
            max_detections=int(cfg["eval"]["max_detections"]),
            use_nms=bool(cfg["eval"].get("use_nms", True)),
            single_object_mode=bool(cfg["eval"].get("single_object_mode", False)),
            return_in_original=True,
        )
        m = compute_detection_metrics(
            map_predictions=map_preds_all,
            operating_predictions=op_preds_all,
            targets_all=targets_all,
            iou_thresholds=list(cfg["eval"]["iou_thresholds"]),
            matching_iou_threshold=float(cfg["eval"].get("matching_iou_threshold", 0.5)),
            small_area_threshold=float(cfg["eval"].get("small_area_threshold", 0.005)),
            medium_area_threshold=float(cfg["eval"].get("medium_area_threshold", 0.03)),
        )
        row = {
            "conf_threshold": float(conf),
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1": float(m["f1"]),
            "fp_per_image": float(m["fp_per_image"]),
            "mean_iou": float(m["mean_iou"]),
            "recall_small": float(m["recall_small"]),
            "recall_medium": float(m["recall_medium"]),
            "recall_large": float(m["recall_large"]),
            "map50": float(m["map50"]),
            "map5095": float(m["map5095"]),
        }
        rows.append(row)

    objective_key = str(args.sweep_objective).lower()
    best_row = max(rows, key=lambda r: (r[objective_key], r["f1"], r["precision"], -r["fp_per_image"]))

    sweep_out = (
        Path(args.sweep_out).expanduser().resolve()
        if args.sweep_out.strip()
        else (ckpt_path.parent.parent / "threshold_sweep.csv")
    )
    _write_sweep_csv(sweep_out, rows)

    logger.info(
        "[INFO] Sweep summary | "
        f"grid={len(rows)} confs "
        f"objective={objective_key} best_conf={best_row['conf_threshold']:.4f} "
        f"P={best_row['precision']:.4f} R={best_row['recall']:.4f} F1={best_row['f1']:.4f} "
        f"fp/img={best_row['fp_per_image']:.4f} "
        f"small/med/large R={best_row['recall_small']:.4f}/{best_row['recall_medium']:.4f}/{best_row['recall_large']:.4f}"
    )
    logger.info(
        "[INFO] Sweep artifacts | "
        f"csv={sweep_out} "
        f"cached_val_loss={loss_metrics['total_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
