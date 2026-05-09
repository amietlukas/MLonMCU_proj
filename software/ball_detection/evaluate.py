from __future__ import annotations

import argparse
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
from ball_detection.src.training.losses import BallDetectionLoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ball detector checkpoint")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

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
        f"mean_iou={metrics['mean_iou']:.4f} center_err_px={metrics['center_error_px']:.2f} fp/img={metrics['fp_per_image']:.4f}"
    )


if __name__ == "__main__":
    main()
