from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Any, Dict

import torch

if __package__ is None or __package__ == "":
    software_root = Path(__file__).resolve().parent.parent
    if str(software_root) not in sys.path:
        sys.path.insert(0, str(software_root))

from ball_detection.src.config import load_config
from ball_detection.src.datasets.spl_ball_dataset import build_dataloaders
from ball_detection.src.datasets.visualization import save_batch_preview
from ball_detection.src.export.export_onnx import export_fp32_onnx
from ball_detection.src.export.quantize_onnx import quantize_int8_qdq
from ball_detection.src.logging_utils import setup_logger
from ball_detection.src.models import build_model
from ball_detection.src.reproducibility import resolve_device, set_seed
from ball_detection.src.training.checkpoint import load_checkpoint, save_checkpoint
from ball_detection.src.training.engine import train_one_epoch, validate_one_epoch
from ball_detection.src.training.losses import BallDetectionLoss
from ball_detection.src.utils.misc import (
    append_metrics_csv,
    count_parameters,
    init_metrics_csv,
    make_run_dir,
    plot_metrics_svg,
    save_yaml,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train STYOLO-style ball detector")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--name", type=str, required=True, help="Run name")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    p.add_argument("--num-workers", type=int, default=None, help="Override dataloader num workers")
    p.add_argument("--max-train-batches", type=int, default=None, help="Debug cap for train batches per epoch")
    p.add_argument("--max-val-batches", type=int, default=None, help="Debug cap for val batches per epoch")
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
    p.add_argument("--prune", type=float, default=None, help="Structured channel-prune ratio in (0,1). Requires --resume.")
    return p.parse_args()


def _build_optimizer(cfg: Dict[str, Any], model: torch.nn.Module):
    opt_name = str(cfg["train"].get("optimizer", "adamw")).lower()
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"].get("weight_decay", 0.0))
    opt_cfg = cfg["train"].get("optimizer_params", {})

    if opt_name == "adamw":
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        eps = float(opt_cfg.get("eps", 1e-8))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas, eps=eps)
    if opt_name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        nesterov = bool(opt_cfg.get("nesterov", True))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            momentum=momentum,
            nesterov=nesterov,
        )

    raise ValueError(f"Unsupported optimizer: {opt_name}")


def _build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    sched_name = str(cfg["train"].get("scheduler", "cosine")).lower()
    epochs = int(cfg["train"]["epochs"])
    sched_cfg = cfg["train"].get("scheduler_params", {})

    if sched_name == "cosine":
        eta_min = float(sched_cfg.get("eta_min", 0.0))
        t_max = int(sched_cfg.get("t_max", max(1, epochs)))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    if sched_name in {"plateau", "reducelronplateau"}:
        sched_mode = str(cfg["train"].get("scheduler_mode", "")).lower().strip()
        if sched_mode not in {"min", "max"}:
            default_monitor = str(cfg["train"].get("early_stopping", {}).get("monitor", "val_map50")).lower()
            sched_mode = "min" if default_monitor in {
                "val_loss",
                "loss",
                "val_total_loss",
                "val_center_error_px",
                "center_error_px",
                "val_center_error_norm_diag",
                "center_error_norm_diag",
                "val_fp_per_image",
                "fp_per_image",
            } else "max"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sched_mode,
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 5)),
            threshold=float(sched_cfg.get("threshold", 1e-4)),
            min_lr=float(sched_cfg.get("min_lr", 0.0)),
            cooldown=int(sched_cfg.get("cooldown", 0)),
        )

    raise ValueError(f"Unsupported scheduler: {sched_name}")


def _is_improved(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == "max":
        return current > (best + min_delta)
    if mode == "min":
        return current < (best - min_delta)
    raise ValueError(f"Unsupported mode: {mode}")


def _resolve_validation_metric(val_metrics: Dict[str, float], monitor: str) -> float:
    monitor_key = str(monitor).lower().strip()
    alias_map = {
        "val_loss": "total_loss",
        "loss": "total_loss",
        "val_total_loss": "total_loss",
        "val_map50": "map50",
        "map50": "map50",
        "val_map5095": "map5095",
        "map5095": "map5095",
        "val_precision": "precision",
        "precision": "precision",
        "val_recall": "recall",
        "recall": "recall",
        "val_f1": "f1",
        "f1": "f1",
        "val_mean_iou": "mean_iou",
        "mean_iou": "mean_iou",
        "val_center_error_px": "center_error_px",
        "center_error_px": "center_error_px",
        "val_center_error_norm_diag": "center_error_norm_diag",
        "center_error_norm_diag": "center_error_norm_diag",
        "val_fp_per_image": "fp_per_image",
        "fp_per_image": "fp_per_image",
    }
    metric_key = alias_map.get(monitor_key)
    if metric_key is None:
        raise ValueError(
            f"Unsupported monitor metric '{monitor}'. "
            "Supported: "
            + ", ".join(sorted(alias_map.keys()))
        )
    if metric_key not in val_metrics:
        raise KeyError(
            f"Monitor metric '{monitor}' resolved to '{metric_key}', "
            f"but this key is not present in val_metrics ({sorted(val_metrics.keys())})."
        )
    return float(val_metrics[metric_key])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    cfg_runtime = copy.deepcopy(cfg)
    if args.reuse_splits is not None:
        cfg_runtime["dataset"]["reuse_splits"] = bool(args.reuse_splits)
    if args.num_workers is not None:
        cfg_runtime["train"]["num_workers"] = int(args.num_workers)

    device = resolve_device(args.device)

    seed = int(cfg_runtime["dataset"]["seed"])
    set_seed(seed)

    output_dir = Path(cfg_runtime["paths"]["output_dir"])
    run_dir = make_run_dir(output_dir=output_dir, run_name=args.name)
    ckpt_dir = run_dir / "checkpoints"
    export_dir = run_dir / "exports"
    gt_preview_dir = run_dir / "groundtruth_preview"
    pred_preview_dir = run_dir / "predictions_preview"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(run_dir / "training.log", debug=args.debug)

    logger.info(f"[INFO] project root: {cfg_runtime['_meta']['project_root']}")
    logger.info(f"[INFO] config path: {cfg_runtime['_meta']['config_path']}")
    logger.info(f"[INFO] dataset_root: {cfg_runtime['paths']['dataset_root']}")
    logger.info(f"[INFO] images_dir: {cfg_runtime['paths']['images_dir']}")
    logger.info(f"[INFO] run directory: {run_dir}")
    logger.info(f"[INFO] reuse_splits: {cfg_runtime['dataset'].get('reuse_splits', True)}")

    save_yaml(cfg_runtime, run_dir / "config_snapshot.yaml")
    init_metrics_csv(run_dir / "metrics.csv")


    # build loaders
    train_loader, val_loader, data_info = build_dataloaders(
        cfg_runtime,
        logger=logger,
        num_workers_override=args.num_workers,
    )

    logger.info(f"[INFO] dataset images: {data_info['num_images']}")
    logger.info(f"[INFO] train images: {data_info['train_images']} | val images: {data_info['val_images']}")
    logger.info(
        f"[INFO] ball annotations used: {data_info['num_ball_boxes']} "
        f"(rows raw: {data_info['num_ball_boxes_rows']})"
    )
    logger.info(
        f"[INFO] images with/without ball: {data_info['images_with_ball']}/{data_info['images_without_ball']}"
    )
    logger.info(
        "[INFO] images with/without annotation rows: "
        f"{data_info.get('images_with_annotation_rows', 0)}/{data_info.get('images_without_annotation_rows', 0)}"
    )
    logger.info(f"[INFO] bbox format: {data_info['bbox_format']}")
    logger.info(f"[INFO] duplicate policy: {data_info['duplicate_policy']}")
    logger.info(f"[INFO] duplicate rows: {data_info.get('duplicate_rows', 0)}")
    logger.info(f"[INFO] duplicate row overwrites: {data_info['duplicate_row_overwrites']}")
    logger.info(f"[INFO] annotation rows with missing image files: {data_info.get('missing_image_rows', 0)}")
    logger.info(f"[INFO] dataset sources: {data_info.get('source_names', [])}")
    logger.info(f"[INFO] all images per source: {data_info.get('all_source_images', {})}")
    logger.info(f"[INFO] train images per source: {data_info.get('train_source_images', {})}")
    logger.info(f"[INFO] val images per source: {data_info.get('val_source_images', {})}")
    source_stats = data_info.get("source_stats", {})
    if isinstance(source_stats, dict):
        for src_name in sorted(source_stats.keys()):
            src = source_stats[src_name]
            logger.info(
                "[INFO] source stats | "
                f"{src_name}: images={src.get('num_images', 0)} rows={src.get('num_rows', 0)} "
                f"imgs_with/without_rows={src.get('images_with_annotation_rows', 0)}/{src.get('images_without_annotation_rows', 0)} "
                f"boxes={src.get('num_ball_boxes', 0)} dup_rows={src.get('duplicate_rows', 0)} "
                f"bbox_format={src.get('bbox_format', 'unknown')} dup_policy={src.get('duplicate_policy', 'unknown')}"
            )
    logger.info(f"[INFO] train batch mixing: {data_info.get('train_batch_mixing', {})}")
    logger.info(
        "[INFO] bbox size stats px | "
        f"w(min/mean/max)=({data_info['bbox_width_min']:.2f}/{data_info['bbox_width_mean']:.2f}/{data_info['bbox_width_max']:.2f}) "
        f"h(min/mean/max)=({data_info['bbox_height_min']:.2f}/{data_info['bbox_height_mean']:.2f}/{data_info['bbox_height_max']:.2f})"
    )
    logger.info(
        f"[INFO] input: {cfg_runtime['input']['width']}x{cfg_runtime['input']['height']} | "
        f"color_mode={cfg_runtime['input']['color_mode']} | channels={cfg_runtime['input']['channels']} | "
        f"resize_policy={cfg_runtime['input'].get('resize_policy', 'letterbox')}"
    )

    batch_images, batch_targets = next(iter(train_loader))
    save_batch_preview(batch_images, batch_targets, gt_preview_dir, prefix="train_gt", max_images=8)
    batch_images_v, batch_targets_v = next(iter(val_loader))
    save_batch_preview(batch_images_v, batch_targets_v, gt_preview_dir, prefix="val_gt", max_images=8)

    # build model
    model = build_model(cfg_runtime).to(device)
    n_params = count_parameters(model)

    logger.info(f"[INFO] model: {cfg_runtime['model']['name']}")
    logger.info(f"[INFO] params: {n_params:,}")
    logger.info(f"[INFO] device: {device}")
    logger.info(f"[INFO] batch_size: {cfg_runtime['train']['batch_size']}")
    logger.info(f"[INFO] num_workers: {cfg_runtime['train']['num_workers']}")

    with (run_dir / "model_summary.txt").open("w", encoding="utf-8") as f:
        f.write(str(model))
        f.write("\n\n")
        f.write(f"parameters: {n_params}\n")

    criterion = BallDetectionLoss(
        strides=cfg_runtime["model"]["strides"],
        obj_weight=float(cfg_runtime["loss"]["obj_weight"]),
        box_weight=float(cfg_runtime["loss"]["box_weight"]),
        obj_loss_mode=str(cfg_runtime["loss"].get("obj_loss_mode", "mean")),
        obj_pos_weight=float(cfg_runtime["loss"].get("obj_pos_weight", 1.0)),
        obj_neg_weight=float(cfg_runtime["loss"].get("obj_neg_weight", 1.0)),
        obj_bce_pos_weight=float(cfg_runtime["loss"].get("obj_bce_pos_weight", 1.0)),
        focal_loss=bool(cfg_runtime["loss"].get("focal_loss", False)),
        focal_alpha=float(cfg_runtime["loss"].get("focal_alpha", 0.25)),
        focal_gamma=float(cfg_runtime["loss"].get("focal_gamma", 2.0)),
        assign_scale_target=float(cfg_runtime["loss"].get("assign_scale_target", 3.0)),
        assign_conflict_policy=str(cfg_runtime["loss"].get("assign_conflict_policy", "largest_area")),
        assign_center_radius=int(cfg_runtime["loss"].get("assign_center_radius", 0)),
        decode_twth_clamp_min=float(cfg_runtime["loss"].get("decode_twth_clamp_min", -4.0)),
        decode_twth_clamp_max=float(cfg_runtime["loss"].get("decode_twth_clamp_max", 4.0)),
    )

    optimizer = _build_optimizer(cfg_runtime, model)
    scheduler = _build_scheduler(cfg_runtime, optimizer)

    amp_enabled = bool(cfg_runtime["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    best_metric = -math.inf

    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", -math.inf))
        logger.info(f"[INFO] Resumed from {resume_path} at epoch {start_epoch}")

    # ========== prune (optional) ==========
    if args.prune is not None:
        if not args.resume:
            raise ValueError("--prune requires --resume (pruning untrained weights is just random)")
        from ball_detection.src.prune import prune_styolo

        example = torch.randn(
            1,
            int(cfg_runtime["input"]["channels"]),
            int(cfg_runtime["input"]["height"]),
            int(cfg_runtime["input"]["width"]),
            device=device,
        )
        logger.info(f"[INFO] Structured channel pruning at ratio {args.prune:.2%}")
        model, n_before, n_after = prune_styolo(model, args.prune, example)
        logger.info(f"[INFO] params: {n_before:,} -> {n_after:,}  ({100*(n_before-n_after)/n_before:.1f}% reduction)")

        # Re-init optimizer/scheduler/scaler since the parameter set changed.
        # Restart the cosine schedule from epoch 1 so it re-anneals over the full budget.
        optimizer = _build_optimizer(cfg_runtime, model)
        scheduler = _build_scheduler(cfg_runtime, optimizer)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        start_epoch = 1
        prune_monitor = str(cfg_runtime["train"].get("early_stopping", {}).get("monitor", "val_map50")).lower().strip()
        best_metric = math.inf if prune_monitor == "val_loss" else -math.inf

        # Re-save snapshots so the run dir reflects the pruned model.
        save_yaml(cfg_runtime, run_dir / "config_snapshot.yaml")
        with (run_dir / "model_summary.txt").open("w", encoding="utf-8") as f:
            f.write(str(model))
            f.write("\n\n")
            f.write(f"parameters: {n_after}\n")
        logger.info(f"[INFO] Updated config snapshot + model summary")

    epochs = int(cfg_runtime["train"]["epochs"])
    grad_clip_norm = float(cfg_runtime["train"].get("grad_clip_norm", 0.0))

    es_cfg = cfg_runtime["train"].get("early_stopping", {})
    early_enabled = bool(es_cfg.get("enabled", True))
    monitor = str(es_cfg.get("monitor", "val_map50")).lower().strip()
    mode = str(es_cfg.get("mode", "max")).lower()
    patience = int(es_cfg.get("patience", 20))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    bad_epochs = 0

    if monitor == "val_loss" and best_metric == -math.inf:
        best_metric = math.inf
    elif monitor in {
        "loss",
        "val_total_loss",
        "val_center_error_px",
        "center_error_px",
        "val_center_error_norm_diag",
        "center_error_norm_diag",
        "val_fp_per_image",
        "fp_per_image",
    } and best_metric == -math.inf:
        best_metric = math.inf

    logger.info(
        f"[INFO] early_stopping enabled={early_enabled} monitor={monitor} mode={mode} "
        f"patience={patience} min_delta={min_delta}"
    )
    scheduler_monitor = str(cfg_runtime["train"].get("scheduler_monitor", monitor)).lower()

    best_ckpt_path = ckpt_dir / "best.pt"
    last_ckpt_path = ckpt_dir / "last.pt"

    max_train_batches = args.max_train_batches
    max_val_batches = args.max_val_batches

    # traning loop
    for epoch in range(start_epoch, epochs + 1):
        lr_now = float(optimizer.param_groups[0]["lr"])
        logger.info(f"[INFO] epoch {epoch}/{epochs} start | lr={lr_now:.6g}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            grad_clip_norm=grad_clip_norm,
            max_batches=max_train_batches,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            strides=cfg_runtime["model"]["strides"],
            input_size=(int(cfg_runtime["input"]["height"]), int(cfg_runtime["input"]["width"])),
            conf_threshold=float(cfg_runtime["eval"]["conf_threshold"]),
            nms_iou_threshold=float(cfg_runtime["eval"]["nms_iou_threshold"]),
            max_detections=int(cfg_runtime["eval"]["max_detections"]),
            use_nms=bool(cfg_runtime["eval"].get("use_nms", True)),
            single_object_mode=bool(cfg_runtime["eval"].get("single_object_mode", False)),
            ap_conf_threshold=float(cfg_runtime["eval"].get("ap_conf_threshold", 0.001)),
            ap_max_detections=int(cfg_runtime["eval"].get("ap_max_detections", max(200, int(cfg_runtime["eval"]["max_detections"])))),
            ap_use_nms=bool(cfg_runtime["eval"].get("ap_use_nms", True)),
            iou_thresholds=list(cfg_runtime["eval"]["iou_thresholds"]),
            matching_iou_threshold=float(cfg_runtime["eval"].get("matching_iou_threshold", 0.5)),
            small_area_threshold=float(cfg_runtime["eval"].get("small_area_threshold", 0.005)),
            medium_area_threshold=float(cfg_runtime["eval"].get("medium_area_threshold", 0.03)),
            decode_twth_clamp_min=float(cfg_runtime["loss"].get("decode_twth_clamp_min", -4.0)),
            decode_twth_clamp_max=float(cfg_runtime["loss"].get("decode_twth_clamp_max", 4.0)),
            max_batches=max_val_batches,
            preview_dir=pred_preview_dir / f"epoch_{epoch:03d}",
        )

        scheduler_name = str(cfg_runtime["train"].get("scheduler", "cosine")).lower()
        if scheduler_name in {"plateau", "reducelronplateau"}:
            scheduler_value = _resolve_validation_metric(val_metrics, scheduler_monitor)
            scheduler.step(scheduler_value)
        else:
            scheduler.step()

        logger.info(
            "[INFO] epoch summary | "
            f"train loss={train_metrics['total_loss']:.4f} obj={train_metrics['obj_loss']:.4f} box={train_metrics['box_loss']:.4f} pos={train_metrics['num_pos']:.2f} | "
            f"val loss={val_metrics['total_loss']:.4f} map50={val_metrics['map50']:.4f} map5095={val_metrics['map5095']:.4f} "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} F1={val_metrics['f1']:.4f}"
        )
        logger.info(
            "[INFO] val extras | "
            f"mean_iou={val_metrics['mean_iou']:.4f} "
            f"center_err_px(mean/med/p95)="
            f"{val_metrics['center_error_px']:.2f}/{val_metrics['center_error_px_median']:.2f}/{val_metrics['center_error_px_p95']:.2f} "
            f"center_err_norm_diag(mean/med/p95)="
            f"{val_metrics['center_error_norm_diag']:.4f}/{val_metrics['center_error_norm_diag_median']:.4f}/{val_metrics['center_error_norm_diag_p95']:.4f} "
            f"fp/img={val_metrics['fp_per_image']:.4f} "
            f"small/med/large recall={val_metrics['recall_small']:.4f}/{val_metrics['recall_medium']:.4f}/{val_metrics['recall_large']:.4f} "
            f"counts={val_metrics['count_small']}/{val_metrics['count_medium']}/{val_metrics['count_large']}"
        )

        append_metrics_csv(
            run_dir / "metrics.csv",
            {
                "epoch": epoch,
                "lr": lr_now,
                "train_total_loss": train_metrics["total_loss"],
                "train_obj_loss": train_metrics["obj_loss"],
                "train_box_loss": train_metrics["box_loss"],
                "train_num_pos": train_metrics["num_pos"],
                "val_total_loss": val_metrics["total_loss"],
                "val_obj_loss": val_metrics["obj_loss"],
                "val_box_loss": val_metrics["box_loss"],
                "val_num_pos": val_metrics["num_pos"],
                "val_map50": val_metrics["map50"],
                "val_map5095": val_metrics["map5095"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_mean_iou": val_metrics["mean_iou"],
                "val_center_error_px": val_metrics["center_error_px"],
                "val_center_error_px_median": val_metrics["center_error_px_median"],
                "val_center_error_px_p95": val_metrics["center_error_px_p95"],
                "val_center_error_norm_diag": val_metrics["center_error_norm_diag"],
                "val_center_error_norm_diag_median": val_metrics["center_error_norm_diag_median"],
                "val_center_error_norm_diag_p95": val_metrics["center_error_norm_diag_p95"],
                "val_fp_per_image": val_metrics["fp_per_image"],
                "val_recall_small": val_metrics["recall_small"],
                "val_recall_medium": val_metrics["recall_medium"],
                "val_recall_large": val_metrics["recall_large"],
                "val_count_small": val_metrics["count_small"],
                "val_count_medium": val_metrics["count_medium"],
                "val_count_large": val_metrics["count_large"],
            },
        )
        plot_metrics_svg(run_dir / "metrics.csv", run_dir / "metrics.svg")

        save_checkpoint(
            last_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            cfg=cfg_runtime,
            best_metric=best_metric,
        )

        current = _resolve_validation_metric(val_metrics, monitor)

        if _is_improved(current=current, best=best_metric, mode=mode, min_delta=min_delta):
            best_metric = current
            bad_epochs = 0
            save_checkpoint(
                best_ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                cfg=cfg_runtime,
                best_metric=best_metric,
            )
            logger.info(f"[INFO] new best checkpoint at epoch {epoch}: {best_ckpt_path}")
        else:
            bad_epochs += 1
            logger.info(f"[INFO] early stopping status: bad_epochs={bad_epochs}/{patience}")

        if early_enabled and bad_epochs >= patience:
            logger.info(f"[INFO] early stopping triggered at epoch {epoch}")
            break

    # Export best checkpoint artifacts.
    if best_ckpt_path.exists():
        logger.info(f"[INFO] Loading best checkpoint for export: {best_ckpt_path}")
        load_checkpoint(best_ckpt_path, model=model, map_location=device)

        export_cfg = cfg_runtime["export"]
        do_fp32 = bool(export_cfg.get("do_fp32_onnx", True))
        do_int8 = bool(export_cfg.get("do_int8_ptq_qdq", True))

        fp32_path = export_dir / "fp32.onnx"
        int8_path = export_dir / "int8_ptq_qdq.onnx"

        if do_fp32:
            export_report = export_fp32_onnx(
                model=model,
                out_path=fp32_path,
                input_shape=(
                    1,
                    int(cfg_runtime["input"]["channels"]),
                    int(cfg_runtime["input"]["height"]),
                    int(cfg_runtime["input"]["width"]),
                ),
                opset=int(export_cfg.get("opset", 13)),
                dynamic_batch=bool(export_cfg.get("dynamic_batch", False)),
                logger=logger,
                sample_batch=batch_images,
            )
        else:
            export_report = {"exported": False}

        if do_int8 and bool(export_report.get("exported", False)) and fp32_path.exists():
            int8_cfg = export_cfg.get("int8", {})
            quantize_int8_qdq(
                fp32_onnx_path=fp32_path,
                int8_onnx_path=int8_path,
                calibration_loader=val_loader,
                calibration_batches=int(export_cfg.get("calibration_batches", 20)),
                batch_size=int(cfg_runtime["train"]["batch_size"]),
                per_channel=bool(int8_cfg.get("per_channel", True)),
                activation_type=str(int8_cfg.get("activation_type", "uint8")),
                weight_type=str(int8_cfg.get("weight_type", "int8")),
                calibrate_method=str(int8_cfg.get("calibrate_method", "minmax")),
                report_path=export_dir / "calibration_report.json",
                cfg=cfg_runtime,
                logger=logger,
            )
        elif do_int8:
            logger.warning("[WARN] Skipping INT8 export because FP32 ONNX export was not successful.")

    logger.info("[INFO] Training pipeline finished.")


if __name__ == "__main__":
    main()
