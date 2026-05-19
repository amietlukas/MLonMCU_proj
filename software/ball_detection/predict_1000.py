"""Forward-pass 1000 randomly sampled images from train+val combined, log every metric
we can compute, and write annotated GT+prediction overlays for visual inspection.

Output layout:
  <run_dir>/../pred_images/<run_name>_pred_images_<N>/
    images/<idx:04d>_<image_stem>.png        per-image GT (lime/orange) + pred (cyan/red) overlay
    per_image_metrics.csv                    one row per sampled image
    predictions.csv                          one row per (image, prediction)
    gt_boxes.csv                             one row per (image, GT) for sanity / size analysis
    size_bucket_metrics.csv                  per-size-bucket recall/IoU/center-error
    confidence_buckets.csv                   tp/fp counts per conf bucket
    source_breakdown.csv                     per-source aggregate metrics (spl vs our)
    metrics_summary.json                     all aggregate metrics + dataset / pred distributions
    run.log                                  debug log
    sampled_ids.txt                          newline-list of image_ids that were sampled
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    software_root = Path(__file__).resolve().parent.parent
    if str(software_root) not in sys.path:
        sys.path.insert(0, str(software_root))

from ball_detection.src.config import load_config
from ball_detection.src.datasets.spl_ball_dataset import (
    SPLBallDetectionDataset,
    _make_transforms,
    _resolve_dataset_sources,
    detection_collate_fn,
    load_spl_ball_records,
)
from ball_detection.src.logging_utils import setup_logger
from ball_detection.src.models import build_model
from ball_detection.src.reproducibility import resolve_device, set_seed, worker_init_fn
from ball_detection.src.training.checkpoint import load_checkpoint
from ball_detection.src.training.decode import decode_outputs
from ball_detection.src.training.metrics import (
    _match_predictions,
    build_image_predictions,
    compute_detection_metrics,
)
from ball_detection.src.utils.boxes import box_iou


PRED_TP_COLOR = "cyan"
PRED_FP_COLOR = "red"
GT_TP_COLOR = "lime"
GT_FN_COLOR = "orange"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forward-pass N images, log every metric, write GT+pred overlays")
    p.add_argument(
        "--run-dir",
        type=str,
        default="/mnt/core/MLonMCU_proj/software/ball_detection/runs/20260519-025736-simota_conf045",
        help="Path to a training run directory (must contain config_snapshot.yaml and checkpoints/)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="Checkpoint filename inside <run_dir>/checkpoints (default: best.pt)",
    )
    p.add_argument(
        "--n-images",
        type=int,
        default=1000,
        help="Number of random images to sample from train+val combined (default: 1000)",
    )
    p.add_argument(
        "--sample-from",
        type=str,
        default="train_val",
        choices=["train_val", "val", "train"],
        help="Which split(s) to sample from (default: train_val = all records)",
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Override inference batch size. 0 = use cfg.train.batch_size",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42, help="RNG seed for the random sample selection")
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Override output directory. Default: <run_dir>/../pred_images/<run_name>_pred_images_<N>",
    )
    p.add_argument(
        "--no-images",
        action="store_true",
        help="Skip writing per-image overlay PNGs (still writes all CSVs/JSON). Speeds the script up.",
    )
    p.add_argument(
        "--all-images",
        action="store_true",
        help="Process every record from the --sample-from pool (overrides --n-images). Output dir suffix becomes `_all`.",
    )
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def _font():
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        return ImageFont.load_default()


def _draw_box_with_label(
    draw: ImageDraw.ImageDraw,
    box_xyxy: Sequence[float],
    color: str,
    label: str | None,
    font,
    width: int = 2,
) -> None:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else (len(label) * 7.0)
        th = 14.0
        ty = max(0.0, y1 - th - 2.0)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=color)
        draw.text((x1 + 2, ty + 1), label, fill="black", font=font)


def _size_bin_from_area_ratio(area_ratio: float, small_t: float, medium_t: float) -> str:
    if area_ratio < small_t:
        return "small"
    if area_ratio < medium_t:
        return "medium"
    return "large"


def _safe_div(a: float, b: float) -> float:
    if b <= 0:
        return 0.0
    return a / b


def _percentile_sorted(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    qq = min(max(float(q), 0.0), 1.0)
    pos = qq * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    t = pos - lo
    return (1.0 - t) * sorted_vals[lo] + t * sorted_vals[hi]


def _stats_summary(values: list[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    sorted_vals = sorted(float(v) for v in values)
    return {
        "count": len(sorted_vals),
        "min": float(sorted_vals[0]),
        "mean": float(sum(sorted_vals) / len(sorted_vals)),
        "median": float(_percentile_sorted(sorted_vals, 0.5)),
        "p95": float(_percentile_sorted(sorted_vals, 0.95)),
        "max": float(sorted_vals[-1]),
    }


def _load_split_ids(run_dir: Path, cfg: Dict[str, Any]) -> tuple[set[str], set[str]]:
    """Try to read train.txt/val.txt that this run actually used."""
    splits_dir = Path(cfg["paths"]["splits_dir"])
    train_file = splits_dir / "train.txt"
    val_file = splits_dir / "val.txt"
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    if train_file.is_file():
        train_ids = {ln.strip() for ln in train_file.read_text().splitlines() if ln.strip()}
    if val_file.is_file():
        val_ids = {ln.strip() for ln in val_file.read_text().splitlines() if ln.strip()}
    return train_ids, val_ids


def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    cfg_path = run_dir / "config_snapshot.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config_snapshot.yaml missing in {run_dir}")
    cfg = load_config(cfg_path)

    ckpt_path = run_dir / "checkpoints" / args.checkpoint
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    run_name = run_dir.name
    n_request = int(args.n_images)
    all_images_mode = bool(args.all_images)

    if args.out_dir.strip():
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        suffix = "all" if all_images_mode else str(n_request)
        out_dir = run_dir.parent.parent / "pred_images" / f"{run_name}_pred_images_{suffix}"
    images_out_dir = out_dir / "images"
    fp_images_out_dir = out_dir / "fp_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_images:
        images_out_dir.mkdir(parents=True, exist_ok=True)
        fp_images_out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir / "run.log", debug=args.debug)

    logger.info(f"[INFO] run_dir       : {run_dir}")
    logger.info(f"[INFO] config        : {cfg_path}")
    logger.info(f"[INFO] checkpoint    : {ckpt_path}")
    logger.info(f"[INFO] out_dir       : {out_dir}")
    logger.info(f"[INFO] n_request     : {n_request}")
    logger.info(f"[INFO] all_images    : {all_images_mode}")
    logger.info(f"[INFO] sample_from   : {args.sample_from}")
    logger.info(f"[INFO] seed          : {args.seed}")
    logger.info(f"[INFO] no_images     : {args.no_images}")

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    logger.info(f"[INFO] device        : {device}")

    # 1) Resolve dataset sources and load all records (no augmentation -- val transform applied later).
    sources = _resolve_dataset_sources(cfg)
    validate_images = bool(cfg["dataset"].get("validate_images", False))
    skip_unreadable = bool(cfg["dataset"].get("skip_unreadable_images", False))
    all_records, ds_stats = load_spl_ball_records(
        sources=sources,
        logger=logger,
        validate_images=validate_images,
        skip_unreadable_images=skip_unreadable,
    )
    logger.info(f"[INFO] total records loaded: {len(all_records)} | sources: {[s.name for s in sources]}")
    logger.info(f"[INFO] per-source counts   : {ds_stats.get('source_names', [])} -> "
                f"{[ds_stats.get('source_stats', {}).get(n, {}).get('num_images', 0) for n in ds_stats.get('source_names', [])]}")

    train_ids, val_ids = _load_split_ids(run_dir, cfg)
    logger.info(f"[INFO] split file ids: train={len(train_ids)} val={len(val_ids)}")

    # 2) Filter records by --sample-from.
    if args.sample_from == "val":
        if not val_ids:
            raise RuntimeError("val.txt is empty or missing; cannot sample from 'val'")
        pool = [r for r in all_records if r.image_id in val_ids]
    elif args.sample_from == "train":
        if not train_ids:
            raise RuntimeError("train.txt is empty or missing; cannot sample from 'train'")
        pool = [r for r in all_records if r.image_id in train_ids]
    else:
        pool = list(all_records)
    logger.info(f"[INFO] pool size for sampling: {len(pool)}")

    if not pool:
        raise RuntimeError("Empty record pool — nothing to sample.")

    # 3) Pick records. --all-images bypasses random sampling and keeps deterministic order
    # (sorted by image_id, which matches load_spl_ball_records' alphabetic walk). Otherwise sample.
    rng = random.Random(int(args.seed))
    if all_images_mode:
        sampled = sorted(pool, key=lambda r: r.image_id)
        logger.info(f"[INFO] --all-images set: processing entire pool ({len(sampled)}) in id order")
    elif len(pool) <= n_request:
        sampled = list(pool)
        rng.shuffle(sampled)
        logger.info(f"[INFO] pool ({len(pool)}) <= n_request ({n_request}); using whole pool, shuffled")
    else:
        sampled = rng.sample(pool, k=n_request)
        logger.info(f"[INFO] sampled {len(sampled)} records of {len(pool)}")

    with (out_dir / "sampled_ids.txt").open("w", encoding="utf-8") as f:
        for r in sampled:
            f.write(f"{r.image_id}\n")

    # Track source breakdown of the sampled pool.
    sample_source_counts = Counter(r.source_name for r in sampled)
    logger.info(f"[INFO] sampled source counts: {dict(sample_source_counts)}")

    # 4) Build val-style transform and dataset.
    val_tf = _make_transforms(cfg, train=False)
    pred_ds = SPLBallDetectionDataset(
        records=sampled,
        transform=val_tf,
        class_name=str(cfg["dataset"].get("class_name", "Ball")),
        skip_read_errors=False,
        read_retry_count=0,
    )
    bs = int(args.batch_size) if int(args.batch_size) > 0 else int(cfg["train"]["batch_size"])
    loader = DataLoader(
        pred_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=detection_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    logger.info(f"[INFO] dataloader: batch_size={bs} num_workers={args.num_workers} num_batches={len(loader)}")

    # 5) Build model + load checkpoint.
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(ckpt_path, model=model, map_location=device)
    epoch_in_ckpt = int(ckpt.get("epoch", -1))
    best_metric_in_ckpt = float(ckpt.get("best_metric", float("nan")))
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[INFO] model params  : {n_params:,}")
    logger.info(f"[INFO] ckpt epoch    : {epoch_in_ckpt}")
    logger.info(f"[INFO] ckpt best_metric: {best_metric_in_ckpt}")
    model.eval()

    # 6) Forward pass + build predictions.
    strides = list(cfg["model"]["strides"])
    in_h = int(cfg["input"]["height"])
    in_w = int(cfg["input"]["width"])
    twth_min = float(cfg["loss"].get("decode_twth_clamp_min", -4.0))
    twth_max = float(cfg["loss"].get("decode_twth_clamp_max", 4.0))
    conf_threshold = float(cfg["eval"]["conf_threshold"])
    nms_iou = float(cfg["eval"]["nms_iou_threshold"])
    max_dets = int(cfg["eval"]["max_detections"])
    use_nms = bool(cfg["eval"].get("use_nms", True))
    single_obj = bool(cfg["eval"].get("single_object_mode", False))
    ap_conf = float(cfg["eval"].get("ap_conf_threshold", 0.001))
    ap_max_dets = int(cfg["eval"].get("ap_max_detections", max(200, max_dets)))
    ap_use_nms = bool(cfg["eval"].get("ap_use_nms", True))
    iou_thresholds = list(cfg["eval"]["iou_thresholds"])
    matching_iou = float(cfg["eval"].get("matching_iou_threshold", 0.5))
    small_t = float(cfg["eval"].get("small_area_threshold", 0.005))
    medium_t = float(cfg["eval"].get("medium_area_threshold", 0.03))

    logger.info(
        f"[INFO] eval params   : conf={conf_threshold} nms_iou={nms_iou} max_dets={max_dets} "
        f"use_nms={use_nms} single_obj={single_obj} matching_iou={matching_iou} "
        f"small_t={small_t} medium_t={medium_t}"
    )

    op_preds_all = []
    map_preds_all = []
    targets_all = []

    forward_times_ms: list[float] = []
    decode_times_ms: list[float] = []
    total_t0 = time.time()

    with torch.no_grad():
        for b_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            decoded_boxes, decoded_scores = decode_outputs(
                outputs,
                strides=strides,
                input_size=(in_h, in_w),
                decode_twth_clamp_min=twth_min,
                decode_twth_clamp_max=twth_max,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.time()

            forward_times_ms.append(1000.0 * (t1 - t0) / max(1, images.shape[0]))
            decode_times_ms.append(1000.0 * (t2 - t1) / max(1, images.shape[0]))

            map_preds = build_image_predictions(
                decoded_boxes=decoded_boxes,
                decoded_scores=decoded_scores,
                targets=targets,
                conf_threshold=ap_conf,
                nms_iou_threshold=nms_iou,
                max_detections=ap_max_dets,
                use_nms=ap_use_nms,
                single_object_mode=False,
                return_in_original=True,
            )
            op_preds = build_image_predictions(
                decoded_boxes=decoded_boxes,
                decoded_scores=decoded_scores,
                targets=targets,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou,
                max_detections=max_dets,
                use_nms=use_nms,
                single_object_mode=single_obj,
                return_in_original=True,
            )

            map_preds_all.extend(map_preds)
            op_preds_all.extend(op_preds)
            targets_all.extend(targets)

            if (b_idx + 1) % 5 == 0 or b_idx == 0:
                logger.info(
                    f"[INFO] batch {b_idx+1}/{len(loader)} done | "
                    f"fwd ms/img={forward_times_ms[-1]:.2f} dec ms/img={decode_times_ms[-1]:.2f}"
                )

    total_t1 = time.time()
    logger.info(f"[INFO] forward+decode over {len(targets_all)} images took {total_t1 - total_t0:.2f}s")

    if len(targets_all) != len(sampled):
        logger.warning(
            f"[WARN] targets_all len ({len(targets_all)}) != sampled len ({len(sampled)}) -- DataLoader may have dropped samples"
        )

    # 7) Aggregate metric block (mirrors validate_one_epoch's metrics).
    agg = compute_detection_metrics(
        map_predictions=map_preds_all,
        operating_predictions=op_preds_all,
        targets_all=targets_all,
        iou_thresholds=iou_thresholds,
        matching_iou_threshold=matching_iou,
        small_area_threshold=small_t,
        medium_area_threshold=medium_t,
    )
    logger.info(
        f"[INFO] aggregate: map50={agg['map50']:.4f} map5095={agg['map5095']:.4f} "
        f"P={agg['precision']:.4f} R={agg['recall']:.4f} F1={agg['f1']:.4f} "
        f"meanIoU={agg['mean_iou']:.4f} fp/img={agg['fp_per_image']:.4f}"
    )
    logger.info(
        f"[INFO] aggregate (size): small/med/large recall = "
        f"{agg['recall_small']:.4f}/{agg['recall_medium']:.4f}/{agg['recall_large']:.4f} "
        f"(counts={agg['count_small']}/{agg['count_medium']}/{agg['count_large']})"
    )

    # 8) Per-image / per-prediction / per-GT accounting.
    op_by_id = {p.image_id: p for p in op_preds_all}
    map_by_id = {p.image_id: p for p in map_preds_all}
    target_by_id = {str(t["image_id"]): t for t in targets_all}

    per_image_rows: list[dict] = []
    pred_rows: list[dict] = []
    gt_rows: list[dict] = []

    # Confidence bucket grid: 0.0..1.0 in steps of 0.05.
    conf_bucket_edges = [round(0.05 * i, 2) for i in range(21)]
    conf_buckets_tp = Counter()
    conf_buckets_fp = Counter()

    # Per-size-bucket aggregates.
    size_bucket_stats: dict[str, dict[str, list[float]]] = {
        "small": {"iou": [], "center_err_px": [], "center_err_norm_diag": [], "matched": [], "total": []},
        "medium": {"iou": [], "center_err_px": [], "center_err_norm_diag": [], "matched": [], "total": []},
        "large": {"iou": [], "center_err_px": [], "center_err_norm_diag": [], "matched": [], "total": []},
    }
    source_stats: dict[str, dict[str, int | float | list]] = defaultdict(
        lambda: {"images": 0, "gt": 0, "tp": 0, "fp": 0, "fn": 0, "ious": [], "center_err_px": []}
    )

    # Global distributions
    all_pred_confs: list[float] = []
    all_pred_widths_px: list[float] = []
    all_pred_heights_px: list[float] = []
    all_gt_widths_px: list[float] = []
    all_gt_heights_px: list[float] = []
    all_gt_aspect_ratios: list[float] = []
    all_gt_area_ratios: list[float] = []
    all_matched_iou: list[float] = []
    all_center_err_px: list[float] = []
    all_center_err_norm_diag: list[float] = []

    n_perfect = 0  # images where tp == num_gt and fp == 0
    n_missed_entirely = 0  # num_gt > 0 and tp == 0
    n_false_alarm_only = 0  # num_gt == 0 and fp > 0
    n_empty_correct = 0  # num_gt == 0 and fp == 0 (true negative image)
    n_no_gt = 0  # ground-truth has no balls

    rec_by_id = {r.image_id: r for r in sampled}

    for idx, target in enumerate(targets_all):
        image_id = str(target["image_id"])
        orig_h, orig_w = (int(target["orig_size"][0]), int(target["orig_size"][1]))
        image_diag = max((orig_h * orig_h + orig_w * orig_w) ** 0.5, 1e-9)
        denom_area = max(orig_h * orig_w, 1)

        gt_boxes = target["boxes_orig"].detach().cpu().float()
        n_gt = int(gt_boxes.shape[0])

        op_pred = op_by_id.get(image_id)
        if op_pred is None:
            pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
            pred_scores = torch.zeros((0,), dtype=torch.float32)
        else:
            pred_boxes = op_pred.boxes.float()
            pred_scores = op_pred.scores.float()
        n_pred = int(pred_boxes.shape[0])

        map_pred = map_by_id.get(image_id)
        n_pred_raw = int(map_pred.boxes.shape[0]) if map_pred is not None else 0
        top_raw_conf = float(map_pred.scores.max().item()) if (map_pred is not None and map_pred.scores.numel() > 0) else 0.0
        top_op_conf = float(pred_scores.max().item()) if pred_scores.numel() > 0 else 0.0

        matches = _match_predictions(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            iou_threshold=matching_iou,
        )
        tp = int(matches["tp"])
        fp = int(matches["fp"])
        fn = int(matches["fn"])
        matched_pred_set = set(matches["matched_pred_indices"].tolist())
        matched_gt_set = set(matches["matched_gt_indices"].tolist())
        ious_list = matches["matched_ious"].tolist()

        # Best IOU between any pred and any GT (even if not matched 1-to-1).
        if n_pred > 0 and n_gt > 0:
            iou_grid = box_iou(pred_boxes, gt_boxes)
            best_iou_any = float(iou_grid.max().item())
        else:
            best_iou_any = 0.0

        # Per-image center error: mean of matched pred-gt center distances.
        matched_center_err_px = []
        matched_center_err_norm_diag = []
        if matches["matched_pred_indices"].numel() > 0:
            p_idx = matches["matched_pred_indices"]
            g_idx = matches["matched_gt_indices"]
            pc = 0.5 * (pred_boxes[p_idx, 0:2] + pred_boxes[p_idx, 2:4])
            gc = 0.5 * (gt_boxes[g_idx, 0:2] + gt_boxes[g_idx, 2:4])
            dist = torch.linalg.norm(pc - gc, dim=1).tolist()
            matched_center_err_px.extend(dist)
            matched_center_err_norm_diag.extend([d / image_diag for d in dist])

        # GT row writing + size bucketing.
        gt_size_bins = []
        for gi in range(n_gt):
            box = gt_boxes[gi]
            bw = float((box[2] - box[0]).clamp(min=0).item())
            bh = float((box[3] - box[1]).clamp(min=0).item())
            area = bw * bh
            area_ratio = area / float(denom_area)
            ar = bw / max(bh, 1e-6)
            bin_name = _size_bin_from_area_ratio(area_ratio, small_t, medium_t)
            gt_size_bins.append(bin_name)
            all_gt_widths_px.append(bw)
            all_gt_heights_px.append(bh)
            all_gt_aspect_ratios.append(ar)
            all_gt_area_ratios.append(area_ratio)

            matched_iou = 0.0
            matched_pred_score = ""
            matched_pred_idx = ""
            center_err_px = ""
            center_err_norm_diag = ""
            if gi in matched_gt_set:
                local_idx = matches["matched_gt_indices"].tolist().index(gi)
                matched_iou = float(matches["matched_ious"][local_idx].item())
                pi = int(matches["matched_pred_indices"][local_idx].item())
                matched_pred_score = float(pred_scores[pi].item())
                matched_pred_idx = pi
                pcx = 0.5 * float((pred_boxes[pi, 0] + pred_boxes[pi, 2]).item())
                pcy = 0.5 * float((pred_boxes[pi, 1] + pred_boxes[pi, 3]).item())
                gcx = 0.5 * float((box[0] + box[2]).item())
                gcy = 0.5 * float((box[1] + box[3]).item())
                de = ((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5
                center_err_px = de
                center_err_norm_diag = de / image_diag

            gt_rows.append({
                "image_id": image_id,
                "source": rec_by_id[image_id].source_name if image_id in rec_by_id else "",
                "gt_idx": gi,
                "x1": float(box[0].item()),
                "y1": float(box[1].item()),
                "x2": float(box[2].item()),
                "y2": float(box[3].item()),
                "width_px": bw,
                "height_px": bh,
                "area_px": area,
                "area_ratio": area_ratio,
                "aspect_ratio_w_over_h": ar,
                "size_bin": bin_name,
                "matched": 1 if gi in matched_gt_set else 0,
                "matched_iou": matched_iou,
                "matched_pred_idx": matched_pred_idx,
                "matched_pred_score": matched_pred_score,
                "center_err_px": center_err_px,
                "center_err_norm_diag": center_err_norm_diag,
            })

            stats = size_bucket_stats[bin_name]
            stats["total"].append(1.0)
            stats["matched"].append(1.0 if gi in matched_gt_set else 0.0)
            if gi in matched_gt_set:
                local_idx = matches["matched_gt_indices"].tolist().index(gi)
                stats["iou"].append(float(matches["matched_ious"][local_idx].item()))
                pi = int(matches["matched_pred_indices"][local_idx].item())
                pcx = 0.5 * float((pred_boxes[pi, 0] + pred_boxes[pi, 2]).item())
                pcy = 0.5 * float((pred_boxes[pi, 1] + pred_boxes[pi, 3]).item())
                gcx = 0.5 * float((box[0] + box[2]).item())
                gcy = 0.5 * float((box[1] + box[3]).item())
                de = ((pcx - gcx) ** 2 + (pcy - gcy) ** 2) ** 0.5
                stats["center_err_px"].append(de)
                stats["center_err_norm_diag"].append(de / image_diag)

        # Pred row writing + conf bucketing.
        for pi in range(n_pred):
            pbox = pred_boxes[pi]
            pscore = float(pred_scores[pi].item())
            pw = float((pbox[2] - pbox[0]).clamp(min=0).item())
            ph = float((pbox[3] - pbox[1]).clamp(min=0).item())
            all_pred_confs.append(pscore)
            all_pred_widths_px.append(pw)
            all_pred_heights_px.append(ph)
            is_tp = pi in matched_pred_set
            matched_gt_idx = ""
            matched_iou = 0.0
            if is_tp:
                local_idx = matches["matched_pred_indices"].tolist().index(pi)
                matched_gt_idx = int(matches["matched_gt_indices"][local_idx].item())
                matched_iou = float(matches["matched_ious"][local_idx].item())

            pred_rows.append({
                "image_id": image_id,
                "source": rec_by_id[image_id].source_name if image_id in rec_by_id else "",
                "pred_idx": pi,
                "score": pscore,
                "x1": float(pbox[0].item()),
                "y1": float(pbox[1].item()),
                "x2": float(pbox[2].item()),
                "y2": float(pbox[3].item()),
                "width_px": pw,
                "height_px": ph,
                "is_tp": int(is_tp),
                "matched_gt_idx": matched_gt_idx,
                "matched_iou": matched_iou,
            })

            # Bucket conf into width 0.05. The `+ 1e-9` defends against float-precision rounding
            # so scores like 0.60/0.70/0.85 don't get keyed as 0.6000000000000001 (which would not
            # match the rounded `conf_bucket_edges` lookup keys and would silently drop counts).
            bucket_idx = max(0, min(19, int(pscore / 0.05 + 1e-9)))
            bucket_lo = round(0.05 * bucket_idx, 2)
            if is_tp:
                conf_buckets_tp[bucket_lo] += 1
            else:
                conf_buckets_fp[bucket_lo] += 1

        # Source-level accumulation.
        src_name = rec_by_id[image_id].source_name if image_id in rec_by_id else "unknown"
        sstats = source_stats[src_name]
        sstats["images"] = int(sstats["images"]) + 1
        sstats["gt"] = int(sstats["gt"]) + n_gt
        sstats["tp"] = int(sstats["tp"]) + tp
        sstats["fp"] = int(sstats["fp"]) + fp
        sstats["fn"] = int(sstats["fn"]) + fn
        sstats["ious"].extend(ious_list)
        sstats["center_err_px"].extend(matched_center_err_px)

        # Global running lists.
        all_matched_iou.extend(ious_list)
        all_center_err_px.extend(matched_center_err_px)
        all_center_err_norm_diag.extend(matched_center_err_norm_diag)

        # Image-level flags.
        if n_gt == 0:
            n_no_gt += 1
            if fp == 0:
                n_empty_correct += 1
            else:
                n_false_alarm_only += 1
        else:
            if tp == n_gt and fp == 0:
                n_perfect += 1
            if tp == 0:
                n_missed_entirely += 1

        per_image_rows.append({
            "idx": idx,
            "image_id": image_id,
            "source": src_name,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "num_gt": n_gt,
            "num_pred": n_pred,
            "num_pred_raw_for_ap": n_pred_raw,
            "top_op_conf": top_op_conf,
            "top_raw_conf": top_raw_conf,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "best_iou_any_pair": best_iou_any,
            "mean_matched_iou": (sum(ious_list) / len(ious_list)) if ious_list else 0.0,
            "mean_center_err_px": (sum(matched_center_err_px) / len(matched_center_err_px)) if matched_center_err_px else "",
            "mean_center_err_norm_diag": (sum(matched_center_err_norm_diag) / len(matched_center_err_norm_diag)) if matched_center_err_norm_diag else "",
            "gt_size_bins": ";".join(gt_size_bins) if gt_size_bins else "",
        })

        # 9) Per-image overlay PNG.
        if not args.no_images:
            try:
                rec = rec_by_id[image_id]
                orig_image = Image.open(rec.image_path).convert("RGB")
                draw = ImageDraw.Draw(orig_image)
                font = _font()

                # Draw matched GTs (TP) in lime, unmatched GTs (FN) in orange.
                for gi in range(n_gt):
                    box = gt_boxes[gi].tolist()
                    is_matched = gi in matched_gt_set
                    color = GT_TP_COLOR if is_matched else GT_FN_COLOR
                    label = "GT" if is_matched else "GT-FN"
                    _draw_box_with_label(draw, box, color, label, font, width=2)

                # Draw predictions: matched (TP) cyan, unmatched (FP) red.
                for pi in range(n_pred):
                    box = pred_boxes[pi].tolist()
                    is_tp = pi in matched_pred_set
                    score = float(pred_scores[pi].item())
                    color = PRED_TP_COLOR if is_tp else PRED_FP_COLOR
                    prefix = "TP" if is_tp else "FP"
                    label = f"{prefix} {score:.2f}"
                    _draw_box_with_label(draw, box, color, label, font, width=2)

                stem = Path(image_id).stem.replace("/", "_")
                out_name = f"{idx:04d}_{src_name}_{stem}.png"
                main_path = images_out_dir / out_name
                orig_image.save(main_path)

                # Mirror into fp_images/ if at least one prediction was a false positive.
                # Use a hard link (same inode, zero extra disk) and fall back to a real copy
                # only if the FS rejects the link (e.g. cross-device).
                if fp > 0:
                    fp_path = fp_images_out_dir / out_name
                    try:
                        if fp_path.exists():
                            fp_path.unlink()
                        import os as _os
                        _os.link(main_path, fp_path)
                    except OSError:
                        try:
                            import shutil as _shutil
                            _shutil.copy2(main_path, fp_path)
                        except Exception as link_exc:
                            logger.warning(f"[WARN] failed to mirror {out_name} into fp_images/: {link_exc}")
            except Exception as exc:
                logger.warning(f"[WARN] failed to save overlay for {image_id}: {exc}")

        if (idx + 1) % 200 == 0:
            logger.info(f"[INFO] processed {idx+1}/{len(targets_all)} images")

    # 10) Write per-image CSV.
    per_image_csv = out_dir / "per_image_metrics.csv"
    per_image_columns = [
        "idx", "image_id", "source", "orig_w", "orig_h",
        "num_gt", "num_pred", "num_pred_raw_for_ap",
        "top_op_conf", "top_raw_conf",
        "tp", "fp", "fn",
        "best_iou_any_pair", "mean_matched_iou",
        "mean_center_err_px", "mean_center_err_norm_diag",
        "gt_size_bins",
    ]
    with per_image_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_image_columns)
        w.writeheader()
        for r in per_image_rows:
            w.writerow({k: r.get(k, "") for k in per_image_columns})

    # 11) Write predictions CSV.
    pred_csv = out_dir / "predictions.csv"
    pred_columns = [
        "image_id", "source", "pred_idx", "score",
        "x1", "y1", "x2", "y2", "width_px", "height_px",
        "is_tp", "matched_gt_idx", "matched_iou",
    ]
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pred_columns)
        w.writeheader()
        for r in pred_rows:
            w.writerow({k: r.get(k, "") for k in pred_columns})

    # 12) Write GT CSV.
    gt_csv = out_dir / "gt_boxes.csv"
    gt_columns = [
        "image_id", "source", "gt_idx",
        "x1", "y1", "x2", "y2", "width_px", "height_px",
        "area_px", "area_ratio", "aspect_ratio_w_over_h", "size_bin",
        "matched", "matched_iou", "matched_pred_idx", "matched_pred_score",
        "center_err_px", "center_err_norm_diag",
    ]
    with gt_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gt_columns)
        w.writeheader()
        for r in gt_rows:
            w.writerow({k: r.get(k, "") for k in gt_columns})

    # 13) Size bucket metrics CSV.
    size_csv = out_dir / "size_bucket_metrics.csv"
    size_columns = [
        "size_bin", "gt_total", "gt_matched", "recall",
        "iou_mean", "iou_median", "iou_p95",
        "center_err_px_mean", "center_err_px_median", "center_err_px_p95",
        "center_err_norm_diag_mean", "center_err_norm_diag_median", "center_err_norm_diag_p95",
    ]
    with size_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=size_columns)
        w.writeheader()
        for bin_name in ("small", "medium", "large"):
            s = size_bucket_stats[bin_name]
            total = int(sum(s["total"]))
            matched_c = int(sum(s["matched"]))
            iou_summary = _stats_summary(s["iou"])
            cep_summary = _stats_summary(s["center_err_px"])
            cen_summary = _stats_summary(s["center_err_norm_diag"])
            w.writerow({
                "size_bin": bin_name,
                "gt_total": total,
                "gt_matched": matched_c,
                "recall": _safe_div(matched_c, total),
                "iou_mean": iou_summary["mean"],
                "iou_median": iou_summary["median"],
                "iou_p95": iou_summary["p95"],
                "center_err_px_mean": cep_summary["mean"],
                "center_err_px_median": cep_summary["median"],
                "center_err_px_p95": cep_summary["p95"],
                "center_err_norm_diag_mean": cen_summary["mean"],
                "center_err_norm_diag_median": cen_summary["median"],
                "center_err_norm_diag_p95": cen_summary["p95"],
            })

    # 14) Confidence bucket CSV.
    conf_csv = out_dir / "confidence_buckets.csv"
    with conf_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket_lo", "bucket_hi", "tp", "fp", "total", "precision"])
        for bucket_lo in conf_bucket_edges:
            bucket_hi = round(bucket_lo + 0.05, 2)
            tp_c = int(conf_buckets_tp.get(bucket_lo, 0))
            fp_c = int(conf_buckets_fp.get(bucket_lo, 0))
            tot = tp_c + fp_c
            w.writerow([bucket_lo, bucket_hi, tp_c, fp_c, tot, _safe_div(tp_c, tot)])

    # 15) Per-source breakdown.
    src_csv = out_dir / "source_breakdown.csv"
    with src_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "images", "gt", "tp", "fp", "fn", "precision", "recall", "f1", "mean_iou", "mean_center_err_px"])
        for src_name in sorted(source_stats.keys()):
            s = source_stats[src_name]
            tp_c = int(s["tp"])
            fp_c = int(s["fp"])
            fn_c = int(s["fn"])
            p = _safe_div(tp_c, tp_c + fp_c)
            r = _safe_div(tp_c, tp_c + fn_c)
            f1 = _safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
            mean_iou = mean(s["ious"]) if s["ious"] else 0.0
            mean_ce = mean(s["center_err_px"]) if s["center_err_px"] else 0.0
            w.writerow([src_name, int(s["images"]), int(s["gt"]), tp_c, fp_c, fn_c, p, r, f1, mean_iou, mean_ce])

    # 16) Summary JSON.
    summary = {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "checkpoint": str(ckpt_path),
        "ckpt_epoch": epoch_in_ckpt,
        "ckpt_best_metric": best_metric_in_ckpt,
        "model_params": int(n_params),
        "n_requested": n_request,
        "n_sampled": len(sampled),
        "all_images_mode": all_images_mode,
        "sample_from": args.sample_from,
        "device": str(device),
        "input_size_hw": [in_h, in_w],
        "eval_params": {
            "conf_threshold": conf_threshold,
            "nms_iou_threshold": nms_iou,
            "max_detections": max_dets,
            "use_nms": use_nms,
            "single_object_mode": single_obj,
            "ap_conf_threshold": ap_conf,
            "ap_max_detections": ap_max_dets,
            "matching_iou_threshold": matching_iou,
            "small_area_threshold": small_t,
            "medium_area_threshold": medium_t,
        },
        "aggregate": agg,
        "image_level": {
            "n_total": len(targets_all),
            "n_with_gt": len(targets_all) - n_no_gt,
            "n_no_gt": n_no_gt,
            "n_perfect": n_perfect,
            "n_missed_entirely": n_missed_entirely,
            "n_false_alarm_only": n_false_alarm_only,
            "n_empty_correct": n_empty_correct,
        },
        "sampled_source_counts": dict(sample_source_counts),
        "gt_box_distribution_px": {
            "width": _stats_summary(all_gt_widths_px),
            "height": _stats_summary(all_gt_heights_px),
            "aspect_ratio_w_over_h": _stats_summary(all_gt_aspect_ratios),
            "area_ratio": _stats_summary(all_gt_area_ratios),
        },
        "pred_box_distribution_px": {
            "width": _stats_summary(all_pred_widths_px),
            "height": _stats_summary(all_pred_heights_px),
            "score": _stats_summary(all_pred_confs),
        },
        "matched_iou_distribution": _stats_summary(all_matched_iou),
        "center_err_px_distribution": _stats_summary(all_center_err_px),
        "center_err_norm_diag_distribution": _stats_summary(all_center_err_norm_diag),
        "timing": {
            "forward_ms_per_image_mean": mean(forward_times_ms) if forward_times_ms else 0.0,
            "forward_ms_per_image_p95": _percentile_sorted(sorted(forward_times_ms), 0.95) if forward_times_ms else 0.0,
            "decode_ms_per_image_mean": mean(decode_times_ms) if decode_times_ms else 0.0,
            "wall_time_sec_total": total_t1 - total_t0,
        },
    }

    summary_json = out_dir / "metrics_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"[INFO] wrote metrics_summary.json -> {summary_json}")
    logger.info(f"[INFO] wrote per_image_metrics.csv -> {per_image_csv}")
    logger.info(f"[INFO] wrote predictions.csv      -> {pred_csv}")
    logger.info(f"[INFO] wrote gt_boxes.csv         -> {gt_csv}")
    logger.info(f"[INFO] wrote size_bucket_metrics.csv -> {size_csv}")
    logger.info(f"[INFO] wrote confidence_buckets.csv  -> {conf_csv}")
    logger.info(f"[INFO] wrote source_breakdown.csv    -> {src_csv}")
    if not args.no_images:
        logger.info(f"[INFO] wrote {len(targets_all)} overlay PNGs -> {images_out_dir}")
        n_fp_imgs = sum(1 for r in per_image_rows if int(r["fp"]) > 0)
        logger.info(f"[INFO] mirrored {n_fp_imgs} FP overlays -> {fp_images_out_dir}")
    logger.info("[INFO] done")


if __name__ == "__main__":
    main()
