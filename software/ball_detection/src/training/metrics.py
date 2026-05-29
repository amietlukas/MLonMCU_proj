from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch

from ball_detection.src.utils.boxes import box_iou, centers_of_boxes, resized_to_original, xyxy_to_xywh_topleft
from ball_detection.src.utils.nms import batched_nms


@dataclass
class PredItem:
    image_id: str
    score: float
    box: torch.Tensor


@dataclass
class ImagePrediction:
    image_id: str
    boxes: torch.Tensor
    scores: torch.Tensor
    boxes_xywh_topleft: torch.Tensor | None = None


def _area_ratio_bin(area_ratio: float, small_threshold: float, medium_threshold: float) -> str:
    if area_ratio < small_threshold:
        return "small"
    if area_ratio < medium_threshold:
        return "medium"
    return "large"


def _mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _median_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _percentile_or_zero(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    qq = min(max(float(q), 0.0), 1.0)
    pos = qq * (len(vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    t = pos - lo
    return (1.0 - t) * vals[lo] + t * vals[hi]


def _voc_ap(rec: torch.Tensor, prec: torch.Tensor) -> float:
    mrec = torch.cat([torch.tensor([0.0], device=rec.device), rec, torch.tensor([1.0], device=rec.device)])
    mpre = torch.cat([torch.tensor([0.0], device=prec.device), prec, torch.tensor([0.0], device=prec.device)])

    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap.item())


def _compute_aps_for_thresholds(
    pred_items: list[PredItem],
    gt_by_image: Dict[str, torch.Tensor],
    iou_thresholds: Sequence[float],
) -> list[float]:
    """Vectorized AP computation that handles multiple IoU thresholds in one pass.

    Computes IoU(image_preds, image_gts) once per image (instead of once per
    prediction × threshold), then walks the score-sorted predictions and assigns
    TP/FP independently per threshold. Returns one AP per input threshold,
    matching plain IoU-based VOC AP semantics.
    """
    n_thr = len(iou_thresholds)
    if n_thr == 0:
        return []
    npos = int(sum(v.shape[0] for v in gt_by_image.values()))
    n_pred = len(pred_items)
    if npos == 0 or n_pred == 0:
        return [0.0] * n_thr

    thr_t = torch.tensor([float(t) for t in iou_thresholds], dtype=torch.float32)

    # ---- Group predictions by image, precompute IoU per image (once) ----
    preds_per_image: dict[str, list[int]] = {}
    for i, item in enumerate(pred_items):
        preds_per_image.setdefault(item.image_id, []).append(i)

    best_iou_per_pred = torch.zeros(n_pred, dtype=torch.float32)
    best_gt_per_pred = torch.full((n_pred,), -1, dtype=torch.long)

    for image_id, idx_list in preds_per_image.items():
        gt = gt_by_image.get(image_id)
        if gt is None or gt.numel() == 0:
            continue
        pred_boxes_img = torch.stack(
            [pred_items[i].box for i in idx_list], dim=0
        ).float()
        ious = box_iou(pred_boxes_img, gt.float())  # (P, K)
        if ious.numel() == 0:
            continue
        best_iou_i, best_gt_i = ious.max(dim=1)
        idx_tensor = torch.tensor(idx_list, dtype=torch.long)
        best_iou_per_pred[idx_tensor] = best_iou_i
        best_gt_per_pred[idx_tensor] = best_gt_i

    # ---- Sort globally by score (descending), pre-materialize CPU views ----
    scores_t = torch.tensor([item.score for item in pred_items], dtype=torch.float32)
    order = scores_t.argsort(descending=True)

    best_iou_sorted = best_iou_per_pred[order].tolist()
    best_gt_sorted = best_gt_per_pred[order].tolist()
    image_id_sorted = [pred_items[int(idx)].image_id for idx in order.tolist()]

    # ---- Per-threshold matching trackers ----
    matched_per_thr: list[dict[str, list[bool]]] = [
        {k: [False] * int(v.shape[0]) for k, v in gt_by_image.items()}
        for _ in range(n_thr)
    ]
    tp_per_thr = [[0.0] * n_pred for _ in range(n_thr)]
    fp_per_thr = [[0.0] * n_pred for _ in range(n_thr)]
    thresholds_list = [float(t) for t in iou_thresholds]

    for rank in range(n_pred):
        img_id = image_id_sorted[rank]
        b_gt = best_gt_sorted[rank]
        b_iou = best_iou_sorted[rank]
        if b_gt < 0:
            # No GT in this image -- always a FP at every threshold.
            for t in range(n_thr):
                fp_per_thr[t][rank] = 1.0
            continue
        for t in range(n_thr):
            if b_iou >= thresholds_list[t] and not matched_per_thr[t][img_id][b_gt]:
                matched_per_thr[t][img_id][b_gt] = True
                tp_per_thr[t][rank] = 1.0
            else:
                fp_per_thr[t][rank] = 1.0

    aps: list[float] = []
    npos_f = float(npos)
    for t in range(n_thr):
        tp_t = torch.tensor(tp_per_thr[t], dtype=torch.float32)
        fp_t = torch.tensor(fp_per_thr[t], dtype=torch.float32)
        tp_c = torch.cumsum(tp_t, dim=0)
        fp_c = torch.cumsum(fp_t, dim=0)
        rec = tp_c / npos_f
        prec = tp_c / (tp_c + fp_c + 1e-9)
        aps.append(_voc_ap(rec, prec))
    return aps


def _compute_ap_for_iou(
    pred_items: list[PredItem],
    gt_by_image: Dict[str, torch.Tensor],
    iou_threshold: float,
) -> float:
    """Thin shim over the multi-threshold vectorized AP for the single-threshold call sites."""
    aps = _compute_aps_for_thresholds(
        pred_items=pred_items,
        gt_by_image=gt_by_image,
        iou_thresholds=[float(iou_threshold)],
    )
    return float(aps[0]) if aps else 0.0


def _match_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
):
    n_gt = int(gt_boxes.shape[0]) if gt_boxes.numel() > 0 else 0
    n_pred = int(pred_boxes.shape[0]) if pred_boxes.numel() > 0 else 0

    if n_pred == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": n_gt,
            "matched_pred_indices": torch.zeros((0,), dtype=torch.long),
            "matched_gt_indices": torch.zeros((0,), dtype=torch.long),
            "matched_ious": torch.zeros((0,), dtype=torch.float32),
        }

    order = pred_scores.argsort(descending=True)

    if n_gt == 0:
        return {
            "tp": 0,
            "fp": n_pred,
            "fn": 0,
            "matched_pred_indices": torch.zeros((0,), dtype=torch.long),
            "matched_gt_indices": torch.zeros((0,), dtype=torch.long),
            "matched_ious": torch.zeros((0,), dtype=torch.float32),
        }

    # Compute the full IoU matrix ONCE, then do greedy matching with plain
    # Python on small CPU tensors (per-image: P <= max_detections, K typically tiny).
    pred_boxes_ord = pred_boxes[order].cpu().float()
    gt_cpu = gt_boxes.cpu().float()
    ious_full = box_iou(pred_boxes_ord, gt_cpu)  # (P, K)
    ious_rows = ious_full.tolist()
    order_list = order.tolist()

    used_gt = [False] * n_gt
    tp = 0
    fp = 0
    matched_pred_idx: list[int] = []
    matched_gt_idx: list[int] = []
    matched_ious: list[float] = []
    thr = float(iou_threshold)

    for rank in range(n_pred):
        row = ious_rows[rank]
        best_iou = -1.0
        best_gt = -1
        for k in range(n_gt):
            if used_gt[k]:
                continue
            v = row[k]
            if v > best_iou:
                best_iou = v
                best_gt = k

        if best_gt >= 0 and best_iou >= thr:
            used_gt[best_gt] = True
            tp += 1
            matched_pred_idx.append(order_list[rank])
            matched_gt_idx.append(best_gt)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = n_gt - sum(used_gt)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_pred_indices": torch.tensor(matched_pred_idx, dtype=torch.long),
        "matched_gt_indices": torch.tensor(matched_gt_idx, dtype=torch.long),
        "matched_ious": torch.tensor(matched_ious, dtype=torch.float32),
    }


def _postprocess_single(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    conf_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    use_nms: bool,
    single_object_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = scores >= float(conf_threshold)
    boxes = boxes[keep]
    scores = scores[keep]

    if boxes.numel() == 0:
        return (
            torch.zeros((0, 4), dtype=boxes.dtype, device=boxes.device),
            torch.zeros((0,), dtype=scores.dtype, device=scores.device),
        )

    if single_object_mode:
        top = int(scores.argmax().item())
        idx = torch.tensor([top], dtype=torch.long, device=scores.device)
        return boxes[idx], scores[idx]

    if use_nms:
        idx = batched_nms(boxes, scores, iou_threshold=float(nms_iou_threshold))
    else:
        idx = scores.argsort(descending=True)
    idx = idx[: int(max_detections)]
    return boxes[idx], scores[idx]


def build_image_predictions(
    decoded_boxes: torch.Tensor,
    decoded_scores: torch.Tensor,
    targets: list[dict],
    conf_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    use_nms: bool,
    single_object_mode: bool,
    return_in_original: bool = True,
) -> list[ImagePrediction]:
    preds: list[ImagePrediction] = []
    for i in range(decoded_boxes.shape[0]):
        boxes_i = decoded_boxes[i]
        scores_i = decoded_scores[i]
        boxes_i, scores_i = _postprocess_single(
            boxes=boxes_i,
            scores=scores_i,
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            use_nms=use_nms,
            single_object_mode=single_object_mode,
        )

        target = targets[i]
        if return_in_original:
            pad_left, pad_top, _pr, _pb = target["pad"]
            scale_x = float(target.get("scale_x", target.get("scale", 1.0)))
            scale_y = float(target.get("scale_y", target.get("scale", 1.0)))
            orig_size = tuple(target["orig_size"])
            boxes_out = resized_to_original(
                boxes_i.detach().cpu(),
                scale_x=scale_x,
                scale_y=scale_y,
                pad_left=int(pad_left),
                pad_top=int(pad_top),
                orig_size=(int(orig_size[0]), int(orig_size[1])),
            )
        else:
            boxes_out = boxes_i.detach().cpu()

        preds.append(
            ImagePrediction(
                image_id=str(target["image_id"]),
                boxes=boxes_out,
                scores=scores_i.detach().cpu(),
                boxes_xywh_topleft=xyxy_to_xywh_topleft(boxes_out),
            )
        )
    return preds


def compute_detection_metrics(
    map_predictions: list[ImagePrediction],
    operating_predictions: list[ImagePrediction],
    targets_all: list[dict],
    iou_thresholds: Sequence[float],
    matching_iou_threshold: float,
    small_area_threshold: float,
    medium_area_threshold: float,
) -> Dict[str, float]:
    gt_by_image: dict[str, torch.Tensor] = {}
    gt_meta: dict[str, dict] = {}

    for t in targets_all:
        image_id = str(t["image_id"])
        gt_by_image[image_id] = t["boxes_orig"].detach().cpu().float()
        gt_meta[image_id] = {
            "orig_size": tuple(int(v) for v in t["orig_size"]),
        }

    pred_items: list[PredItem] = []
    for pred in map_predictions:
        for box, score in zip(pred.boxes, pred.scores):
            pred_items.append(PredItem(image_id=pred.image_id, score=float(score.item()), box=box.float()))

    # Bundle the requested sweep thresholds with the standalone 0.5 threshold so
    # the whole AP computation walks the pred list ONCE (instead of len(thresholds)+1 times).
    iou_thresholds_list = [float(t) for t in iou_thresholds]
    threshold_for_map = iou_thresholds_list + [0.5]
    all_aps = _compute_aps_for_thresholds(
        pred_items=pred_items,
        gt_by_image=gt_by_image,
        iou_thresholds=threshold_for_map,
    )
    aps = all_aps[: len(iou_thresholds_list)]
    map50 = all_aps[-1] if all_aps else 0.0
    map5095 = float(sum(aps) / max(len(aps), 1))

    op_by_image = {p.image_id: p for p in operating_predictions}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    matched_ious = []
    center_errors = []
    center_errors_norm_diag = []

    count_small = 0
    count_medium = 0
    count_large = 0
    match_small = 0
    match_medium = 0
    match_large = 0

    for image_id, gt_boxes in gt_by_image.items():
        orig_h, orig_w = gt_meta[image_id]["orig_size"]
        image_diag = max((float(orig_h) ** 2 + float(orig_w) ** 2) ** 0.5, 1e-9)

        pred = op_by_image.get(image_id)
        if pred is None:
            pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
            pred_scores = torch.zeros((0,), dtype=torch.float32)
        else:
            pred_boxes = pred.boxes.float()
            pred_scores = pred.scores.float()

        matches = _match_predictions(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            iou_threshold=float(matching_iou_threshold),
        )

        total_tp += int(matches["tp"])
        total_fp += int(matches["fp"])
        total_fn += int(matches["fn"])

        if matches["matched_ious"].numel() > 0:
            matched_ious.extend(matches["matched_ious"].tolist())

            p_idx = matches["matched_pred_indices"]
            g_idx = matches["matched_gt_indices"]
            pred_centers = centers_of_boxes(pred_boxes[p_idx])
            gt_centers = centers_of_boxes(gt_boxes[g_idx])
            center_dist = torch.linalg.norm(pred_centers - gt_centers, dim=1)
            center_vals = center_dist.tolist()
            center_errors.extend(center_vals)
            center_errors_norm_diag.extend([float(v) / image_diag for v in center_vals])

        denom = float(max(orig_h * orig_w, 1))

        matched_gt_set = set(matches["matched_gt_indices"].tolist())
        for gi, gt_box in enumerate(gt_boxes):
            area = float(((gt_box[2] - gt_box[0]).clamp(min=0) * (gt_box[3] - gt_box[1]).clamp(min=0)).item())
            area_ratio = area / denom
            size_bin = _area_ratio_bin(
                area_ratio,
                small_threshold=float(small_area_threshold),
                medium_threshold=float(medium_area_threshold),
            )

            if size_bin == "small":
                count_small += 1
                if gi in matched_gt_set:
                    match_small += 1
            elif size_bin == "medium":
                count_medium += 1
                if gi in matched_gt_set:
                    match_medium += 1
            else:
                count_large += 1
                if gi in matched_gt_set:
                    match_large += 1

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    metrics = {
        "map50": float(map50),
        "map5095": float(map5095),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": _mean_or_zero(matched_ious),
        "center_error_px": _mean_or_zero(center_errors),
        "center_error_px_median": _median_or_zero(center_errors),
        "center_error_px_p95": _percentile_or_zero(center_errors, 0.95),
        "center_error_norm_diag": _mean_or_zero(center_errors_norm_diag),
        "center_error_norm_diag_median": _median_or_zero(center_errors_norm_diag),
        "center_error_norm_diag_p95": _percentile_or_zero(center_errors_norm_diag, 0.95),
        "fp_per_image": float(total_fp / max(len(gt_by_image), 1)),
        "recall_small": float(match_small / max(count_small, 1)),
        "recall_medium": float(match_medium / max(count_medium, 1)),
        "recall_large": float(match_large / max(count_large, 1)),
        "count_small": int(count_small),
        "count_medium": int(count_medium),
        "count_large": int(count_large),
    }

    return metrics


SWEEP_CSV_COLUMNS = [
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


def build_threshold_grid(start: float, end: float, step: float) -> list[float]:
    """Inclusive grid of conf thresholds. Clipped to [0,1]."""
    if step <= 0:
        raise ValueError(f"threshold_sweep.step must be > 0 (got {step})")
    if end < start:
        raise ValueError(f"threshold_sweep.end ({end}) must be >= start ({start})")
    vals: list[float] = []
    cur = start
    guard = 0
    while cur <= end + 1e-12 and guard < 10000:
        vals.append(round(float(cur), 6))
        cur += step
        guard += 1
    uniq = sorted(set(v for v in vals if 0.0 <= v <= 1.0))
    if not uniq:
        raise ValueError("Threshold sweep grid empty after bounds filtering")
    return uniq


def run_conf_threshold_sweep(
    *,
    decoded_boxes_all: torch.Tensor,
    decoded_scores_all: torch.Tensor,
    targets_all: list[dict],
    conf_grid: Sequence[float],
    nms_iou_threshold: float,
    max_detections: int,
    use_nms: bool,
    single_object_mode: bool,
    iou_thresholds: Sequence[float],
    matching_iou_threshold: float,
    small_area_threshold: float,
    medium_area_threshold: float,
    ap_conf_threshold: float,
    ap_max_detections: int,
    ap_use_nms: bool,
) -> list[dict]:
    """For each conf threshold, build operating predictions and recompute deployment metrics.

    map_predictions are computed once (at ap_conf_threshold) and shared across the grid -- mAP
    does not depend on the deployment threshold, only P/R/F1/fp_per_image do.
    """
    map_preds_all = build_image_predictions(
        decoded_boxes=decoded_boxes_all,
        decoded_scores=decoded_scores_all,
        targets=targets_all,
        conf_threshold=float(ap_conf_threshold),
        nms_iou_threshold=float(nms_iou_threshold),
        max_detections=int(ap_max_detections),
        use_nms=bool(ap_use_nms),
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
            nms_iou_threshold=float(nms_iou_threshold),
            max_detections=int(max_detections),
            use_nms=bool(use_nms),
            single_object_mode=bool(single_object_mode),
            return_in_original=True,
        )
        m = compute_detection_metrics(
            map_predictions=map_preds_all,
            operating_predictions=op_preds_all,
            targets_all=targets_all,
            iou_thresholds=list(iou_thresholds),
            matching_iou_threshold=float(matching_iou_threshold),
            small_area_threshold=float(small_area_threshold),
            medium_area_threshold=float(medium_area_threshold),
        )
        rows.append({
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
        })
    return rows


def pick_best_sweep_row(rows: list[dict], objective: str) -> dict:
    """Choose the threshold maximizing `objective`, tie-breaking on F1, P, then fewest FPs."""
    key = str(objective).lower()
    if key not in {"f1", "precision", "recall"}:
        raise ValueError(f"threshold_sweep.objective must be one of f1/precision/recall (got {objective})")
    return max(rows, key=lambda r: (r[key], r["f1"], r["precision"], -r["fp_per_image"]))


def write_sweep_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SWEEP_CSV_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in SWEEP_CSV_COLUMNS})
