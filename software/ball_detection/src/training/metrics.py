from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import torch

from ball_detection.src.utils.boxes import box_iou, centers_of_boxes, resized_to_original
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


def _area_ratio_bin(area_ratio: float, small_threshold: float, medium_threshold: float) -> str:
    if area_ratio < small_threshold:
        return "small"
    if area_ratio < medium_threshold:
        return "medium"
    return "large"


def _voc_ap(rec: torch.Tensor, prec: torch.Tensor) -> float:
    mrec = torch.cat([torch.tensor([0.0], device=rec.device), rec, torch.tensor([1.0], device=rec.device)])
    mpre = torch.cat([torch.tensor([0.0], device=prec.device), prec, torch.tensor([0.0], device=prec.device)])

    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap.item())


def _compute_ap_for_iou(
    pred_items: list[PredItem],
    gt_by_image: Dict[str, torch.Tensor],
    iou_threshold: float,
) -> float:
    npos = int(sum(v.shape[0] for v in gt_by_image.values()))
    if npos == 0:
        return 0.0

    pred_items = sorted(pred_items, key=lambda x: x.score, reverse=True)

    matched = {k: torch.zeros((v.shape[0],), dtype=torch.bool) for k, v in gt_by_image.items()}
    tp = []
    fp = []

    for item in pred_items:
        gt = gt_by_image.get(item.image_id)
        if gt is None or gt.numel() == 0:
            tp.append(0.0)
            fp.append(1.0)
            continue

        ious = box_iou(item.box.unsqueeze(0), gt).squeeze(0)
        best_iou, best_idx = (ious.max(dim=0) if ious.numel() > 0 else (torch.tensor(0.0), torch.tensor(0)))
        bi = int(best_idx.item())

        if float(best_iou.item()) >= iou_threshold and not matched[item.image_id][bi]:
            matched[item.image_id][bi] = True
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    tp_t = torch.tensor(tp, dtype=torch.float32)
    fp_t = torch.tensor(fp, dtype=torch.float32)

    tp_c = torch.cumsum(tp_t, dim=0)
    fp_c = torch.cumsum(fp_t, dim=0)

    rec = tp_c / float(npos)
    prec = tp_c / (tp_c + fp_c + 1e-9)

    return _voc_ap(rec, prec)


def _match_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
):
    if pred_boxes.numel() == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": int(gt_boxes.shape[0]),
            "matched_pred_indices": torch.zeros((0,), dtype=torch.long),
            "matched_gt_indices": torch.zeros((0,), dtype=torch.long),
            "matched_ious": torch.zeros((0,), dtype=torch.float32),
        }

    order = pred_scores.argsort(descending=True)
    used_gt = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=pred_boxes.device)

    tp = 0
    fp = 0
    matched_pred_idx = []
    matched_gt_idx = []
    matched_ious = []

    for idx in order:
        i = int(idx.item())
        pb = pred_boxes[i : i + 1]

        if gt_boxes.numel() == 0:
            fp += 1
            continue

        ious = box_iou(pb, gt_boxes).squeeze(0)
        if ious.numel() == 0:
            fp += 1
            continue

        best_iou, best_gt = ious.max(dim=0)
        best_gt_i = int(best_gt.item())

        if float(best_iou.item()) >= iou_threshold and not bool(used_gt[best_gt_i].item()):
            used_gt[best_gt_i] = True
            tp += 1
            matched_pred_idx.append(i)
            matched_gt_idx.append(best_gt_i)
            matched_ious.append(float(best_iou.item()))
        else:
            fp += 1

    fn = int(gt_boxes.shape[0] - used_gt.sum().item())

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

    aps = []
    for thr in iou_thresholds:
        aps.append(_compute_ap_for_iou(pred_items=pred_items, gt_by_image=gt_by_image, iou_threshold=float(thr)))

    map50 = _compute_ap_for_iou(pred_items=pred_items, gt_by_image=gt_by_image, iou_threshold=0.5)
    map5095 = float(sum(aps) / max(len(aps), 1))

    op_by_image = {p.image_id: p for p in operating_predictions}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    matched_ious = []
    center_errors = []

    count_small = 0
    count_medium = 0
    count_large = 0
    match_small = 0
    match_medium = 0
    match_large = 0

    for image_id, gt_boxes in gt_by_image.items():
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
            center_errors.extend(center_dist.tolist())

        orig_h, orig_w = gt_meta[image_id]["orig_size"]
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
        "mean_iou": float(sum(matched_ious) / max(len(matched_ious), 1)),
        "center_error_px": float(sum(center_errors) / max(len(center_errors), 1)),
        "fp_per_image": float(total_fp / max(len(gt_by_image), 1)),
        "recall_small": float(match_small / max(count_small, 1)),
        "recall_medium": float(match_medium / max(count_medium, 1)),
        "recall_large": float(match_large / max(count_large, 1)),
        "count_small": int(count_small),
        "count_medium": int(count_medium),
        "count_large": int(count_large),
    }

    return metrics
