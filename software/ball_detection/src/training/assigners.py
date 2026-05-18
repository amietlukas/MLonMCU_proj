from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, List, Sequence

import torch
import torch.nn.functional as F

from ball_detection.src.training.decode import decode_single_scale_with_params
from ball_detection.src.utils.boxes import box_iou


def _select_scale_by_size(
    box_w: float,
    box_h: float,
    strides: Sequence[int],
    assign_scale_target: float,
) -> int:
    size = sqrt(max(box_w * box_h, 1e-6))
    targets = [abs((size / max(float(s), 1.0)) - float(assign_scale_target)) for s in strides]
    return int(torch.tensor(targets).argmin().item())


def build_targets(
    targets: List[dict],
    output_shapes: List[tuple[int, int]],
    strides: Sequence[int],
    device: torch.device,
    assign_scale_target: float = 3.0,
    conflict_policy: str = "largest_area",
    center_radius: int = 0,
):
    batch_size = len(targets)

    obj_targets: list[torch.Tensor] = []
    box_targets_abs: list[torch.Tensor] = []
    pos_masks: list[torch.Tensor] = []
    area_maps: list[torch.Tensor] = []

    for (h, w) in output_shapes:
        obj_targets.append(torch.zeros((batch_size, 1, h, w), dtype=torch.float32, device=device))
        box_targets_abs.append(torch.zeros((batch_size, 4, h, w), dtype=torch.float32, device=device))
        pos_masks.append(torch.zeros((batch_size, 1, h, w), dtype=torch.bool, device=device))
        area_maps.append(torch.full((batch_size, h, w), fill_value=-1.0, dtype=torch.float32, device=device))

    for b_idx, target in enumerate(targets):
        boxes = target["boxes"].to(device=device, dtype=torch.float32)
        if boxes.numel() == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 0.0 or bh <= 0.0:
                continue

            scale_idx = _select_scale_by_size(
                bw,
                bh,
                strides,
                assign_scale_target=float(assign_scale_target),
            )
            stride = float(strides[scale_idx])
            h, w = output_shapes[scale_idx]

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            gx = int(cx / stride)
            gy = int(cy / stride)

            if gx < 0 or gy < 0 or gx >= w or gy >= h:
                continue

            area = bw * bh
            r = max(int(center_radius), 0)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    gx_n = gx + dx
                    gy_n = gy + dy
                    if gx_n < 0 or gy_n < 0 or gx_n >= w or gy_n >= h:
                        continue

                    prev_area = float(area_maps[scale_idx][b_idx, gy_n, gx_n].item())
                    if conflict_policy == "largest_area":
                        if area <= prev_area:
                            continue
                    elif conflict_policy == "last":
                        pass
                    else:
                        raise ValueError(
                            f"Unsupported conflict_policy: {conflict_policy}. "
                            "Expected one of ['largest_area', 'last']."
                        )

                    area_maps[scale_idx][b_idx, gy_n, gx_n] = float(area)
                    obj_targets[scale_idx][b_idx, 0, gy_n, gx_n] = 1.0
                    pos_masks[scale_idx][b_idx, 0, gy_n, gx_n] = True
                    box_targets_abs[scale_idx][b_idx, :, gy_n, gx_n] = torch.tensor([x1, y1, x2, y2], device=device)

    num_pos = int(sum(pm.sum().item() for pm in pos_masks))
    return obj_targets, box_targets_abs, pos_masks, num_pos


def _mean_min_max(values: list[int | float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    vals = [float(v) for v in values]
    return sum(vals) / len(vals), min(vals), max(vals)


@dataclass
class FlatGrid:
    centers: torch.Tensor
    scale_indices: torch.Tensor
    ys: torch.Tensor
    xs: torch.Tensor
    strides: torch.Tensor


@dataclass
class SimOTABatchStats:
    images: int
    strides: tuple[int, ...]
    empty_images: int = 0
    gt_count: int = 0
    candidate_counts: list[int] = field(default_factory=list)
    positives_total: int = 0
    positives_per_gt: list[int] = field(default_factory=list)
    positives_per_stride: dict[int, int] = field(default_factory=dict)
    unmatched_gt_count: int = 0
    fallback_count: int = 0
    conflict_count: int = 0
    dynamic_k_values: list[int] = field(default_factory=list)
    nonfinite_cost_count: int = 0

    def __post_init__(self) -> None:
        for stride in self.strides:
            self.positives_per_stride.setdefault(int(stride), 0)

    def format_debug(self, context: str | None = None) -> str:
        cand_mean, cand_min, cand_max = _mean_min_max(self.candidate_counts)
        pos_mean, pos_min, pos_max = _mean_min_max(self.positives_per_gt)
        dyn_mean, dyn_min, dyn_max = _mean_min_max(self.dynamic_k_values)
        stride_lines = [
            f"  positives_stride{int(stride)}: {int(self.positives_per_stride.get(int(stride), 0))}"
            for stride in self.strides
        ]
        title = "[DEBUG] SimOTA batch stats:"
        if context:
            title = f"[DEBUG] SimOTA {context} batch stats:"
        lines = [
            title,
            f"  images: {self.images}",
            f"  empty_images: {self.empty_images}",
            f"  gt_count: {self.gt_count}",
            f"  candidate_count_mean/min/max: {cand_mean:.2f}/{cand_min:.0f}/{cand_max:.0f}",
            f"  positives_total: {self.positives_total}",
            f"  positives_per_gt_mean/min/max: {pos_mean:.2f}/{pos_min:.0f}/{pos_max:.0f}",
            *stride_lines,
            f"  unmatched_gt_count: {self.unmatched_gt_count}",
            f"  fallback_count: {self.fallback_count}",
            f"  conflict_count: {self.conflict_count}",
            f"  dynamic_k_mean/min/max: {dyn_mean:.2f}/{dyn_min:.0f}/{dyn_max:.0f}",
            f"  nonfinite_cost_count: {self.nonfinite_cost_count}",
        ]
        return "\n".join(lines)


class SimOTALiteAssigner:
    """YOLOX-style dynamic assigner adapted for one-class objectness-only heads.

    Candidate filtering uses grid cell centers. Matching costs use decoded
    prediction boxes from the shared decoder; the center term is normalized by
    the resized input-image diagonal.
    """

    def __init__(
        self,
        *,
        strides: Sequence[int],
        input_size: tuple[int, int] | None,
        simota_cfg: dict[str, Any] | None,
        decode_twth_clamp_min: float,
        decode_twth_clamp_max: float,
        assign_scale_target: float,
    ) -> None:
        cfg = dict(simota_cfg or {})
        self.strides = tuple(int(s) for s in strides)
        self.input_size = input_size
        self.center_radius = float(cfg.get("center_radius", 2.5))
        self.topk_iou = int(cfg.get("topk_iou", 10))
        self.min_k = int(cfg.get("min_k", 1))
        self.max_k = int(cfg.get("max_k", 5))
        self.iou_weight = float(cfg.get("iou_weight", 3.0))
        self.obj_weight = float(cfg.get("obj_weight", 1.0))
        self.center_weight = float(cfg.get("center_weight", 0.5))
        self.iou_cost_type = str(cfg.get("iou_cost_type", "neg_log_iou")).lower().strip()
        self.eps = float(cfg.get("eps", 1.0e-8))
        self.debug = bool(cfg.get("debug", False))
        self.debug_every_n_batches = max(int(cfg.get("debug_every_n_batches", 20)), 1)
        self.decode_twth_clamp_min = float(decode_twth_clamp_min)
        self.decode_twth_clamp_max = float(decode_twth_clamp_max)
        self.assign_scale_target = float(assign_scale_target)

        if self.center_radius < 0:
            raise ValueError("assigner.simota.center_radius must be >= 0")
        if self.topk_iou <= 0:
            raise ValueError("assigner.simota.topk_iou must be > 0")
        if self.min_k <= 0:
            raise ValueError("assigner.simota.min_k must be > 0")
        if self.max_k < self.min_k:
            raise ValueError("assigner.simota.max_k must be >= min_k")
        if self.iou_cost_type not in {"neg_log_iou", "one_minus_iou"}:
            raise ValueError("assigner.simota.iou_cost_type must be one of: neg_log_iou, one_minus_iou")
        if self.eps <= 0:
            raise ValueError("assigner.simota.eps must be > 0")

    def should_log_debug(self, call_idx: int) -> bool:
        if not self.debug:
            return False
        return call_idx == 1 or (call_idx % self.debug_every_n_batches) == 0

    def _resolve_input_size(self, output_shapes: list[tuple[int, int]]) -> tuple[int, int]:
        if self.input_size is not None:
            in_h, in_w = self.input_size
            return int(in_h), int(in_w)

        inferred_h = 0
        inferred_w = 0
        for (h, w), stride in zip(output_shapes, self.strides):
            inferred_h = max(inferred_h, int(h) * int(stride))
            inferred_w = max(inferred_w, int(w) * int(stride))
        if inferred_h <= 0 or inferred_w <= 0:
            raise ValueError("Could not infer positive input size from output shapes and strides")
        return inferred_h, inferred_w

    def _build_flat_grid(self, output_shapes: list[tuple[int, int]], device: torch.device) -> FlatGrid:
        centers_all: list[torch.Tensor] = []
        scale_indices_all: list[torch.Tensor] = []
        ys_all: list[torch.Tensor] = []
        xs_all: list[torch.Tensor] = []
        strides_all: list[torch.Tensor] = []

        for scale_idx, ((h, w), stride) in enumerate(zip(output_shapes, self.strides)):
            ys, xs = torch.meshgrid(
                torch.arange(int(h), device=device, dtype=torch.long),
                torch.arange(int(w), device=device, dtype=torch.long),
                indexing="ij",
            )
            centers = torch.stack(
                [
                    (xs.to(torch.float32) + 0.5) * float(stride),
                    (ys.to(torch.float32) + 0.5) * float(stride),
                ],
                dim=-1,
            ).reshape(-1, 2)

            n = int(h) * int(w)
            centers_all.append(centers)
            scale_indices_all.append(torch.full((n,), int(scale_idx), dtype=torch.long, device=device))
            ys_all.append(ys.reshape(-1))
            xs_all.append(xs.reshape(-1))
            strides_all.append(torch.full((n,), float(stride), dtype=torch.float32, device=device))

        return FlatGrid(
            centers=torch.cat(centers_all, dim=0),
            scale_indices=torch.cat(scale_indices_all, dim=0),
            ys=torch.cat(ys_all, dim=0),
            xs=torch.cat(xs_all, dim=0),
            strides=torch.cat(strides_all, dim=0),
        )

    def _flatten_decoded(self, raw_outputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        boxes_all: list[torch.Tensor] = []
        obj_logits_all: list[torch.Tensor] = []

        for pred, stride in zip(raw_outputs, self.strides):
            pred_detached = pred.detach().to(dtype=torch.float32)
            boxes_s, _ = decode_single_scale_with_params(
                pred_detached,
                stride=int(stride),
                decode_twth_clamp_min=self.decode_twth_clamp_min,
                decode_twth_clamp_max=self.decode_twth_clamp_max,
            )
            b = boxes_s.shape[0]
            boxes_all.append(boxes_s.permute(0, 2, 3, 1).reshape(b, -1, 4))
            obj_logits_all.append(pred_detached[:, 4, :, :].reshape(b, -1))

        return torch.cat(boxes_all, dim=1), torch.cat(obj_logits_all, dim=1)

    def _candidate_indices_for_gt(
        self,
        *,
        box: torch.Tensor,
        grid: FlatGrid,
        stats: SimOTABatchStats,
    ) -> torch.Tensor:
        x1, y1, x2, y2 = box
        gt_cx = 0.5 * (x1 + x2)
        gt_cy = 0.5 * (y1 + y2)

        centers = grid.centers
        inside_box = (
            (centers[:, 0] >= x1)
            & (centers[:, 0] <= x2)
            & (centers[:, 1] >= y1)
            & (centers[:, 1] <= y2)
        )
        radius_px = self.center_radius * grid.strides
        inside_center = (
            (centers[:, 0] >= gt_cx - radius_px)
            & (centers[:, 0] <= gt_cx + radius_px)
            & (centers[:, 1] >= gt_cy - radius_px)
            & (centers[:, 1] <= gt_cy + radius_px)
        )
        candidate_idx = torch.nonzero(inside_box | inside_center, as_tuple=False).flatten()
        if candidate_idx.numel() > 0:
            return candidate_idx

        bw = float((x2 - x1).clamp(min=0).item())
        bh = float((y2 - y1).clamp(min=0).item())
        scale_idx = _select_scale_by_size(
            bw,
            bh,
            strides=self.strides,
            assign_scale_target=self.assign_scale_target,
        )
        level_idx = torch.nonzero(grid.scale_indices == int(scale_idx), as_tuple=False).flatten()
        if level_idx.numel() == 0:
            return level_idx

        gt_center = torch.stack([gt_cx, gt_cy])
        level_centers = grid.centers[level_idx]
        dist2 = ((level_centers - gt_center.unsqueeze(0)) ** 2).sum(dim=1)
        nearest = level_idx[int(dist2.argmin().item())].view(1)
        stats.fallback_count += 1
        return nearest

    def _matching_cost(
        self,
        *,
        cand_boxes: torch.Tensor,
        cand_obj_logits: torch.Tensor,
        gt_box: torch.Tensor,
        image_diag: float,
        stats: SimOTABatchStats,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ious = box_iou(cand_boxes, gt_box.view(1, 4)).squeeze(1).clamp(min=0.0, max=1.0)
        if self.iou_cost_type == "neg_log_iou":
            iou_cost = -torch.log(ious.clamp(min=self.eps))
        else:
            iou_cost = 1.0 - ious

        obj_cost = F.binary_cross_entropy_with_logits(
            cand_obj_logits,
            torch.ones_like(cand_obj_logits),
            reduction="none",
        )

        pred_centers = 0.5 * (cand_boxes[:, 0:2] + cand_boxes[:, 2:4])
        gt_center = 0.5 * (gt_box[0:2] + gt_box[2:4])
        center_cost = torch.linalg.vector_norm(pred_centers - gt_center.unsqueeze(0), dim=1) / max(image_diag, self.eps)

        total_cost = (
            self.iou_weight * iou_cost
            + self.obj_weight * obj_cost
            + self.center_weight * center_cost
        )
        finite = torch.isfinite(total_cost) & torch.isfinite(ious)
        stats.nonfinite_cost_count += int((~finite).sum().item())
        total_cost = torch.nan_to_num(total_cost, nan=1.0e6, posinf=1.0e6, neginf=0.0)
        ious = torch.nan_to_num(ious, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0, max=1.0)
        return total_cost, ious

    def _dynamic_k(self, ious: torch.Tensor, num_candidates: int) -> int:
        topk = min(int(self.topk_iou), int(num_candidates))
        top_iou_values = torch.topk(ious, k=topk, largest=True).values
        dynamic_k = int(float(top_iou_values.sum().item()))
        dynamic_k = max(int(self.min_k), dynamic_k)
        dynamic_k = min(int(self.max_k), dynamic_k)
        dynamic_k = min(dynamic_k, int(num_candidates))
        return dynamic_k

    def assign(
        self,
        *,
        raw_outputs: Sequence[torch.Tensor],
        targets: List[dict],
        output_shapes: list[tuple[int, int]],
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], int, SimOTABatchStats]:
        batch_size = len(targets)
        if len(raw_outputs) != len(self.strides):
            raise ValueError(f"Expected {len(self.strides)} raw output heads, got {len(raw_outputs)}")
        if raw_outputs[0].shape[0] != batch_size:
            raise ValueError(f"raw output batch size {raw_outputs[0].shape[0]} != target batch size {batch_size}")

        obj_targets: list[torch.Tensor] = []
        box_targets_abs: list[torch.Tensor] = []
        pos_masks: list[torch.Tensor] = []
        for (h, w) in output_shapes:
            obj_targets.append(torch.zeros((batch_size, 1, h, w), dtype=torch.float32, device=device))
            box_targets_abs.append(torch.zeros((batch_size, 4, h, w), dtype=torch.float32, device=device))
            pos_masks.append(torch.zeros((batch_size, 1, h, w), dtype=torch.bool, device=device))

        stats = SimOTABatchStats(images=batch_size, strides=self.strides)
        input_h, input_w = self._resolve_input_size(output_shapes)
        image_diag = sqrt(float(input_h * input_h + input_w * input_w))
        grid = self._build_flat_grid(output_shapes, device=device)

        with torch.no_grad():
            decoded_boxes_flat, obj_logits_flat = self._flatten_decoded(raw_outputs)

            for b_idx, target in enumerate(targets):
                boxes = target["boxes"].to(device=device, dtype=torch.float32)
                if boxes.numel() == 0:
                    stats.empty_images += 1
                    continue

                valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes = boxes[valid]
                if boxes.numel() == 0:
                    stats.empty_images += 1
                    continue

                stats.gt_count += int(boxes.shape[0])
                assignments: dict[int, tuple[int, float]] = {}

                for gt_idx, gt_box in enumerate(boxes):
                    candidate_idx = self._candidate_indices_for_gt(
                        box=gt_box,
                        grid=grid,
                        stats=stats,
                    )
                    num_candidates = int(candidate_idx.numel())
                    stats.candidate_counts.append(num_candidates)
                    if num_candidates == 0:
                        stats.dynamic_k_values.append(0)
                        continue

                    cand_boxes = decoded_boxes_flat[b_idx, candidate_idx]
                    cand_obj_logits = obj_logits_flat[b_idx, candidate_idx]
                    total_cost, ious = self._matching_cost(
                        cand_boxes=cand_boxes,
                        cand_obj_logits=cand_obj_logits,
                        gt_box=gt_box,
                        image_diag=image_diag,
                        stats=stats,
                    )

                    dynamic_k = self._dynamic_k(ious, num_candidates=num_candidates)
                    stats.dynamic_k_values.append(dynamic_k)
                    if dynamic_k <= 0:
                        continue

                    try:
                        order = torch.argsort(total_cost, descending=False, stable=True)
                    except TypeError:
                        order = torch.argsort(total_cost, descending=False)
                    chosen_local = order[:dynamic_k]

                    for local_idx_t in chosen_local:
                        local_idx = int(local_idx_t.item())
                        global_idx = int(candidate_idx[local_idx].item())
                        cost_value = float(total_cost[local_idx].item())
                        prev = assignments.get(global_idx)
                        if prev is not None:
                            stats.conflict_count += 1
                            prev_gt_idx, prev_cost = prev
                            if cost_value > prev_cost or (cost_value == prev_cost and gt_idx >= prev_gt_idx):
                                continue
                        assignments[global_idx] = (gt_idx, cost_value)

                positives_this_image = [0 for _ in range(int(boxes.shape[0]))]
                for global_idx, (gt_idx, _cost) in sorted(assignments.items()):
                    scale_idx = int(grid.scale_indices[global_idx].item())
                    gy = int(grid.ys[global_idx].item())
                    gx = int(grid.xs[global_idx].item())
                    stride = int(self.strides[scale_idx])

                    obj_targets[scale_idx][b_idx, 0, gy, gx] = 1.0
                    pos_masks[scale_idx][b_idx, 0, gy, gx] = True
                    box_targets_abs[scale_idx][b_idx, :, gy, gx] = boxes[gt_idx]
                    positives_this_image[gt_idx] += 1
                    stats.positives_per_stride[stride] = stats.positives_per_stride.get(stride, 0) + 1

                stats.positives_per_gt.extend(positives_this_image)
                stats.unmatched_gt_count += sum(1 for count in positives_this_image if count <= 0)

        num_pos = int(sum(pm.sum().item() for pm in pos_masks))
        stats.positives_total = num_pos
        return obj_targets, box_targets_abs, pos_masks, num_pos, stats
