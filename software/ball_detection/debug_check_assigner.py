from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

if __package__ is None or __package__ == "":
    software_root = Path(__file__).resolve().parent.parent
    if str(software_root) not in sys.path:
        sys.path.insert(0, str(software_root))

from ball_detection.src.training.assigners import SimOTALiteAssigner, build_targets
from ball_detection.src.training.losses import BallDetectionLoss


def _resolve_config_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_file():
        return path.resolve()
    software_root = Path(__file__).resolve().parent.parent
    candidate = software_root / path
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Config not found: {path_value}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must contain a YAML dictionary")
    return cfg


def _make_raw_outputs(batch_size: int, input_size: tuple[int, int], strides: list[int]) -> tuple[torch.Tensor, ...]:
    in_h, in_w = input_size
    outs = []
    for stride in strides:
        outs.append(torch.zeros((batch_size, 5, in_h // int(stride), in_w // int(stride)), dtype=torch.float32))
    return tuple(outs)


def _make_targets() -> list[dict]:
    return [
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        },
        {
            "boxes": torch.tensor([[300.0, 220.0, 340.0, 260.0]], dtype=torch.float32),
            "labels": torch.zeros((1,), dtype=torch.long),
        },
        {
            "boxes": torch.tensor(
                [
                    [96.0, 96.0, 136.0, 136.0],
                    [96.0, 96.0, 136.0, 136.0],
                ],
                dtype=torch.float32,
            ),
            "labels": torch.zeros((2,), dtype=torch.long),
        },
    ]


def _assert_positive_boxes_valid(box_targets: list[torch.Tensor], pos_masks: list[torch.Tensor]) -> None:
    for box_t, pos_m in zip(box_targets, pos_masks):
        pos = pos_m.squeeze(1)
        if not pos.any():
            continue
        boxes = box_t.permute(0, 2, 3, 1)[pos]
        if not torch.isfinite(boxes).all():
            raise AssertionError("positive target boxes contain NaN/Inf")
        if not ((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])).all():
            raise AssertionError("positive target boxes are not valid xyxy")


def _assert_grid_mapping(assigner: SimOTALiteAssigner, output_shapes: list[tuple[int, int]]) -> None:
    grid = assigner._build_flat_grid(output_shapes, device=torch.device("cpu"))
    expected = sum(h * w for h, w in output_shapes)
    if int(grid.centers.shape[0]) != expected:
        raise AssertionError("flat grid size does not match summed output cells")
    for scale_idx, (h, w) in enumerate(output_shapes):
        level = grid.scale_indices == scale_idx
        if int(level.sum().item()) != h * w:
            raise AssertionError(f"scale {scale_idx} flat grid count mismatch")
        if not (grid.ys[level].min() >= 0 and grid.ys[level].max() < h):
            raise AssertionError(f"scale {scale_idx} y index out of range")
        if not (grid.xs[level].min() >= 0 and grid.xs[level].max() < w):
            raise AssertionError(f"scale {scale_idx} x index out of range")


def run_checks(cfg: dict[str, Any]) -> None:
    strides = [int(s) for s in cfg["model"]["strides"]]
    input_size = (int(cfg["input"]["height"]), int(cfg["input"]["width"]))
    raw_outputs = _make_raw_outputs(batch_size=3, input_size=input_size, strides=strides)
    output_shapes = [(int(o.shape[2]), int(o.shape[3])) for o in raw_outputs]
    targets = _make_targets()

    center_obj, center_box, center_pos, _center_num_pos = build_targets(
        targets=targets,
        output_shapes=output_shapes,
        strides=strides,
        device=torch.device("cpu"),
        assign_scale_target=float(cfg["loss"].get("assign_scale_target", 3.0)),
        conflict_policy=str(cfg["loss"].get("assign_conflict_policy", "largest_area")),
        center_radius=int(cfg["loss"].get("assign_center_radius", 0)),
    )

    assigner_cfg = cfg.get("assigner", {})
    if str(assigner_cfg.get("type", "center")).lower().strip() != "simota_lite":
        raise AssertionError("debug_check_assigner expects assigner.type: simota_lite")

    simota = SimOTALiteAssigner(
        strides=strides,
        input_size=input_size,
        simota_cfg=assigner_cfg.get("simota", {}),
        decode_twth_clamp_min=float(cfg["loss"].get("decode_twth_clamp_min", -4.0)),
        decode_twth_clamp_max=float(cfg["loss"].get("decode_twth_clamp_max", 4.0)),
        assign_scale_target=float(cfg["loss"].get("assign_scale_target", 3.0)),
    )
    sim_obj, sim_box, sim_pos, sim_num_pos, stats = simota.assign(
        raw_outputs=raw_outputs,
        targets=targets,
        output_shapes=output_shapes,
        device=torch.device("cpu"),
    )
    sim_obj_2, sim_box_2, sim_pos_2, sim_num_pos_2, stats_2 = simota.assign(
        raw_outputs=raw_outputs,
        targets=targets,
        output_shapes=output_shapes,
        device=torch.device("cpu"),
    )

    if [x.shape for x in sim_obj] != [x.shape for x in center_obj]:
        raise AssertionError("objectness target shapes differ from center assigner")
    if [x.shape for x in sim_box] != [x.shape for x in center_box]:
        raise AssertionError("box target shapes differ from center assigner")
    if [x.shape for x in sim_pos] != [x.shape for x in center_pos]:
        raise AssertionError("positive mask shapes differ from center assigner")

    if any(pm[0].any().item() for pm in sim_pos):
        raise AssertionError("empty image produced positives")
    if sim_num_pos <= 0:
        raise AssertionError("SimOTA-lite produced no positives")
    if sum(int(pm[1].sum().item()) for pm in sim_pos) <= 0:
        raise AssertionError("one-GT image did not create a positive")
    if max(stats.positives_per_gt or [0]) > int(assigner_cfg["simota"].get("max_k", 5)):
        raise AssertionError("max positives per GT was exceeded")
    if stats.conflict_count <= 0:
        raise AssertionError("synthetic overlapping GTs did not exercise conflict resolution")
    if stats.nonfinite_cost_count != 0:
        raise AssertionError("matching costs contained NaN/Inf")

    for lhs, rhs in zip(sim_obj, sim_obj_2):
        if not torch.equal(lhs, rhs):
            raise AssertionError("objectness targets are not deterministic")
    for lhs, rhs in zip(sim_box, sim_box_2):
        if not torch.equal(lhs, rhs):
            raise AssertionError("box targets are not deterministic")
    for lhs, rhs in zip(sim_pos, sim_pos_2):
        if not torch.equal(lhs, rhs):
            raise AssertionError("positive masks are not deterministic")
    if sim_num_pos != sim_num_pos_2 or stats.conflict_count != stats_2.conflict_count:
        raise AssertionError("debug stats are not deterministic")

    _assert_positive_boxes_valid(sim_box, sim_pos)
    _assert_grid_mapping(simota, output_shapes)

    positives_by_stride = {
        stride: int(sim_pos[i].sum().item())
        for i, stride in enumerate(strides)
    }
    if sum(positives_by_stride.values()) != sim_num_pos:
        raise AssertionError("per-stride positive count does not sum to total positives")

    default_loss = BallDetectionLoss(strides=strides)
    if default_loss.assigner_type != "center":
        raise AssertionError("missing assigner.type should default to center")

    simota_loss = BallDetectionLoss(
        strides=strides,
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
        assigner_cfg=assigner_cfg,
        input_size=input_size,
    )
    losses = simota_loss(raw_outputs, targets)
    for name, value in losses.items():
        if not torch.isfinite(value).all():
            raise AssertionError(f"loss output is not finite: {name}")

    print("[OK] SimOTA-lite assigner sanity checks passed")
    print(stats.format_debug())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight SimOTA-lite assignment sanity checks")
    parser.add_argument("--config", required=True, help="Path to a ball_detection YAML config")
    args = parser.parse_args()

    cfg = _load_yaml(_resolve_config_path(args.config))
    run_checks(cfg)


if __name__ == "__main__":
    main()
