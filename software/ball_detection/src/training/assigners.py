from __future__ import annotations

from math import sqrt
from typing import List, Sequence

import torch


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
