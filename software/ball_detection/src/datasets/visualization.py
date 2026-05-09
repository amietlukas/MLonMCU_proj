from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageDraw


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    x = image_tensor.detach().cpu().clamp(0.0, 1.0)
    if x.shape[0] == 1:
        arr = (x[0].numpy() * 255.0).astype("uint8")
        return Image.fromarray(arr, mode="L")
    arr = (x.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def draw_boxes(image: Image.Image, boxes_xyxy: torch.Tensor, color: str = "red") -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    for b in boxes_xyxy.detach().cpu().tolist():
        draw.rectangle([b[0], b[1], b[2], b[3]], outline=color, width=2)
    return out


def save_batch_preview(
    images: torch.Tensor,
    targets: list[dict],
    out_dir: Path,
    prefix: str,
    max_images: int = 8,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(max_images, images.shape[0])
    for i in range(n):
        pil_img = tensor_to_pil(images[i])
        boxed = draw_boxes(pil_img, targets[i]["boxes"], color="lime")
        image_id_raw = str(targets[i].get("image_id", i))
        image_id = Path(image_id_raw).stem.replace("/", "_")
        boxed.save(out_dir / f"{prefix}_{i:02d}_{image_id}.png")


def save_prediction_preview(
    images: torch.Tensor,
    pred_boxes: Iterable[torch.Tensor],
    targets: list[dict],
    out_dir: Path,
    prefix: str,
    max_images: int = 8,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_list = list(pred_boxes)
    n = min(max_images, images.shape[0], len(pred_list))

    for i in range(n):
        pil_img = tensor_to_pil(images[i])
        gt_img = draw_boxes(pil_img, targets[i]["boxes"], color="lime")
        pr_img = draw_boxes(gt_img, pred_list[i], color="red")
        image_id_raw = str(targets[i].get("image_id", i))
        image_id = Path(image_id_raw).stem.replace("/", "_")
        pr_img.save(out_dir / f"{prefix}_{i:02d}_{image_id}.png")
