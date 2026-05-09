from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import ColorJitter

from ball_detection.src.utils.boxes import clamp_xyxy


@dataclass
class ResizeMeta:
    mode: str  # "letterbox" or "resize"
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    orig_size: Tuple[int, int]  # (H, W)
    resized_size: Tuple[int, int]  # (H, W)


def _resolve_interpolation(name: str) -> int:
    key = str(name).lower().strip()
    if key in {"nearest", "nn"}:
        return Image.NEAREST
    if key in {"bilinear", "linear"}:
        return Image.BILINEAR
    if key in {"bicubic", "cubic"}:
        return Image.BICUBIC
    if key == "lanczos":
        return Image.LANCZOS
    raise ValueError(f"Unsupported interpolation mode: {name}")


def letterbox_image_and_boxes(
    image: Image.Image,
    boxes_xyxy: torch.Tensor,
    out_w: int,
    out_h: int,
    interpolation: int,
    fill_value: int = 114,
) -> tuple[Image.Image, torch.Tensor, ResizeMeta]:
    orig_w, orig_h = image.size

    scale = min(out_w / max(orig_w, 1), out_h / max(orig_h, 1))
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = image.resize((new_w, new_h), interpolation)

    canvas = Image.new(resized.mode, (out_w, out_h), color=fill_value)
    pad_left = (out_w - new_w) // 2
    pad_top = (out_h - new_h) // 2
    pad_right = out_w - new_w - pad_left
    pad_bottom = out_h - new_h - pad_top
    canvas.paste(resized, (pad_left, pad_top))

    out_boxes = boxes_xyxy.clone()
    if out_boxes.numel() > 0:
        out_boxes[:, [0, 2]] = out_boxes[:, [0, 2]] * scale + pad_left
        out_boxes[:, [1, 3]] = out_boxes[:, [1, 3]] * scale + pad_top
        out_boxes = clamp_xyxy(out_boxes, width=out_w, height=out_h)

    meta = ResizeMeta(
        mode="letterbox",
        scale_x=scale,
        scale_y=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        orig_size=(orig_h, orig_w),
        resized_size=(out_h, out_w),
    )
    return canvas, out_boxes, meta


def resize_image_and_boxes(
    image: Image.Image,
    boxes_xyxy: torch.Tensor,
    out_w: int,
    out_h: int,
    interpolation: int,
) -> tuple[Image.Image, torch.Tensor, ResizeMeta]:
    orig_w, orig_h = image.size
    sx = float(out_w) / max(float(orig_w), 1.0)
    sy = float(out_h) / max(float(orig_h), 1.0)

    resized = image.resize((out_w, out_h), interpolation)
    out_boxes = boxes_xyxy.clone()
    if out_boxes.numel() > 0:
        out_boxes[:, [0, 2]] = out_boxes[:, [0, 2]] * sx
        out_boxes[:, [1, 3]] = out_boxes[:, [1, 3]] * sy
        out_boxes = clamp_xyxy(out_boxes, width=out_w, height=out_h)

    meta = ResizeMeta(
        mode="resize",
        scale_x=sx,
        scale_y=sy,
        pad_left=0,
        pad_top=0,
        pad_right=0,
        pad_bottom=0,
        orig_size=(orig_h, orig_w),
        resized_size=(out_h, out_w),
    )
    return resized, out_boxes, meta


def _to_tensor(image: Image.Image, color_mode: Literal["rgb", "grayscale"]) -> torch.Tensor:
    if color_mode == "rgb":
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t

    if color_mode == "grayscale":
        if image.mode != "L":
            image = image.convert("L")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0).contiguous()
        return t

    raise ValueError(f"Unsupported color_mode: {color_mode}")


class DetectionTransform:
    def __init__(
        self,
        *,
        train: bool,
        out_w: int,
        out_h: int,
        color_mode: Literal["rgb", "grayscale"],
        resize_policy: str,
        interpolation: str,
        letterbox_fill_value: int,
        horizontal_flip: bool,
        hflip_prob: float,
        color_jitter: bool,
        brightness: float,
        contrast: float,
        saturation: float,
        blur: bool,
        blur_prob: float = 0.2,
        blur_radius: float = 0.8,
    ) -> None:
        self.train = train
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.color_mode = color_mode
        self.resize_policy = str(resize_policy).lower()
        if self.resize_policy not in {"letterbox", "resize"}:
            raise ValueError(f"Unsupported resize_policy: {resize_policy}")
        self.interpolation = _resolve_interpolation(interpolation)
        self.letterbox_fill_value = int(letterbox_fill_value)
        self.horizontal_flip = horizontal_flip
        self.hflip_prob = hflip_prob
        self.color_jitter_enabled = color_jitter and color_mode == "rgb"
        self.blur = blur
        self.blur_prob = float(blur_prob)
        self.blur_radius = float(blur_radius)

        self.color_jitter = ColorJitter(
            brightness=float(brightness),
            contrast=float(contrast),
            saturation=float(saturation),
            hue=0.0,
        )

    def _hflip(self, image: Image.Image, boxes: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
        w, _ = image.size
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        out_boxes = boxes.clone()
        if out_boxes.numel() > 0:
            x1 = out_boxes[:, 0].clone()
            x2 = out_boxes[:, 2].clone()
            out_boxes[:, 0] = w - x2
            out_boxes[:, 2] = w - x1
        return flipped, out_boxes

    def __call__(self, image: Image.Image, boxes_xyxy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, ResizeMeta]:
        boxes = boxes_xyxy.clone().float()

        if self.train:
            if self.horizontal_flip and random.random() < self.hflip_prob:
                image, boxes = self._hflip(image, boxes)

            if self.color_jitter_enabled:
                image = self.color_jitter(image)

            if self.blur and random.random() < self.blur_prob:
                image = image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        if self.resize_policy == "letterbox":
            image_out, boxes_out, meta = letterbox_image_and_boxes(
                image=image,
                boxes_xyxy=boxes,
                out_w=self.out_w,
                out_h=self.out_h,
                interpolation=self.interpolation,
                fill_value=self.letterbox_fill_value,
            )
        else:
            image_out, boxes_out, meta = resize_image_and_boxes(
                image=image,
                boxes_xyxy=boxes,
                out_w=self.out_w,
                out_h=self.out_h,
                interpolation=self.interpolation,
            )

        tensor = _to_tensor(image_out, self.color_mode)
        return tensor, boxes_out, meta
