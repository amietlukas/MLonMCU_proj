"""Shared visualization helpers for the BallDetector_N6 host tools.

Looks up ground truth from the dataset's per-folder annotations.csv and renders
predictions (red) + GT ball (green) onto the 384x288 model-space image. Always
renders to a NEW file under n6_viz/ — never overwrites the (symlinked) dataset
images (datasets/BALL/*/ are symlink farms into archive/).
"""
from __future__ import annotations

import csv
from pathlib import Path
from PIL import Image, ImageDraw

MODEL_W, MODEL_H = 384, 288


def load_gt_model_space(image_path: Path) -> list[tuple[float, float, float, float]]:
    """GT ball boxes (x1,y1,x2,y2) in 384x288 space.

    Reads the sibling annotations.csv (columns: filename,x,y,width,height — top-left
    xywh in ORIGINAL image pixels) and scales to model space by the image's own size.
    Returns [] if there is no annotations.csv or no row for this image.
    """
    image_path = Path(image_path)
    csv_path = image_path.parent / "annotations.csv"
    if not csv_path.is_file():
        return []
    try:
        with Image.open(image_path) as im:
            ow, oh = im.size
    except Exception:
        return []
    sx, sy = MODEL_W / ow, MODEL_H / oh
    out: list[tuple[float, float, float, float]] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("filename") != image_path.name:
                continue
            x, y, w, h = (float(row["x"]), float(row["y"]),
                          float(row["width"]), float(row["height"]))
            out.append((x * sx, y * sy, (x + w) * sx, (y + h) * sy))
    return out


def render_detections(image_path, preds, out_path, scale: int = 2, tag: str = "") -> Path:
    """Render GT (green) + predictions (red) onto the resized model-space image.

    preds: iterable of (x1, y1, x2, y2, score) in 384x288 model space.
    """
    image_path = Path(image_path)
    gts = load_gt_model_space(image_path)
    img = (Image.open(image_path).convert("RGB")
           .resize((MODEL_W, MODEL_H), Image.BILINEAR)
           .resize((MODEL_W * scale, MODEL_H * scale), Image.NEAREST))
    d = ImageDraw.Draw(img)
    for (x1, y1, x2, y2) in gts:
        d.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale], outline=(0, 255, 0), width=3)
        d.text((x1 * scale, max(0, y1 * scale - 11)), "GT", fill=(0, 255, 0))
    for (x1, y1, x2, y2, s) in preds:
        d.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale], outline=(255, 40, 40), width=2)
        d.text((x1 * scale, max(0, y1 * scale - 11)), f"{s:.2f}", fill=(255, 120, 120))
    if tag:
        d.text((4, 4), tag, fill=(255, 255, 0))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
