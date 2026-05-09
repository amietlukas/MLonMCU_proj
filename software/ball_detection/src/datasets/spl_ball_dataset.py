from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ball_detection.src.datasets.transforms_detection import DetectionTransform
from ball_detection.src.reproducibility import worker_init_fn
from ball_detection.src.utils.boxes import xywh_to_xyxy


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class SampleRecord:
    image_id: str
    image_path: Path
    orig_size: tuple[int, int]  # (H, W)
    boxes_orig: torch.Tensor  # [N,4] xyxy absolute in original image


class SPLBallDetectionDataset(Dataset):
    def __init__(
        self,
        records: List[SampleRecord],
        transform: DetectionTransform,
        class_name: str = "Ball",
    ) -> None:
        self.records = records
        self.transform = transform
        self.class_name = class_name

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        with Image.open(rec.image_path) as img:
            image = img.convert("RGB")

        img_tensor, boxes_resized, meta = self.transform(image, rec.boxes_orig)

        labels = torch.zeros((boxes_resized.shape[0],), dtype=torch.long)
        target = {
            "boxes": boxes_resized.float(),
            "labels": labels,
            "image_id": rec.image_id,
            "orig_size": rec.orig_size,
            "resized_size": meta.resized_size,
            "resize_mode": meta.mode,
            "scale_x": float(meta.scale_x),
            "scale_y": float(meta.scale_y),
            "scale": float(meta.scale_x),  # legacy alias for square letterbox workflows
            "pad": (meta.pad_left, meta.pad_top, meta.pad_right, meta.pad_bottom),
            "boxes_orig": rec.boxes_orig.clone().float(),
            "image_path": str(rec.image_path),
        }
        return img_tensor, target



def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _parse_csv_row_for_ball(
    row: List[str],
    class_name: str,
    img_w: int,
    img_h: int,
    bbox_format: str,
    logger,
    context: str,
) -> torch.Tensor:
    boxes: list[list[float]] = []

    if len(row) < 1:
        return torch.zeros((0, 4), dtype=torch.float32)

    i = 1
    while i < len(row):
        if i + 4 >= len(row):
            if logger is not None:
                logger.warning(f"[WARN] malformed annotation tail skipped: {context} | row={row[:8]}")
            break

        cls = row[i].strip()
        x = _safe_float(row[i + 1])
        y = _safe_float(row[i + 2])
        w = _safe_float(row[i + 3])
        h = _safe_float(row[i + 4])

        i += 5

        if cls != class_name:
            continue

        if None in {x, y, w, h}:
            if logger is not None:
                logger.warning(f"[WARN] invalid numeric box skipped: {context}")
            continue

        if bbox_format == "xywh_topleft":
            x1, y1, x2, y2 = xywh_to_xyxy(x, y, w, h)
        elif bbox_format == "cxcywh_center":
            x1 = float(x) - float(w) * 0.5
            y1 = float(y) - float(h) * 0.5
            x2 = float(x) + float(w) * 0.5
            y2 = float(y) + float(h) * 0.5
        elif bbox_format == "cxcywh_radius":
            x1 = float(x) - float(w)
            y1 = float(y) - float(h)
            x2 = float(x) + float(w)
            y2 = float(y) + float(h)
        else:
            raise ValueError(f"Unsupported bbox_format: {bbox_format}")

        if x2 <= x1 or y2 <= y1:
            if logger is not None:
                logger.warning(f"[WARN] degenerate box skipped: {context} -> {(x1, y1, x2, y2)}")
            continue

        x1 = max(0.0, min(float(img_w - 1), x1))
        y1 = max(0.0, min(float(img_h - 1), y1))
        x2 = max(0.0, min(float(img_w - 1), x2))
        y2 = max(0.0, min(float(img_h - 1), y2))

        if x2 <= x1 or y2 <= y1:
            if logger is not None:
                logger.warning(f"[WARN] invalid clamped box skipped: {context} -> {(x1, y1, x2, y2)}")
            continue

        boxes.append([x1, y1, x2, y2])

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def _find_csv_files(images_dir: Path) -> List[Path]:
    return sorted(images_dir.rglob("*.csv"))


def load_spl_ball_records(
    images_dir: Path,
    class_name: str,
    bbox_format: str,
    duplicate_policy: str,
    logger,
) -> tuple[List[SampleRecord], Dict[str, Any]]:
    csv_files = _find_csv_files(images_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {images_dir}")

    records_map: dict[str, SampleRecord] = {}
    total_rows = 0
    rows_with_ball = 0
    total_ball_boxes_rows = 0
    duplicate_row_overwrites = 0

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader, start=1):
                total_rows += 1
                if len(row) == 0:
                    if logger is not None:
                        logger.warning(f"[WARN] empty CSV row skipped: {csv_path}:{row_idx}")
                    continue

                image_name = row[0].strip()
                if not image_name:
                    if logger is not None:
                        logger.warning(f"[WARN] missing image name skipped: {csv_path}:{row_idx}")
                    continue

                image_path = csv_path.parent / image_name
                if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTS:
                    if logger is not None:
                        logger.warning(f"[WARN] image missing or unsupported extension skipped: {image_path}")
                    continue

                with Image.open(image_path) as img:
                    img_w, img_h = img.size

                context = f"{csv_path.name}:{row_idx}:{image_name}"
                boxes_orig = _parse_csv_row_for_ball(
                    row=row,
                    class_name=class_name,
                    img_w=img_w,
                    img_h=img_h,
                    bbox_format=bbox_format,
                    logger=logger,
                    context=context,
                )

                if boxes_orig.shape[0] > 0:
                    rows_with_ball += 1
                total_ball_boxes_rows += int(boxes_orig.shape[0])

                rel_image = image_path.relative_to(images_dir).as_posix()
                rec = SampleRecord(
                    image_id=rel_image,
                    image_path=image_path,
                    orig_size=(img_h, img_w),
                    boxes_orig=boxes_orig,
                )

                if rel_image in records_map:
                    duplicate_row_overwrites += 1
                    if duplicate_policy == "first":
                        continue
                    if duplicate_policy != "last":
                        raise ValueError(
                            f"Unsupported dataset.duplicate_policy: {duplicate_policy}. "
                            "Expected one of ['first', 'last']."
                        )

                records_map[rel_image] = rec

    records = [records_map[k] for k in sorted(records_map.keys())]
    if not records:
        raise RuntimeError("No records parsed from SPLBall dataset")

    bbox_ws: list[float] = []
    bbox_hs: list[float] = []
    total_ball_boxes = 0
    images_with_ball = 0
    for rec in records:
        if rec.boxes_orig.shape[0] > 0:
            images_with_ball += 1
        total_ball_boxes += int(rec.boxes_orig.shape[0])
        if rec.boxes_orig.numel() > 0:
            wh = (rec.boxes_orig[:, 2:4] - rec.boxes_orig[:, 0:2]).clamp(min=0)
            bbox_ws.extend(wh[:, 0].tolist())
            bbox_hs.extend(wh[:, 1].tolist())

    bbox_stats = {
        "bbox_width_min": min(bbox_ws) if bbox_ws else 0.0,
        "bbox_width_mean": (sum(bbox_ws) / len(bbox_ws)) if bbox_ws else 0.0,
        "bbox_width_max": max(bbox_ws) if bbox_ws else 0.0,
        "bbox_height_min": min(bbox_hs) if bbox_hs else 0.0,
        "bbox_height_mean": (sum(bbox_hs) / len(bbox_hs)) if bbox_hs else 0.0,
        "bbox_height_max": max(bbox_hs) if bbox_hs else 0.0,
    }

    stats = {
        "num_csv_files": len(csv_files),
        "num_images": len(records),
        "num_rows": total_rows,
        "duplicate_row_overwrites": duplicate_row_overwrites,
        "rows_with_ball": rows_with_ball,
        "rows_without_ball": total_rows - rows_with_ball,
        "images_with_ball": images_with_ball,
        "images_without_ball": len(records) - images_with_ball,
        "num_ball_boxes_rows": total_ball_boxes_rows,
        "num_ball_boxes": total_ball_boxes,
        **bbox_stats,
    }

    return records, stats


def _save_split(split_path: Path, image_ids: Iterable[str]) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as f:
        for image_id in image_ids:
            f.write(f"{image_id}\n")


def _load_split(split_path: Path) -> List[str]:
    with split_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def create_or_load_splits(
    image_ids: List[str],
    splits_dir: Path,
    split_ratio: float,
    seed: int,
    reuse_splits: bool,
    logger,
) -> tuple[List[str], List[str], Path, Path]:
    train_file = splits_dir / "train.txt"
    val_file = splits_dir / "val.txt"

    if reuse_splits and train_file.exists() and val_file.exists():
        train_ids = _load_split(train_file)
        val_ids = _load_split(val_file)
        available_ids = set(image_ids)
        train_set = set(train_ids)
        val_set = set(val_ids)

        split_valid = True
        if len(train_ids) == 0 or len(val_ids) == 0:
            split_valid = False
        if len(train_ids) != len(train_set) or len(val_ids) != len(val_set):
            split_valid = False
        if (train_set | val_set) - available_ids:
            split_valid = False
        if train_set & val_set:
            split_valid = False
        if (train_set | val_set) != available_ids:
            split_valid = False

        if split_valid:
            if logger is not None:
                logger.info(f"[INFO] Reusing existing split files: {train_file}, {val_file}")
            return train_ids, val_ids, train_file, val_file
        if logger is not None:
            logger.warning("[WARN] Existing split files invalid for current dataset. Regenerating splits.")

    uniq_ids = sorted(set(image_ids))
    rng = random.Random(seed)
    rng.shuffle(uniq_ids)

    split_idx = int(len(uniq_ids) * split_ratio)
    split_idx = max(1, min(len(uniq_ids) - 1, split_idx))

    train_ids = uniq_ids[:split_idx]
    val_ids = uniq_ids[split_idx:]

    _save_split(train_file, train_ids)
    _save_split(val_file, val_ids)

    if logger is not None:
        logger.info(f"[INFO] Created split files: {train_file}, {val_file}")

    return train_ids, val_ids, train_file, val_file


def _make_transforms(cfg: Dict[str, Any], train: bool) -> DetectionTransform:
    inp = cfg["input"]
    aug = cfg.get("augmentation", {})
    resize_policy = str(inp.get("resize_policy", "letterbox")).lower()
    if resize_policy not in {"letterbox", "resize"}:
        raise ValueError(
            f"Unsupported input.resize_policy: {resize_policy}. "
            "Supported values: ['letterbox', 'resize']."
        )

    return DetectionTransform(
        train=train,
        out_w=int(inp["width"]),
        out_h=int(inp["height"]),
        color_mode=str(inp["color_mode"]).lower(),
        resize_policy=resize_policy,
        interpolation=str(inp.get("interpolation", "bilinear")),
        letterbox_fill_value=int(inp.get("letterbox_fill_value", 114)),
        horizontal_flip=bool(aug.get("horizontal_flip", False)),
        hflip_prob=float(aug.get("hflip_prob", 0.5)),
        color_jitter=bool(aug.get("color_jitter", False)),
        brightness=float(aug.get("brightness", 0.2)),
        contrast=float(aug.get("contrast", 0.2)),
        saturation=float(aug.get("saturation", 0.2)),
        blur=bool(aug.get("blur", False)),
        blur_prob=float(aug.get("blur_prob", 0.2)),
        blur_radius=float(aug.get("blur_radius", 0.8)),
    )


def build_spl_ball_datasets(cfg: Dict[str, Any], logger):
    class_name = str(cfg["dataset"]["class_name"]) # ball
    images_dir = Path(cfg["paths"]["images_dir"])
    bbox_format = str(cfg["dataset"].get("bbox_format", "cxcywh_radius")).lower()
    duplicate_policy = str(cfg["dataset"].get("duplicate_policy", "last")).lower()

    records, stats = load_spl_ball_records(
        images_dir=images_dir,
        class_name=class_name,
        bbox_format=bbox_format,
        duplicate_policy=duplicate_policy,
        logger=logger,
    )

    splits_dir = Path(cfg["paths"]["splits_dir"])
    train_ids, val_ids, train_file, val_file = create_or_load_splits(
        image_ids=[r.image_id for r in records],
        splits_dir=splits_dir,
        split_ratio=float(cfg["dataset"]["train_val_split"]),
        seed=int(cfg["dataset"]["seed"]),
        reuse_splits=bool(cfg["dataset"].get("reuse_splits", True)),
        logger=logger,
    )

    train_set_ids = set(train_ids)
    val_set_ids = set(val_ids)

    train_records = [r for r in records if r.image_id in train_set_ids]
    val_records = [r for r in records if r.image_id in val_set_ids]

    if not train_records:
        raise RuntimeError("Train split is empty")
    if not val_records:
        raise RuntimeError("Val split is empty")

    train_tf = _make_transforms(cfg, train=True)
    val_tf = _make_transforms(cfg, train=False)

    train_ds = SPLBallDetectionDataset(records=train_records, transform=train_tf, class_name=class_name)
    val_ds = SPLBallDetectionDataset(records=val_records, transform=val_tf, class_name=class_name)

    info = {
        **stats,
        "bbox_format": bbox_format,
        "duplicate_policy": duplicate_policy,
        "train_images": len(train_records),
        "val_images": len(val_records),
        "train_split_file": str(train_file),
        "val_split_file": str(val_file),
    }
    return train_ds, val_ds, info


def export_yolo_labels(records: List[SampleRecord], out_dir: Path, class_id: int = 0) -> None:
    """Optional utility: export labels to YOLO txt format.

    Each row is:
      <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        orig_h, orig_w = rec.orig_size
        label_path = out_dir / (Path(rec.image_id).with_suffix(".txt").as_posix())
        label_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for box in rec.boxes_orig:
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            cx = x1 + 0.5 * bw
            cy = y1 + 0.5 * bh
            cx_n = cx / max(float(orig_w), 1.0)
            cy_n = cy / max(float(orig_h), 1.0)
            bw_n = bw / max(float(orig_w), 1.0)
            bh_n = bh / max(float(orig_h), 1.0)
            lines.append(f"{class_id} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

        with label_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def build_dataloaders(
    cfg: Dict[str, Any],
    logger,
    num_workers_override: int | None = None,
):
    train_ds, val_ds, info = build_spl_ball_datasets(cfg, logger)

    bs = int(cfg["train"]["batch_size"])
    num_workers = int(num_workers_override if num_workers_override is not None else cfg["train"]["num_workers"])
    seed = int(cfg["dataset"]["seed"])
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=detection_collate_fn,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=detection_collate_fn,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader, info
