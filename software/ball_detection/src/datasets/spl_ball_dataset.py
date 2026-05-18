from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from ball_detection.src.datasets.transforms_detection import DetectionTransform
from ball_detection.src.reproducibility import worker_init_fn
from ball_detection.src.utils.boxes import xywh_to_xyxy, xyxy_to_xywh_topleft


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class DatasetSource:
    name: str
    images_dir: Path
    annotation_file: str
    annotation_format: str
    bbox_format: str
    class_name: str
    duplicate_policy: str


@dataclass
class SampleRecord:
    image_id: str
    image_path: Path
    orig_size: tuple[int, int]  # (H, W)
    boxes_orig: torch.Tensor  # [N,4] xyxy absolute in original image
    source_name: str


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

        boxes_resized = boxes_resized.float()
        boxes_resized_xywh_topleft = xyxy_to_xywh_topleft(boxes_resized)
        boxes_orig = rec.boxes_orig.clone().float()
        boxes_orig_xywh_topleft = xyxy_to_xywh_topleft(boxes_orig)

        labels = torch.zeros((boxes_resized.shape[0],), dtype=torch.long)
        target = {
            "boxes": boxes_resized,
            "boxes_xywh_topleft": boxes_resized_xywh_topleft,
            "labels": labels,
            "image_id": rec.image_id,
            "source": rec.source_name,
            "orig_size": rec.orig_size,
            "resized_size": meta.resized_size,
            "resize_mode": meta.mode,
            "scale_x": float(meta.scale_x),
            "scale_y": float(meta.scale_y),
            "scale": float(meta.scale_x),  # legacy alias for square letterbox workflows
            "pad": (meta.pad_left, meta.pad_top, meta.pad_right, meta.pad_bottom),
            "boxes_orig": boxes_orig,
            "boxes_orig_xywh_topleft": boxes_orig_xywh_topleft,
            "image_path": str(rec.image_path),
        }
        return img_tensor, target


class MixedSourceBatchSampler(Sampler[list[int]]):
    """Builds mixed-source batches with guaranteed per-batch source presence.

    This sampler is intended for training only. It oversamples smaller sources by
    cycling through shuffled indices so that every batch contains all sources.
    """

    def __init__(
        self,
        source_to_indices: Dict[str, List[int]],
        batch_size: int,
        seed: int,
        shuffle: bool = True,
    ) -> None:
        if not source_to_indices:
            raise ValueError("source_to_indices must not be empty")

        self.source_to_indices = {k: list(v) for k, v in source_to_indices.items() if len(v) > 0}
        self.sources = sorted(self.source_to_indices.keys())
        if not self.sources:
            raise ValueError("All sources are empty")

        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._epoch = 0

        if self.batch_size < len(self.sources):
            raise ValueError(
                f"batch_size={self.batch_size} is smaller than num_sources={len(self.sources)}. "
                "Cannot guarantee mixed-source batches."
            )

        self.per_source_batch_count = self._build_per_source_batch_count()
        self.num_batches = self._compute_num_batches()

    def _build_per_source_batch_count(self) -> Dict[str, int]:
        base = self.batch_size // len(self.sources)
        rem = self.batch_size % len(self.sources)
        counts: dict[str, int] = {}
        for i, src in enumerate(self.sources):
            counts[src] = base + (1 if i < rem else 0)
        return counts

    def _compute_num_batches(self) -> int:
        batches = []
        for src in self.sources:
            need = max(self.per_source_batch_count[src], 1)
            n = len(self.source_to_indices[src])
            batches.append((n + need - 1) // need)
        return max(batches) if batches else 0

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        if self.num_batches <= 0:
            return

        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        pools: dict[str, List[int]] = {}
        pos: dict[str, int] = {}
        for src in self.sources:
            arr = list(self.source_to_indices[src])
            if self.shuffle:
                rng.shuffle(arr)
            pools[src] = arr
            pos[src] = 0

        for _ in range(self.num_batches):
            batch: list[int] = []
            for src in self.sources:
                take = self.per_source_batch_count[src]
                arr = pools[src]
                if not arr:
                    continue

                p = pos[src]
                for _k in range(take):
                    if p >= len(arr):
                        p = 0
                        if self.shuffle:
                            rng.shuffle(arr)
                    batch.append(arr[p])
                    p += 1
                pos[src] = p

            if self.shuffle:
                rng.shuffle(batch)
            yield batch



def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _convert_box_to_xyxy(
    x: float,
    y: float,
    w: float,
    h: float,
    bbox_format: str,
) -> tuple[float, float, float, float]:
    if bbox_format == "xywh_topleft":
        return xywh_to_xyxy(x, y, w, h)
    if bbox_format == "cxcywh_center":
        x1 = float(x) - float(w) * 0.5
        y1 = float(y) - float(h) * 0.5
        x2 = float(x) + float(w) * 0.5
        y2 = float(y) + float(h) * 0.5
        return x1, y1, x2, y2
    if bbox_format == "cxcywh_radius":
        x1 = float(x) - float(w)
        y1 = float(y) - float(h)
        x2 = float(x) + float(w)
        y2 = float(y) + float(h)
        return x1, y1, x2, y2
    raise ValueError(f"Unsupported bbox_format: {bbox_format}")


def _clamp_and_validate_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float] | None:
    if x2 <= x1 or y2 <= y1:
        return None

    x1 = max(0.0, min(float(img_w - 1), x1))
    y1 = max(0.0, min(float(img_h - 1), y1))
    x2 = max(0.0, min(float(img_w - 1), x2))
    y2 = max(0.0, min(float(img_h - 1), y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _parse_legacy_multiclass_row_for_ball(
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

        x1, y1, x2, y2 = _convert_box_to_xyxy(float(x), float(y), float(w), float(h), bbox_format)
        box = _clamp_and_validate_box(x1, y1, x2, y2, img_w=img_w, img_h=img_h)
        if box is None:
            if logger is not None:
                logger.warning(f"[WARN] invalid box skipped: {context} -> {(x1, y1, x2, y2)}")
            continue

        boxes.append([box[0], box[1], box[2], box[3]])

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def _is_header_like_simple_row(row: List[str]) -> bool:
    if len(row) < 5:
        return False
    first = row[0].strip().lower()
    if first in {"filename", "image", "image_name"}:
        return True
    second = _safe_float(row[1])
    third = _safe_float(row[2])
    fourth = _safe_float(row[3])
    fifth = _safe_float(row[4])
    return second is None and third is None and fourth is None and fifth is None


def _parse_simple_xywh_row_for_ball(
    row: List[str],
    img_w: int,
    img_h: int,
    bbox_format: str,
    logger,
    context: str,
) -> torch.Tensor:
    if len(row) < 5:
        if logger is not None:
            logger.warning(f"[WARN] malformed simple annotation row skipped: {context} | row={row}")
        return torch.zeros((0, 4), dtype=torch.float32)

    x = _safe_float(row[1])
    y = _safe_float(row[2])
    w = _safe_float(row[3])
    h = _safe_float(row[4])

    if None in {x, y, w, h}:
        if _is_header_like_simple_row(row):
            return torch.zeros((0, 4), dtype=torch.float32)
        if logger is not None:
            logger.warning(f"[WARN] invalid numeric simple box skipped: {context} | row={row}")
        return torch.zeros((0, 4), dtype=torch.float32)

    x1, y1, x2, y2 = _convert_box_to_xyxy(float(x), float(y), float(w), float(h), bbox_format)
    box = _clamp_and_validate_box(x1, y1, x2, y2, img_w=img_w, img_h=img_h)
    if box is None:
        if logger is not None:
            logger.warning(f"[WARN] invalid simple box skipped: {context} -> {(x1, y1, x2, y2)}")
        return torch.zeros((0, 4), dtype=torch.float32)

    return torch.tensor([[box[0], box[1], box[2], box[3]]], dtype=torch.float32)


def _find_annotation_files(images_dir: Path, annotation_file: str) -> List[Path]:
    pattern = annotation_file.strip()
    if not pattern:
        pattern = "annotations.csv"
    return sorted(images_dir.rglob(pattern))


def _find_image_files(images_dir: Path) -> List[Path]:
    files: list[Path] = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def _resolve_dataset_sources(cfg: Dict[str, Any]) -> List[DatasetSource]:
    dataset_cfg = cfg["dataset"]
    paths_cfg = cfg["paths"]

    dataset_root = Path(paths_cfg["dataset_root"])

    sources_cfg = dataset_cfg.get("sources", None)
    if isinstance(sources_cfg, list) and len(sources_cfg) > 0:
        sources: list[DatasetSource] = []
        used_names: set[str] = set()

        for idx, src in enumerate(sources_cfg, start=1):
            if not isinstance(src, dict):
                raise ValueError(f"dataset.sources[{idx - 1}] must be a dictionary")

            name_raw = str(src.get("name", f"source_{idx}")).strip()
            name = name_raw.lower().replace(" ", "_")
            if name in used_names:
                raise ValueError(f"Duplicate dataset source name: {name}")
            used_names.add(name)

            rel_or_abs_dir = src.get("images_dir", src.get("path", None))
            if rel_or_abs_dir is None:
                raise ValueError(
                    f"dataset.sources[{idx - 1}] must define 'images_dir' (or 'path')"
                )
            src_dir = Path(str(rel_or_abs_dir))
            if not src_dir.is_absolute():
                src_dir = (dataset_root / src_dir).resolve()

            annotation_format = str(src.get("annotation_format", "csv_xywh_topleft")).lower()
            default_file = "*.csv" if annotation_format == "legacy_multiclass" else "annotations.csv"
            annotation_file = str(src.get("annotation_file", default_file))
            bbox_format = str(src.get("bbox_format", dataset_cfg.get("bbox_format", "xywh_topleft"))).lower()
            class_name = str(src.get("class_name", dataset_cfg.get("class_name", "Ball")))
            duplicate_policy = str(src.get("duplicate_policy", dataset_cfg.get("duplicate_policy", "append"))).lower()

            sources.append(
                DatasetSource(
                    name=name,
                    images_dir=src_dir,
                    annotation_file=annotation_file,
                    annotation_format=annotation_format,
                    bbox_format=bbox_format,
                    class_name=class_name,
                    duplicate_policy=duplicate_policy,
                )
            )

        return sources

    # Backward-compat: single legacy source driven by paths.images_dir.
    return [
        DatasetSource(
            name="spl",
            images_dir=Path(paths_cfg["images_dir"]),
            annotation_file="*.csv",
            annotation_format="legacy_multiclass",
            bbox_format=str(dataset_cfg.get("bbox_format", "cxcywh_radius")).lower(),
            class_name=str(dataset_cfg.get("class_name", "Ball")),
            duplicate_policy=str(dataset_cfg.get("duplicate_policy", "last")).lower(),
        )
    ]


def _load_records_from_source(
    source: DatasetSource,
    logger,
) -> tuple[List[SampleRecord], Dict[str, Any]]:
    if not source.images_dir.exists():
        raise FileNotFoundError(f"Dataset source directory does not exist: {source.images_dir}")

    image_files = _find_image_files(source.images_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found under: {source.images_dir}")

    csv_files = _find_annotation_files(source.images_dir, source.annotation_file)
    if not csv_files:
        raise FileNotFoundError(
            f"No annotation files matching '{source.annotation_file}' found under: {source.images_dir}"
        )

    records_map: dict[str, dict[str, Any]] = {}
    for image_path in image_files:
        rel_image = image_path.relative_to(source.images_dir).as_posix()
        image_id = f"{source.name}/{rel_image}"
        with Image.open(image_path) as img:
            img_w, img_h = img.size
        records_map[image_id] = {
            "image_path": image_path,
            "orig_size": (img_h, img_w),
            "boxes_list": [],
            "has_annotation_row": False,
        }

    total_rows = 0
    rows_with_ball = 0
    total_ball_boxes_rows = 0
    duplicate_rows = 0
    duplicate_row_overwrites = 0
    missing_image_rows = 0

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader, start=1):
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
                    if _is_header_like_simple_row(row):
                        continue
                    missing_image_rows += 1
                    if logger is not None:
                        logger.warning(f"[WARN] image missing or unsupported extension skipped: {image_path}")
                    continue

                total_rows += 1

                rel_image = image_path.relative_to(source.images_dir).as_posix()
                image_id = f"{source.name}/{rel_image}"

                entry = records_map.get(image_id)
                if entry is None:
                    missing_image_rows += 1
                    if logger is not None:
                        logger.warning(f"[WARN] annotation row references image outside source map: {image_path}")
                    continue

                orig_h, orig_w = entry["orig_size"]
                img_w = int(orig_w)
                img_h = int(orig_h)
                context = f"{csv_path.name}:{row_idx}:{image_name}"
                if source.annotation_format == "legacy_multiclass":
                    boxes_orig = _parse_legacy_multiclass_row_for_ball(
                        row=row,
                        class_name=source.class_name,
                        img_w=img_w,
                        img_h=img_h,
                        bbox_format=source.bbox_format,
                        logger=logger,
                        context=context,
                    )
                elif source.annotation_format in {"csv_xywh_topleft", "simple_csv_xywh"}:
                    boxes_orig = _parse_simple_xywh_row_for_ball(
                        row=row,
                        img_w=img_w,
                        img_h=img_h,
                        bbox_format=source.bbox_format,
                        logger=logger,
                        context=context,
                    )
                else:
                    raise ValueError(
                        f"Unsupported annotation format '{source.annotation_format}' for source '{source.name}'. "
                        "Supported values: ['legacy_multiclass', 'csv_xywh_topleft', 'simple_csv_xywh']."
                    )

                prior_has_annotation_row = bool(entry.get("has_annotation_row", False))
                entry["has_annotation_row"] = True
                if boxes_orig.shape[0] > 0:
                    rows_with_ball += 1
                total_ball_boxes_rows += int(boxes_orig.shape[0])

                if not prior_has_annotation_row:
                    if boxes_orig.shape[0] > 0:
                        entry["boxes_list"] = [boxes_orig]
                    continue

                duplicate_rows += 1
                policy = source.duplicate_policy
                if policy == "append":
                    if boxes_orig.shape[0] > 0:
                        entry["boxes_list"].append(boxes_orig)
                elif policy == "first":
                    continue
                elif policy == "last":
                    duplicate_row_overwrites += 1
                    entry["boxes_list"] = [boxes_orig] if boxes_orig.shape[0] > 0 else []
                else:
                    raise ValueError(
                        f"Unsupported duplicate_policy '{policy}' for source '{source.name}'. "
                        "Expected one of ['append', 'first', 'last']."
                    )

    records: list[SampleRecord] = []
    images_with_annotation_rows = 0
    for image_id in sorted(records_map.keys()):
        payload = records_map[image_id]
        if bool(payload.get("has_annotation_row", False)):
            images_with_annotation_rows += 1
        boxes_list: list[torch.Tensor] = payload["boxes_list"]
        if boxes_list:
            boxes_orig = torch.cat(boxes_list, dim=0)
        else:
            boxes_orig = torch.zeros((0, 4), dtype=torch.float32)

        records.append(
            SampleRecord(
                image_id=image_id,
                image_path=payload["image_path"],
                orig_size=payload["orig_size"],
                boxes_orig=boxes_orig,
                source_name=source.name,
            )
        )

    if not records:
        raise RuntimeError(f"No records parsed for source '{source.name}'")

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

    stats = {
        "source_name": source.name,
        "num_csv_files": len(csv_files),
        "num_images": len(records),
        "num_rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "duplicate_row_overwrites": duplicate_row_overwrites,
        "missing_image_rows": missing_image_rows,
        "rows_with_ball": rows_with_ball,
        "rows_without_ball": total_rows - rows_with_ball,
        "images_with_annotation_rows": images_with_annotation_rows,
        "images_without_annotation_rows": len(records) - images_with_annotation_rows,
        "images_with_ball": images_with_ball,
        "images_without_ball": len(records) - images_with_ball,
        "num_ball_boxes_rows": total_ball_boxes_rows,
        "num_ball_boxes": total_ball_boxes,
        "bbox_width_min": min(bbox_ws) if bbox_ws else 0.0,
        "bbox_width_mean": (sum(bbox_ws) / len(bbox_ws)) if bbox_ws else 0.0,
        "bbox_width_max": max(bbox_ws) if bbox_ws else 0.0,
        "bbox_height_min": min(bbox_hs) if bbox_hs else 0.0,
        "bbox_height_mean": (sum(bbox_hs) / len(bbox_hs)) if bbox_hs else 0.0,
        "bbox_height_max": max(bbox_hs) if bbox_hs else 0.0,
        "bbox_format": source.bbox_format,
        "duplicate_policy": source.duplicate_policy,
    }

    return records, stats


def load_spl_ball_records(
    sources: List[DatasetSource],
    logger,
) -> tuple[List[SampleRecord], Dict[str, Any]]:
    all_records: list[SampleRecord] = []
    source_stats: dict[str, Dict[str, Any]] = {}

    seen_ids: set[str] = set()
    for source in sources:
        records, stats = _load_records_from_source(source=source, logger=logger)
        for rec in records:
            if rec.image_id in seen_ids:
                raise RuntimeError(f"Duplicate image_id across sources: {rec.image_id}")
            seen_ids.add(rec.image_id)
        all_records.extend(records)
        source_stats[source.name] = stats

    if not all_records:
        raise RuntimeError("No records parsed from configured sources")

    bbox_ws: list[float] = []
    bbox_hs: list[float] = []
    images_with_ball = 0
    total_ball_boxes = 0
    for rec in all_records:
        if rec.boxes_orig.shape[0] > 0:
            images_with_ball += 1
        total_ball_boxes += int(rec.boxes_orig.shape[0])
        if rec.boxes_orig.numel() > 0:
            wh = (rec.boxes_orig[:, 2:4] - rec.boxes_orig[:, 0:2]).clamp(min=0)
            bbox_ws.extend(wh[:, 0].tolist())
            bbox_hs.extend(wh[:, 1].tolist())

    stats = {
        "num_sources": len(sources),
        "source_names": [s.name for s in sources],
        "source_stats": source_stats,
        "num_csv_files": int(sum(s["num_csv_files"] for s in source_stats.values())),
        "num_images": len(all_records),
        "num_rows": int(sum(s["num_rows"] for s in source_stats.values())),
        "duplicate_rows": int(sum(s["duplicate_rows"] for s in source_stats.values())),
        "duplicate_row_overwrites": int(sum(s["duplicate_row_overwrites"] for s in source_stats.values())),
        "missing_image_rows": int(sum(s.get("missing_image_rows", 0) for s in source_stats.values())),
        "rows_with_ball": int(sum(s["rows_with_ball"] for s in source_stats.values())),
        "rows_without_ball": int(sum(s["rows_without_ball"] for s in source_stats.values())),
        "images_with_annotation_rows": int(sum(s.get("images_with_annotation_rows", 0) for s in source_stats.values())),
        "images_without_annotation_rows": int(sum(s.get("images_without_annotation_rows", 0) for s in source_stats.values())),
        "images_with_ball": images_with_ball,
        "images_without_ball": len(all_records) - images_with_ball,
        "num_ball_boxes_rows": int(sum(s["num_ball_boxes_rows"] for s in source_stats.values())),
        "num_ball_boxes": total_ball_boxes,
        "bbox_width_min": min(bbox_ws) if bbox_ws else 0.0,
        "bbox_width_mean": (sum(bbox_ws) / len(bbox_ws)) if bbox_ws else 0.0,
        "bbox_width_max": max(bbox_ws) if bbox_ws else 0.0,
        "bbox_height_min": min(bbox_hs) if bbox_hs else 0.0,
        "bbox_height_mean": (sum(bbox_hs) / len(bbox_hs)) if bbox_hs else 0.0,
        "bbox_height_max": max(bbox_hs) if bbox_hs else 0.0,
    }

    return all_records, stats


def _save_split(split_path: Path, image_ids: Iterable[str]) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as f:
        for image_id in image_ids:
            f.write(f"{image_id}\n")


def _load_split(split_path: Path) -> List[str]:
    with split_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def create_or_load_splits(
    records: List[SampleRecord],
    splits_dir: Path,
    split_ratio: float,
    seed: int,
    reuse_splits: bool,
    logger,
) -> tuple[List[str], List[str], Path, Path]:
    train_file = splits_dir / "train.txt"
    val_file = splits_dir / "val.txt"

    image_ids = [r.image_id for r in records]
    id_to_source = {r.image_id: r.source_name for r in records}
    all_sources = set(id_to_source.values())

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
            train_sources = {id_to_source[i] for i in train_ids}
            val_sources = {id_to_source[i] for i in val_ids}
            if train_sources != all_sources or val_sources != all_sources:
                split_valid = False

        if split_valid:
            if logger is not None:
                logger.info(f"[INFO] Reusing existing split files: {train_file}, {val_file}")
            return train_ids, val_ids, train_file, val_file

        if logger is not None:
            logger.warning("[WARN] Existing split files invalid for current dataset/sources. Regenerating splits.")

    source_to_ids: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        source_to_ids[rec.source_name].append(rec.image_id)

    rng = random.Random(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []

    for source_name in sorted(source_to_ids.keys()):
        uniq_ids = sorted(set(source_to_ids[source_name]))
        if len(uniq_ids) < 2:
            raise RuntimeError(
                f"Source '{source_name}' has only {len(uniq_ids)} image(s). "
                "Need at least 2 for train/val split."
            )

        rng.shuffle(uniq_ids)
        split_idx = int(len(uniq_ids) * split_ratio)
        split_idx = max(1, min(len(uniq_ids) - 1, split_idx))

        train_ids.extend(uniq_ids[:split_idx])
        val_ids.extend(uniq_ids[split_idx:])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)

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


def _count_records_by_source(records: List[SampleRecord]) -> Dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        counts[rec.source_name] += 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def build_spl_ball_datasets(cfg: Dict[str, Any], logger):
    dataset_cfg = cfg["dataset"]
    class_name = str(dataset_cfg.get("class_name", "Ball"))

    sources = _resolve_dataset_sources(cfg)
    records, stats = load_spl_ball_records(sources=sources, logger=logger)

    splits_dir = Path(cfg["paths"]["splits_dir"])
    train_ids, val_ids, train_file, val_file = create_or_load_splits(
        records=records,
        splits_dir=splits_dir,
        split_ratio=float(dataset_cfg["train_val_split"]),
        seed=int(dataset_cfg["seed"]),
        reuse_splits=bool(dataset_cfg.get("reuse_splits", True)),
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

    source_formats = {s.name: s.bbox_format for s in sources}
    source_policies = {s.name: s.duplicate_policy for s in sources}

    info = {
        **stats,
        "bbox_format": source_formats,
        "duplicate_policy": source_policies,
        "train_images": len(train_records),
        "val_images": len(val_records),
        "train_split_file": str(train_file),
        "val_split_file": str(val_file),
        "train_source_images": _count_records_by_source(train_records),
        "val_source_images": _count_records_by_source(val_records),
        "all_source_images": _count_records_by_source(records),
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

    mix_sources_per_batch = bool(cfg["dataset"].get("mix_sources_per_batch", True))
    train_source_indices: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(train_ds.records):
        train_source_indices[rec.source_name].append(idx)

    use_mixed_train_sampler = (
        mix_sources_per_batch
        and len(train_source_indices) > 1
        and bs >= len(train_source_indices)
    )

    # Default PyTorch behavior (no persistent_workers, no explicit prefetch_factor). Earlier
    # attempts to enable both stalled the trainer; keep this baseline until we can A/B test
    # them in isolation. Re-introduce via knobs (not unconditionally) when revisited.
    loader_extra: dict = {}

    if use_mixed_train_sampler:
        train_batch_sampler = MixedSourceBatchSampler(
            source_to_indices=dict(train_source_indices),
            batch_size=bs,
            seed=seed,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=detection_collate_fn,
            worker_init_fn=worker_init_fn,
            **loader_extra,
        )
        info["train_batch_mixing"] = {
            "enabled": True,
            "per_source_batch_count": dict(train_batch_sampler.per_source_batch_count),
            "num_batches": len(train_batch_sampler),
            "strategy": "balanced_cycle",
        }
    else:
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
            **loader_extra,
        )
        info["train_batch_mixing"] = {
            "enabled": False,
            "reason": "single_source_or_disabled_or_small_batch",
        }

    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=detection_collate_fn,
        worker_init_fn=worker_init_fn,
        **loader_extra,
    )

    return train_loader, val_loader, info
