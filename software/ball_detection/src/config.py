from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


REQUIRED_TOP_LEVEL_KEYS = [
    "project",
    "paths",
    "dataset",
    "input",
    "model",
    "train",
    "loss",
    "eval",
    "export",
    "augmentation",
]


def _validate_config(cfg: Dict[str, Any]) -> None:
    inp = cfg.get("input", {})
    color_mode = str(inp.get("color_mode", "")).lower()
    channels = int(inp.get("channels", 0))
    resize_policy = str(inp.get("resize_policy", "")).lower()
    interpolation = str(inp.get("interpolation", "bilinear")).lower()
    width = int(inp.get("width", 0))
    height = int(inp.get("height", 0))

    if color_mode not in {"rgb", "grayscale"}:
        raise ValueError("input.color_mode must be one of: rgb, grayscale")

    expected_channels = 3 if color_mode == "rgb" else 1
    if channels != expected_channels:
        raise ValueError(
            f"input.channels={channels} does not match input.color_mode={color_mode} "
            f"(expected {expected_channels})"
        )

    if width <= 0 or height <= 0:
        raise ValueError("input.width and input.height must be positive integers")

    if resize_policy not in {"letterbox", "resize"}:
        raise ValueError("input.resize_policy currently supports only: letterbox, resize")

    if interpolation not in {"nearest", "nn", "bilinear", "linear", "bicubic", "cubic", "lanczos"}:
        raise ValueError(
            "input.interpolation must be one of: nearest, nn, bilinear, linear, bicubic, cubic, lanczos"
        )

    split_ratio = float(cfg.get("dataset", {}).get("train_val_split", 0.8))
    if not (0.0 < split_ratio < 1.0):
        raise ValueError("dataset.train_val_split must be in the open interval (0, 1)")

    num_classes = int(cfg.get("model", {}).get("num_classes", 1))
    if num_classes != 1:
        raise ValueError("This first pipeline version currently supports model.num_classes == 1 only")
    neck_out_ch = int(cfg.get("model", {}).get("neck_out_ch", 96))
    if neck_out_ch <= 0:
        raise ValueError("model.neck_out_ch must be > 0")

    eval_cfg = cfg.get("eval", {})
    max_det = int(eval_cfg.get("max_detections", 1))
    ap_max_det = int(eval_cfg.get("ap_max_detections", max(200, max_det)))
    if max_det <= 0:
        raise ValueError("eval.max_detections must be > 0")
    if ap_max_det <= 0:
        raise ValueError("eval.ap_max_detections must be > 0")

    loss_cfg = cfg.get("loss", {})
    obj_loss_mode = str(loss_cfg.get("obj_loss_mode", "mean")).lower()
    if obj_loss_mode not in {"mean", "balanced"}:
        raise ValueError("loss.obj_loss_mode must be one of: mean, balanced")
    if float(loss_cfg.get("obj_pos_weight", 1.0)) < 0.0:
        raise ValueError("loss.obj_pos_weight must be >= 0")
    if float(loss_cfg.get("obj_neg_weight", 1.0)) < 0.0:
        raise ValueError("loss.obj_neg_weight must be >= 0")
    if float(loss_cfg.get("obj_bce_pos_weight", 1.0)) <= 0.0:
        raise ValueError("loss.obj_bce_pos_weight must be > 0")
    if int(loss_cfg.get("assign_center_radius", 0)) < 0:
        raise ValueError("loss.assign_center_radius must be >= 0")


def _resolve_path(path_value: str | None, *bases: Path) -> Path | None:
    if path_value is None:
        return None
    p = Path(path_value)
    if p.is_absolute():
        return p.resolve()
    for base in bases:
        cand = (base / p).resolve()
        if cand.exists():
            return cand
    return (bases[0] / p).resolve()


def _check_required_keys(cfg: Dict[str, Any]) -> None:
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in cfg:
            raise KeyError(f"Missing required config key: '{key}'")


def load_config(config_path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a YAML dictionary")

    _check_required_keys(cfg)
    _validate_config(cfg)

    project_root = Path(__file__).resolve().parents[1]  # software/ball_detection
    software_root = project_root.parent
    repo_root = software_root.parent
    config_dir = cfg_path.parent

    paths = cfg.setdefault("paths", {})

    dataset_root = _resolve_path(paths.get("dataset_root"), config_dir, software_root, repo_root)
    if dataset_root is None:
        raise ValueError("paths.dataset_root must be set")
    paths["dataset_root"] = dataset_root

    images_dir = _resolve_path(paths.get("images_dir"), config_dir, dataset_root, software_root, repo_root)
    if images_dir is None:
        images_dir = (dataset_root / "SPLBallDataset" / "full_size_images").resolve()
    paths["images_dir"] = images_dir

    annotations_csv = _resolve_path(paths.get("annotations_csv"), config_dir, images_dir, software_root, repo_root)
    paths["annotations_csv"] = annotations_csv

    output_dir = _resolve_path(paths.get("output_dir", "ball_detection/runs"), config_dir, software_root, repo_root)
    export_dir = _resolve_path(paths.get("export_dir", "ball_detection/exports"), config_dir, software_root, repo_root)
    splits_dir = _resolve_path(paths.get("splits_dir", "ball_detection/splits"), config_dir, software_root, repo_root)
    assert output_dir is not None and export_dir is not None and splits_dir is not None
    paths["output_dir"] = output_dir
    paths["export_dir"] = export_dir
    paths["splits_dir"] = splits_dir

    cfg["_meta"] = {
        "config_path": cfg_path,
        "project_root": project_root,
        "software_root": software_root,
        "repo_root": repo_root,
    }

    if not images_dir.exists():
        raise FileNotFoundError(f"Dataset images_dir does not exist: {images_dir}")

    return cfg


def config_to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: config_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [config_to_serializable(v) for v in obj]
    return obj
