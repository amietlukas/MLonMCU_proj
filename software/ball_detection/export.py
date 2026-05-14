from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    software_root = Path(__file__).resolve().parent.parent
    if str(software_root) not in sys.path:
        sys.path.insert(0, str(software_root))

from ball_detection.src.config import load_config
from ball_detection.src.datasets.spl_ball_dataset import build_dataloaders
from ball_detection.src.export.export_onnx import export_fp32_onnx
from ball_detection.src.export.quantize_onnx import quantize_int8_qdq
from ball_detection.src.logging_utils import setup_logger
from ball_detection.src.models import build_model
from ball_detection.src.reproducibility import resolve_device
from ball_detection.src.training.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ball detector to ONNX and INT8 QDQ")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", type=str, default=None, help="Override export output directory")
    p.add_argument("--num-workers", type=int, default=None)
    split_group = p.add_mutually_exclusive_group()
    split_group.add_argument(
        "--reuse-splits",
        dest="reuse_splits",
        action="store_true",
        help="Reuse split files if valid for the current dataset",
    )
    split_group.add_argument(
        "--regenerate-splits",
        dest="reuse_splits",
        action="store_false",
        help="Force regeneration of split files",
    )
    p.set_defaults(reuse_splits=None)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.reuse_splits is not None:
        cfg["dataset"]["reuse_splits"] = bool(args.reuse_splits)
    device = resolve_device(args.device)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    default_out = Path(args.out_dir).resolve() if args.out_dir else (ckpt_path.parent.parent / "exports")
    default_out.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(default_out / "export.log", debug=args.debug)

    model = build_model(cfg).to(device)
    load_checkpoint(ckpt_path, model=model, map_location=device)

    train_loader, val_loader, _info = build_dataloaders(cfg, logger=logger, num_workers_override=args.num_workers)
    sample_batch, _sample_targets = next(iter(train_loader))

    export_cfg = cfg["export"]
    fp32_path = default_out / "fp32.onnx"
    int8_path = default_out / "int8_ptq_qdq.onnx"

    if bool(export_cfg.get("do_fp32_onnx", True)):
        export_report = export_fp32_onnx(
            model=model,
            out_path=fp32_path,
            input_shape=(
                1,
                int(cfg["input"]["channels"]),
                int(cfg["input"]["height"]),
                int(cfg["input"]["width"]),
            ),
            opset=int(export_cfg.get("opset", 13)),
            dynamic_batch=bool(export_cfg.get("dynamic_batch", False)),
            logger=logger,
            sample_batch=sample_batch,
        )
    else:
        export_report = {"exported": False}

    if bool(export_cfg.get("do_int8_ptq_qdq", True)) and bool(export_report.get("exported", False)) and fp32_path.exists():
        int8_cfg = export_cfg.get("int8", {})
        quantize_int8_qdq(
            fp32_onnx_path=fp32_path,
            int8_onnx_path=int8_path,
            calibration_loader=val_loader,
            calibration_batches=int(export_cfg.get("calibration_batches", 20)),
            batch_size=int(cfg["train"]["batch_size"]),
            per_channel=bool(int8_cfg.get("per_channel", True)),
            activation_type=str(int8_cfg.get("activation_type", "uint8")),
            weight_type=str(int8_cfg.get("weight_type", "int8")),
            calibrate_method=str(int8_cfg.get("calibrate_method", "minmax")),
            report_path=default_out / "calibration_report.json",
            cfg=cfg,
            logger=logger,
        )
    elif bool(export_cfg.get("do_int8_ptq_qdq", True)):
        logger.warning("[WARN] Skipping INT8 export because FP32 ONNX export was not successful.")


if __name__ == "__main__":
    main()
