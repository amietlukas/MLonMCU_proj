"""Re-quantize an existing fp32.onnx without rebuilding the PyTorch model.

Used for the BallDetector_N6 deployment: the pruned checkpoint can't be
re-loaded into a config-built model (topology mismatch), but the
fp32.onnx exported alongside it already encodes the pruned graph. We
just need to re-run ORT's QDQ quantization with the updated
activation_type from the config (int8 instead of uint8).

Usage:
    python software/ball_detection/requantize_only.py \\
        --config software/ball_detection/configs/ball_styolo_nano_simota.yaml \\
        --fp32   software/ball_detection/runs/.../exports/fp32.onnx \\
        --out    software/ball_detection/runs/.../exports_int8acts/int8_ptq_qdq.onnx
"""
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
from ball_detection.src.export.quantize_onnx import quantize_int8_qdq
from ball_detection.src.logging_utils import setup_logger


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--fp32", required=True, help="path to existing fp32.onnx")
    p.add_argument("--out", required=True, help="output int8 onnx path")
    p.add_argument("--num-workers", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    fp32 = Path(args.fp32).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out.parent / "requantize.log")

    _train_loader, val_loader, _info = build_dataloaders(
        cfg, logger=logger, num_workers_override=args.num_workers,
    )

    int8_cfg = cfg["export"].get("int8", {})
    ok = quantize_int8_qdq(
        fp32_onnx_path=fp32,
        int8_onnx_path=out,
        calibration_loader=val_loader,
        calibration_batches=int(cfg["export"].get("calibration_batches", 20)),
        batch_size=int(cfg["train"]["batch_size"]),
        per_channel=bool(int8_cfg.get("per_channel", True)),
        activation_type=str(int8_cfg.get("activation_type", "int8")),
        weight_type=str(int8_cfg.get("weight_type", "int8")),
        calibrate_method=str(int8_cfg.get("calibrate_method", "minmax")),
        report_path=out.parent / "calibration_report.json",
        cfg=cfg,
        logger=logger,
    )
    if not ok:
        raise SystemExit("quantization failed (see log)")
    print(f"ok  -> {out}")


if __name__ == "__main__":
    main()
