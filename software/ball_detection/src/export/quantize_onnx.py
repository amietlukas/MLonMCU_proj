from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ball_detection.src.config import config_to_serializable



def _try_import_ort_quant():
    try:
        from onnxruntime.quantization import (  # type: ignore
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )

        return CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    except Exception:
        return None


class TensorCalibrationDataReader:
    def __init__(self, samples: list[np.ndarray], input_name: str):
        self.samples = samples
        self.input_name = input_name
        self._idx = 0

    def get_next(self):
        if self._idx >= len(self.samples):
            return None
        item = {self.input_name: self.samples[self._idx]}
        self._idx += 1
        return item

    def rewind(self):
        self._idx = 0


def _collect_calibration_samples(loader, max_images: int) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    for images, _targets in loader:
        for i in range(images.shape[0]):
            x = images[i : i + 1].detach().cpu().numpy().astype(np.float32)
            samples.append(x)
            if len(samples) >= max_images:
                return samples
    return samples


def quantize_int8_qdq(
    *,
    fp32_onnx_path: Path,
    int8_onnx_path: Path,
    calibration_loader,
    calibration_batches: int,
    batch_size: int,
    per_channel: bool,
    activation_type: str,
    weight_type: str,
    calibrate_method: str,
    report_path: Path,
    cfg: dict[str, Any],
    logger,
) -> bool:
    quant_mod = _try_import_ort_quant()
    if quant_mod is None:
        logger.warning(
            "[WARN] onnxruntime quantization APIs unavailable. "
            "Install onnxruntime to enable INT8 PTQ QDQ export."
        )
        return False

    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static = quant_mod

    try:
        import onnxruntime as ort
    except Exception:
        logger.warning("[WARN] onnxruntime not installed; cannot run quantize_static.")
        return False

    max_images = max(1, int(calibration_batches) * max(1, int(batch_size)))
    samples = _collect_calibration_samples(calibration_loader, max_images=max_images)
    if len(samples) == 0:
        logger.warning("[WARN] No calibration samples available; skipping INT8 PTQ QDQ export.")
        return False

    sess = ort.InferenceSession(str(fp32_onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    class ORTReader(CalibrationDataReader):
        def __init__(self, base_reader: TensorCalibrationDataReader):
            self.base_reader = base_reader

        def get_next(self):
            return self.base_reader.get_next()

        def rewind(self):
            self.base_reader.rewind()

    reader = ORTReader(TensorCalibrationDataReader(samples=samples, input_name=input_name))

    act_type_map = {
        "uint8": QuantType.QUInt8,
        "int8": QuantType.QInt8,
    }
    wt_type_map = {
        "uint8": QuantType.QUInt8,
        "int8": QuantType.QInt8,
    }
    method_map = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }

    act_qtype = act_type_map.get(str(activation_type).lower(), QuantType.QInt8)
    wt_qtype = wt_type_map.get(str(weight_type).lower(), QuantType.QInt8)
    cal_method = method_map.get(str(calibrate_method).lower(), CalibrationMethod.MinMax)

    int8_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=act_qtype,
        weight_type=wt_qtype,
        per_channel=bool(per_channel),
        calibrate_method=cal_method,
    )

    report = {
        "num_calibration_images": len(samples),
        "fp32_onnx": str(fp32_onnx_path),
        "int8_qdq_onnx": str(int8_onnx_path),
        "quantization": {
            "quant_format": "QDQ",
            "per_channel": bool(per_channel),
            "activation_type": str(activation_type),
            "weight_type": str(weight_type),
            "calibrate_method": str(calibrate_method),
            "calibration_batches": int(calibration_batches),
        },
        "config_snapshot": config_to_serializable(cfg),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"[INFO] Exported INT8 QDQ ONNX: {int8_onnx_path}")
    logger.info(f"[INFO] Saved calibration report: {report_path}")
    return True
