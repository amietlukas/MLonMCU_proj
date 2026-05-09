"""
File dedicated for all checkpointing related stuff.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)


# ===== CHECKPOINTING during training =====

def save_checkpoint(
    path: str | Path,
    *,
    run_id: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    cfg: Dict[str, Any],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # convert Path objects in cfg so torch.save doesn't choke
    import copy
    def _sanitize(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    payload = {
        "run_id": run_id,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": _sanitize(copy.deepcopy(cfg)),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, path)


# ===== POSTTRAINING export inkl. quantization =====
def export_best_model_fp32_and_int8_qdq(
    *,
    model: nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
    out_dir: str | Path,
    run_id: str,
    best_epoch: int,
    calibration_loader,
    calibration_batches: int = 32,
) -> Dict[str, Path]:
    """
    Export best model as:
      1) FP32 ONNX
      2) INT8 QDQ ONNX (static quantization)
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- FUSE INPUT NORMALIZATION ---
    # We bake the full inference preprocessing directly into the first Convolution:
    # both the /255 (uint8 -> [0,1]) and the (x - mu) / sigma steps are folded in.
    # The fused graph (FP32 and INT8 QDQ) expects float inputs in the raw [0, 255]
    # pixel range. The INT8 input QuantizeLinear's scale is therefore calibrated for
    # [0, 255], so on the MCU you can feed uint8 pixels directly (mapped to int8 via
    # a zero-point shift) without any divide or normalize at runtime.
    # Math: combined effective normalization is (pixel/255 - mu) / sigma
    #     = (pixel - 255*mu) / (255*sigma), so we use mu' = 255*mu, sigma' = 255*sigma.
    in_ch = int(cfg.get("model", {}).get("in_channels", 1))
    norm_cfg = cfg.get("preprocess", {}).get("normalize", {})

    # Parse norm_mean and norm_std, then scale by 255 to absorb the uint8->[0,1] step.
    def _parse(v, n):
        if isinstance(v, (list, tuple)): return [float(x) for x in v] if len(v) == n else [float(v[0])] * n
        return [float(v)] * n
    norm_mean = [m * 255.0 for m in _parse(norm_cfg.get("mean", 0.5), in_ch)]
    norm_std  = [s * 255.0 for s in _parse(norm_cfg.get("std", 0.5),  in_ch)]
    
    with torch.no_grad():
        conv = model.features[0][0]
        bn = model.features[0][1]
        
        mu = torch.tensor(norm_mean, dtype=torch.float32, device=conv.weight.device).view(1, -1, 1, 1)
        sigma = torch.tensor(norm_std, dtype=torch.float32, device=conv.weight.device).view(1, -1, 1, 1)
        
        # W' = W / sigma
        new_weight = conv.weight / sigma
        # Delta = sum(W' * mu) over in_ch, h, w
        delta = (new_weight * mu).sum(dim=(1, 2, 3))
        
        conv.weight.copy_(new_weight)
        bn.running_mean.add_(delta)

    # Fixed input shape
    size_cfg = cfg.get("data", {}).get("input_size", 128)
    if isinstance(size_cfg, (list, tuple)):
        h, w = size_cfg
    else:
        h = w = int(size_cfg)
    dummy_input = torch.randn(1, in_ch, h, w, device=device)

    model.eval()

    # nameing
    fp32_path = out_dir / f"{run_id}_best-epoch{best_epoch}_fp32_op13.onnx"
    int8_qdq_path = out_dir / f"{run_id}_best-epoch{best_epoch}_int8_qdq_op13.onnx"

    # 1) FP32 ONNX export
    torch.onnx.export(
        model,
        dummy_input,
        str(fp32_path),
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,  # fixed-shape export
        #opset_version=13,   # Use 13 for ST, if onnx doesnt work
        opset_version=17,   # Fallback from 18 to 17 avoiding the node-assert bug
        verbose=False,
    )

    # Data reader for ONNX Runtime static calibration
    class _TorchCalibrationDataReader(CalibrationDataReader):
        def __init__(self, loader, input_name: str, max_batches: int):
            self._samples = []
            n = 0
            
            # --- Unnormalize inputs for the calibrated graph ---
            # Loader emits xb = (pixel/255 - mu) / sigma. With mu' = 255*mu and
            # sigma' = 255*sigma below, xb * sigma' + mu' = raw pixel in [0, 255].
            mu_t = torch.tensor(norm_mean).view(1, -1, 1, 1)
            sig_t = torch.tensor(norm_std).view(1, -1, 1, 1)
            
            for xb, _yb, _paths in loader:
                if n >= max_batches:
                    break
                
                xb = xb * sig_t + mu_t
                
                # Model is exported with fixed batch_size=1, so we split the batch.
                for i in range(xb.shape[0]):
                    self._samples.append({input_name: xb[i:i+1].detach().cpu().numpy().astype(np.float32)})
                n += 1
            self._iter = iter(self._samples)

        # feed one item at a time to quantizer
        def get_next(self):
            return next(self._iter, None)

        def rewind(self):
            self._iter = iter(self._samples)

    # Use actual ONNX input name to avoid name mismatch.
    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    reader = _TorchCalibrationDataReader(
        loader=calibration_loader,
        input_name=input_name,
        max_batches=calibration_batches,
    )

    # 2) INT8 QDQ ONNX export (static quantization)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_qdq_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=True,
        reduce_range=False,
    )

    def _fmt_size(path: Path) -> str:
        size = path.stat().st_size
        if size < 1024:
            return f"{size} B"
        if size < 1024 ** 2:
            return f"{size / 1024:.2f} KB"
        return f"{size / (1024 ** 2):.2f} MB"

    print("[INFO] ONNX export artifacts")
    print(f"  FP32 ONNX     : {_fmt_size(fp32_path)}")
    print(f"  INT8 QDQ ONNX : {_fmt_size(int8_qdq_path)}")
    print()

    return {
        "fp32": fp32_path,
        "int8_qdq": int8_qdq_path,
    }