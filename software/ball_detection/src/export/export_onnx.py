from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch



def _try_import_onnx():
    try:
        import onnx

        return onnx
    except Exception:
        return None


def _try_import_onnxruntime():
    try:
        import onnxruntime as ort

        return ort
    except Exception:
        return None


def export_fp32_onnx(
    *,
    model: torch.nn.Module,
    out_path: Path,
    input_shape: tuple[int, int, int, int],
    opset: int,
    dynamic_batch: bool,
    logger,
    sample_batch: torch.Tensor | None = None,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device

    dummy = torch.randn(*input_shape, device=device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "p8": {0: "batch"},
            "p16": {0: "batch"},
            "p32": {0: "batch"},
        }

    report = {
        "onnx_path": str(out_path),
        "exported": False,
        "export_backend": None,
        "requested_opset": int(opset),
        "actual_opset": None,
        "onnx_checked": False,
        "ort_compared": False,
        "output_max_abs_diff": {},
        "output_mean_abs_diff": {},
        "error": None,
    }

    export_kwargs = dict(
        model=model,
        args=(dummy,),
        f=str(out_path),
        export_params=True,
        opset_version=int(opset),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["p8", "p16", "p32"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    prefer_legacy = int(opset) <= 17
    first_backend = "legacy_dynamo_false" if prefer_legacy else "default"
    second_backend = "default" if prefer_legacy else "legacy_dynamo_false"

    def _run_export(backend: str) -> None:
        if backend == "legacy_dynamo_false":
            torch.onnx.export(**export_kwargs, dynamo=False)
        elif backend == "default":
            torch.onnx.export(**export_kwargs)
        else:
            raise ValueError(f"Unsupported exporter backend: {backend}")

    first_error = None
    try:
        _run_export(first_backend)
        report["exported"] = True
        report["export_backend"] = first_backend
        logger.info(f"[INFO] Exported FP32 ONNX using {first_backend}: {out_path}")
    except Exception as exc:
        first_error = str(exc)
        logger.warning(f"[WARN] FP32 ONNX export failed with {first_backend}: {exc}")
        try:
            _run_export(second_backend)
            report["exported"] = True
            report["export_backend"] = second_backend
            logger.info(f"[INFO] Exported FP32 ONNX using fallback {second_backend}: {out_path}")
        except Exception as exc2:
            report["error"] = f"{first_backend}: {first_error} | {second_backend}: {exc2}"
            logger.warning(f"[WARN] FP32 ONNX export fallback failed: {exc2}")
            return report

    onnx = _try_import_onnx()
    if onnx is None:
        logger.warning("[WARN] onnx package not installed. Skipping onnx.checker.")
    else:
        model_onnx = onnx.load(str(out_path))
        opset_imports = [int(x.version) for x in model_onnx.opset_import if x.domain in ("", "ai.onnx")]
        if opset_imports:
            report["actual_opset"] = int(max(opset_imports))
            if report["actual_opset"] != int(opset):
                logger.warning(
                    f"[WARN] Requested opset {int(opset)} but exported model opset is {report['actual_opset']}."
                )
        onnx.checker.check_model(model_onnx)
        logger.info("[INFO] ONNX checker passed")
        report["onnx_checked"] = True

    ort = _try_import_onnxruntime()
    if ort is None:
        logger.warning("[WARN] onnxruntime not installed. Skipping PyTorch-vs-ONNX comparison.")
        return report

    if sample_batch is None:
        sample_batch = dummy.detach().cpu()

    sample_batch = sample_batch[:1].detach().cpu()

    with torch.no_grad():
        pt_out = model(sample_batch.to(device))
        pt_out_np = [x.detach().cpu().numpy() for x in pt_out]

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {input_name: sample_batch.numpy().astype(np.float32)})

    report["ort_compared"] = True
    for name, p, o in zip(["p8", "p16", "p32"], pt_out_np, ort_out):
        diff = np.abs(p - o)
        report["output_max_abs_diff"][name] = float(diff.max())
        report["output_mean_abs_diff"][name] = float(diff.mean())

    logger.info(
        "[INFO] ONNX parity | "
        + " ".join(
            [
                f"{k}:max={report['output_max_abs_diff'][k]:.6g},mean={report['output_mean_abs_diff'][k]:.6g}"
                for k in ["p8", "p16", "p32"]
            ]
        )
    )

    return report
