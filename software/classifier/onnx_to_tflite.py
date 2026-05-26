#!/usr/bin/env python3
"""Convert the int8 QDQ ONNX classifier export to a TFLite Micro-friendly .tflite.

The training pipeline emits ONNX with the full preprocessing fused into the
first Conv (so the graph eats raw uint8 pixels reinterpreted as float). We
push that through onnx2tf, which lifts the int8 QDQ graph back into a Keras
SavedModel and then re-quantizes to a fully int8 .tflite. We don't keep the
SavedModel/JSON intermediates on disk.

Run inside software/venv_tflite (separate venv from the main training one,
since tensorflow 2.x pins numpy<2.1 and the training pipeline pins numpy 2.4).

Usage:
    venv_tflite/bin/python software/classifier/onnx_to_tflite.py \
        path/to/model_int8_qdq_op13.onnx \
        --out path/to/model.tflite
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def patch_onnx_kernel_shapes(src: Path, dst: Path) -> None:
    """Fill in Conv `kernel_shape` attrs missing from the PyTorch ONNX export.

    The PyTorch exporter sometimes drops Conv.kernel_shape because the ONNX
    spec lists it as optional when the weight tensor has a static shape.
    onnx2tf 1.27 assumes it's always present and falls back to kernel_size=0,
    which then asks numpy to transpose a 4-D conv weight with a 2-element
    perm — yielding the cryptic "axes don't match array" failure.
    """
    import onnx
    m = onnx.load(str(src))
    inits = {init.name: init for init in m.graph.initializer}
    producers = {out: node for node in m.graph.node for out in node.output}

    def initializer_shape_via_qdq(tensor_name: str) -> list[int] | None:
        """Return the shape of the ONNX initializer feeding `tensor_name`,
        walking back through DequantizeLinear / QuantizeLinear nodes."""
        seen = set()
        cur = tensor_name
        for _ in range(8):
            if cur in inits:
                return list(inits[cur].dims)
            if cur in seen or cur not in producers:
                return None
            seen.add(cur)
            prod = producers[cur]
            if prod.op_type in ("DequantizeLinear", "QuantizeLinear") and prod.input:
                cur = prod.input[0]
            else:
                return None
        return None

    patched = 0
    for node in m.graph.node:
        if node.op_type != "Conv":
            continue
        if any(a.name == "kernel_shape" for a in node.attribute):
            continue
        if len(node.input) < 2:
            continue
        wshape = initializer_shape_via_qdq(node.input[1])
        if wshape is None or len(wshape) < 3:
            print(f"  ! could not resolve weight shape for {node.name}")
            continue
        # ONNX Conv weight is [C_out, C_in/group, *spatial]; spatial dims are the rest.
        kshape = wshape[2:]
        node.attribute.append(onnx.helper.make_attribute("kernel_shape", kshape))
        patched += 1
    print(f"[+] patched Conv kernel_shape on {patched} node(s)")
    onnx.save(m, str(dst))


def _build_representative_dataset(calib_dir: Path, num_samples: int,
                                  h: int, w: int):
    """Yield FP32 grayscale tensors shaped (1, h, w, 1) in raw [0, 255] range.

    The classifier ONNX bakes /255 + normalization into the first Conv, so the
    on-MCU input pipeline (and TFLite Micro after re-quantization) sees raw
    pixel values cast to float. The representative dataset must mirror that.
    """
    import numpy as np
    from PIL import Image

    paths: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(calib_dir.rglob(ext))
    if not paths:
        raise RuntimeError(
            f"No calibration images found under {calib_dir} — pass --calib-dir.")
    paths = sorted(paths)[:num_samples]
    print(f"[+] using {len(paths)} calibration images from {calib_dir}")

    def gen():
        for p in paths:
            img = Image.open(p).convert("L").resize((w, h), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)  # (h, w) in [0, 255]
            arr = arr.reshape(1, h, w, 1)
            yield [arr]
    return gen


def convert(
    onnx_path: Path,
    out_path: Path,
    keep_workdir: Path | None,
    calib_dir: Path,
    num_samples: int,
    input_h: int,
    input_w: int,
) -> None:
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    workdir = Path(tempfile.mkdtemp(prefix="onnx2tf_")) if keep_workdir is None else keep_workdir
    workdir.mkdir(parents=True, exist_ok=True)

    patched_onnx = workdir / "input_patched.onnx"
    patch_onnx_kernel_shapes(onnx_path, patched_onnx)

    # Stage 1: ONNX -> TF SavedModel via onnx2tf. We let onnx2tf emit the
    # FP32 .tflite/saved_model only; full int8 PTQ happens in stage 2 against
    # a representative dataset of real grayscale images.
    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", str(patched_onnx),
        "-o", str(workdir),
        "-b", "1",
    ]
    print("[+] onnx2tf:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    saved_model_dir = workdir
    if not (saved_model_dir / "saved_model.pb").exists():
        raise RuntimeError(f"onnx2tf did not produce a SavedModel under {saved_model_dir}")

    # Stage 2: TFLite full-int8 PTQ from the SavedModel.
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = _build_representative_dataset(
        calib_dir, num_samples, input_h, input_w)
    # Force int8 input + output so the on-MCU host protocol can feed
    # int8 = (uint8 - 128) directly without an extra Quantize op at the entry.
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    print(f"[+] wrote:  {out_path}  ({out_path.stat().st_size:,} bytes)")

    # Echo the picked quantization params so the firmware constants.h can be
    # synced with whatever TFLite chose during calibration.
    interp = tf.lite.Interpreter(model_path=str(out_path))
    interp.allocate_tensors()
    print("[+] input  :", interp.get_input_details()[0])
    print("[+] output :", interp.get_output_details()[0])

    if keep_workdir is None:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("onnx", type=Path, help="Path to ONNX (fp32 or int8 QDQ)")
    p.add_argument("--out", type=Path, required=True, help="Destination .tflite path")
    p.add_argument(
        "--calib-dir",
        type=Path,
        default=Path("datasets/HAGRID/hagrid_full_qqvga_resize/val"),
        help="Directory of representative grayscale images for int8 PTQ "
             "(walked recursively for *.jpg/*.png).",
    )
    p.add_argument("--num-samples", type=int, default=128,
                   help="Number of calibration images to use.")
    p.add_argument("--input-h", type=int, default=120, help="Model input height.")
    p.add_argument("--input-w", type=int, default=160, help="Model input width.")
    p.add_argument(
        "--keep-workdir",
        type=Path,
        default=None,
        help="If set, keep onnx2tf intermediates under this directory for inspection.",
    )
    args = p.parse_args()
    if not args.onnx.exists():
        p.error(f"ONNX not found: {args.onnx}")
    calib_dir = args.calib_dir if args.calib_dir.is_absolute() \
        else (Path(__file__).resolve().parents[2] / args.calib_dir)
    if not calib_dir.exists():
        p.error(f"Calibration directory not found: {calib_dir}")
    convert(
        args.onnx.resolve(),
        args.out,
        args.keep_workdir,
        calib_dir,
        args.num_samples,
        args.input_h,
        args.input_w,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
