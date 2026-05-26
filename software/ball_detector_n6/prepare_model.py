"""Sanity-check the int8 ONNX before handing it to stedgeai for the N6 NPU.

- Verifies expected I/O names and shapes.
- Confirms QDQ wrapping is present (a hint the model is properly quantized).
- Copies the chosen checkpoint to model/int8_ptq_qdq.onnx for stedgeai.

Run from the repo root:
    python software/ball_detector_n6/prepare_model.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import onnx

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ONNX = REPO_ROOT / (
    "software/ball_detection/runs/"
    "20260521-195948-simota_diou_v2_pruned_30/exports_int8acts/int8_320x256.onnx"
)
DST_DIR = REPO_ROOT / "software/ball_detector_n6/model"
DST_ONNX = DST_DIR / "int8_ptq_qdq.onnx"

EXPECTED_INPUT_SHAPE = [1, 3, 256, 320]
EXPECTED_OUTPUTS = {
    "p8":  [1, 5, 32, 40],
    "p16": [1, 5, 16, 20],
    "p32": [1, 5,  8, 10],
}


def _shape(t) -> list[int]:
    return [d.dim_value for d in t.type.tensor_type.shape.dim]


def main() -> None:
    if not SRC_ONNX.exists():
        raise SystemExit(f"source model not found: {SRC_ONNX}")

    m = onnx.load(str(SRC_ONNX))
    onnx.checker.check_model(m)

    inputs = {i.name: _shape(i) for i in m.graph.input}
    outputs = {o.name: _shape(o) for o in m.graph.output}

    if list(inputs.values())[0] != EXPECTED_INPUT_SHAPE:
        raise SystemExit(f"unexpected input shape {inputs}, want {EXPECTED_INPUT_SHAPE}")
    for name, shape in EXPECTED_OUTPUTS.items():
        if outputs.get(name) != shape:
            raise SystemExit(f"output {name} shape {outputs.get(name)} != {shape}")

    op_types = {n.op_type for n in m.graph.node}
    if "QuantizeLinear" not in op_types or "DequantizeLinear" not in op_types:
        raise SystemExit("model does not appear to be QDQ-quantized")

    DST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC_ONNX, DST_ONNX)

    src_mb = SRC_ONNX.stat().st_size / 1024 / 1024
    print(f"ok  source : {SRC_ONNX.relative_to(REPO_ROOT)}  ({src_mb:.2f} MB)")
    print(f"ok  input  : {list(inputs.items())[0]}")
    print(f"ok  outputs: {outputs}")
    print(f"ok  copied : {DST_ONNX.relative_to(REPO_ROOT)}")
    print()
    print("next:")
    print("  cd firmware/BallDetector_N6")
    print("  scripts/stedgeai_compile.sh")


if __name__ == "__main__":
    main()
