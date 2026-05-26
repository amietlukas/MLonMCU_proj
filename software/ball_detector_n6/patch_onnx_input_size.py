"""Patch an FCN ONNX model to use a different fixed input resolution.

STYOLO is fully convolutional, so re-shaping the input propagates
cleanly. We rewrite the input/output dims explicitly and re-run shape
inference so QDQ nodes pick up correct intermediate shapes.

The new dims must be divisible by the largest stride (32) for the
3-scale [8,16,32] decoder.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import shape_inference

EXPECTED_STRIDES = (8, 16, 32)


def patch(in_path: Path, out_path: Path, new_h: int, new_w: int) -> None:
    if new_h % 32 != 0 or new_w % 32 != 0:
        raise ValueError(f"H={new_h}, W={new_w} must both be divisible by 32")

    m = onnx.load(str(in_path))

    # --- rewrite input dim ----------------------------------------------------
    if len(m.graph.input) != 1:
        raise ValueError(f"expected 1 input, got {len(m.graph.input)}")
    inp = m.graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    if len(dims) != 4:
        raise ValueError(f"expected 4D input, got shape {[d.dim_value for d in dims]}")
    old = [d.dim_value for d in dims]
    dims[2].dim_value = new_h
    dims[3].dim_value = new_w
    print(f"input  : {old} -> [{dims[0].dim_value}, {dims[1].dim_value}, {new_h}, {new_w}]")

    # --- rewrite output dims (CHW = [5, H/stride, W/stride]) ------------------
    expected_by_name = {
        "p8":  (5, new_h // 8,  new_w // 8),
        "p16": (5, new_h // 16, new_w // 16),
        "p32": (5, new_h // 32, new_w // 32),
    }
    for out in m.graph.output:
        if out.name not in expected_by_name:
            raise ValueError(f"unexpected output name: {out.name}")
        target = expected_by_name[out.name]
        out_dims = out.type.tensor_type.shape.dim
        if len(out_dims) != 4:
            raise ValueError(f"output {out.name} must be 4D")
        old = [d.dim_value for d in out_dims]
        out_dims[1].dim_value = target[0]
        out_dims[2].dim_value = target[1]
        out_dims[3].dim_value = target[2]
        print(f"output {out.name:<3}: {old} -> [1, {target[0]}, {target[1]}, {target[2]}]")

    # --- clear all intermediate value_info shapes; let inference recompute ----
    while len(m.graph.value_info) > 0:
        m.graph.value_info.pop()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m_inferred = shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
    onnx.checker.check_model(m_inferred)
    onnx.save(m_inferred, str(out_path))
    print(f"saved  : {out_path}  ({out_path.stat().st_size/1024/1024:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    args = p.parse_args()
    patch(Path(args.in_path), Path(args.out), args.height, args.width)


if __name__ == "__main__":
    main()
