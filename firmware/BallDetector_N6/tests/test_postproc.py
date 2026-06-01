"""Cross-check Core/Src/yolo_postproc.c against firmware/Host/yolo_decode.py.

Builds yolo_postproc.c into a shared library, calls it from Python via
ctypes on a fixed RNG-seeded fake-output tensor, runs the same input
through the Python decoder, and asserts the boxes match.

If this fails, the C and Python decoders disagree — fix before flashing.

    cc -O2 -shared -fPIC -o /tmp/libyolo_postproc.so yolo_postproc.c -lm
    python test_postproc.py
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # firmware/BallDetector_N6/tests
REPO = HERE.parents[2]                           # repo root
# Reference Python decoder now lives next to the host script.
sys.path.insert(0, str(REPO / "firmware/Host"))
import yolo_decode  # noqa: E402

LIB = Path("/tmp/libyolo_postproc.so")
# Match the table baked into yolo_postproc.c.
HEAD_QPARAMS = [
    (0.256579876, 41,  8, 36, 48),  # p8
    (0.204285681, 0, 16, 18, 24),  # p16
    (0.146335885, 0, 32,  9, 12),  # p32
]


class YoloBox(ctypes.Structure):
    _fields_ = [("x1", ctypes.c_float), ("y1", ctypes.c_float),
                ("x2", ctypes.c_float), ("y2", ctypes.c_float),
                ("score", ctypes.c_float)]


def build_lib() -> ctypes.CDLL:
    subprocess.run(
        ["cc", "-O2", "-Wall", "-Wextra", "-shared", "-fPIC",
         "-I", str(HERE.parent / "FSBL" / "Core" / "Inc"),   # yolo_postproc.h (CubeMX FSBL tree)
         "-o", str(LIB), str(HERE.parent / "FSBL" / "Core" / "Src" / "yolo_postproc.c"), "-lm"],
        check=True,
    )
    lib = ctypes.CDLL(str(LIB))
    lib.yolo_postprocess.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),  # heads[3]
        ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(YoloBox), ctypes.c_int,
    ]
    lib.yolo_postprocess.restype = ctypes.c_int
    return lib


def make_fake_quantized_heads(seed: int = 0):
    """Build synthetic float head outputs, then quantize to uint8 the way
    stedgeai would, so we can drive both decoders from the same numbers."""
    rng = np.random.default_rng(seed)
    float_heads = []
    u8_heads = []
    for scale, zp, _stride, H, W in HEAD_QPARAMS:
        # Random tx/ty/tw/th around 0, plus a handful of high-objectness cells.
        h = rng.normal(0, 0.5, size=(1, 5, H, W)).astype(np.float32)
        # Inject 2 strong detections per head so NMS has something to do.
        for _ in range(2):
            y, x = rng.integers(0, H), rng.integers(0, W)
            h[0, 4, y, x] = rng.uniform(1.5, 3.0)  # high tobj -> sigmoid ~0.9
            h[0, 0:4, y, x] = rng.normal(0, 0.3, size=4)
        float_heads.append(h)

        # Symmetric quantization to uint8 using the same (scale, zp).
        q = np.round(h / scale + zp).clip(0, 255).astype(np.uint8)
        u8_heads.append(q)
    return float_heads, u8_heads


def call_c(lib: ctypes.CDLL, u8_heads, conf=0.50, iou=0.25, max_det=8):
    # Ensure contiguous CHW byte buffers.
    bufs = []
    for q in u8_heads:
        b = np.ascontiguousarray(q[0])  # drop batch -> [5,H,W]
        bufs.append(b)
    arr_t = ctypes.c_void_p * 3
    heads = arr_t(*(b.ctypes.data for b in bufs))
    out = (YoloBox * max_det)()
    n = lib.yolo_postprocess(heads, ctypes.c_float(conf), ctypes.c_float(iou),
                             out, max_det)
    return [(out[i].x1, out[i].y1, out[i].x2, out[i].y2, out[i].score)
            for i in range(n)]


def call_py(u8_heads, conf=0.50, iou=0.25, max_det=8):
    # Dequantize back to float so the Python decoder sees the same numbers
    # the C decoder reconstructs from uint8.
    floats = []
    for q, (scale, zp, *_ ) in zip(u8_heads, HEAD_QPARAMS):
        f = (q.astype(np.float32) - zp) * scale
        floats.append(f)
    boxes, scores = yolo_decode.decode_all(*floats)
    keep = yolo_decode.nms(boxes[0], scores[0],
                           iou_thresh=iou, conf_thresh=conf, max_det=max_det)
    return [(*boxes[0, k].tolist(), float(scores[0, k])) for k in keep]


def main() -> int:
    lib = build_lib()
    _, u8_heads = make_fake_quantized_heads(seed=42)

    c_boxes  = call_c(lib, u8_heads)
    py_boxes = call_py(u8_heads)

    print(f"C  produced {len(c_boxes)} boxes")
    print(f"Py produced {len(py_boxes)} boxes")
    if len(c_boxes) != len(py_boxes):
        print("FAIL: different counts")
        for label, bs in (("C", c_boxes), ("Py", py_boxes)):
            for b in bs:
                print(f"  {label}: {b}")
        return 1

    # Sort by a canonical key (score desc, then x1, y1) so a tie in objectness
    # score between two boxes -- where both decoders are correct but pick a
    # different order -- doesn't false-positive the comparison.
    def _canonical(b):
        return (-b[4], b[0], b[1], b[2], b[3])
    c_boxes  = sorted(c_boxes,  key=_canonical)
    py_boxes = sorted(py_boxes, key=_canonical)

    ok = True
    for i, (cb, pb) in enumerate(zip(c_boxes, py_boxes)):
        diffs = [abs(a - b) for a, b in zip(cb, pb)]
        if max(diffs) > 1e-3:
            ok = False
            print(f"  box {i} DIFF max={max(diffs):.4f}")
            print(f"    C : {cb}")
            print(f"    Py: {pb}")
        else:
            print(f"  box {i} ok  score={cb[4]:.3f}  "
                  f"[{cb[0]:.1f},{cb[1]:.1f},{cb[2]:.1f},{cb[3]:.1f}]")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
