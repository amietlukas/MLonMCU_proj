#!/usr/bin/env python3
"""Local PC sanity-check for BallDetector_N6.

Runs the SAME quantized model (int8_ptq_qdq.onnx) on the SAME images with the
SAME decode + thresholds as the firmware (constants.h / yolo_postproc.c), then
renders results to n6_viz/<name>_pc.png next to the board's <name>_mcu.png.

Interpretation:
  * PC boxes good, MCU boxes bad  -> the MCU/firmware computation is wrong
    (e.g. input layout, output handling) — NOT the model.
  * PC and MCU similarly bad       -> the model degraded from quantization; the
    MCU is faithfully reproducing a weak quantized model.

Run with the venv that has onnxruntime:
  software/venv/bin/python firmware/Host/local_infer.py \
      --image-list software/ball_detection/splits/val.txt --limit 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

import viz_common

MODEL_W, MODEL_H = 384, 288
CONF_THR, NMS_IOU, MAX_DET = 0.45, 0.25, 8     # == firmware constants.h
TWTH_CLAMP = 4.0                               # == yolo_postproc.c
STRIDE_BY_HW = {(36, 48): 8, (18, 24): 16, (9, 12): 32}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_head(pred, stride):                 # pred: (5, H, W) float
    _, H, W = pred.shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    tx, ty, tw, th, to = pred[0], pred[1], pred[2], pred[3], pred[4]
    cx = (sigmoid(tx) + gx) * stride
    cy = (sigmoid(ty) + gy) * stride
    pw = np.exp(np.clip(tw, -TWTH_CLAMP, TWTH_CLAMP)) * stride
    ph = np.exp(np.clip(th, -TWTH_CLAMP, TWTH_CLAMP)) * stride
    boxes = np.stack([cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2], axis=-1).reshape(-1, 4)
    return boxes, sigmoid(to).reshape(-1)


def nms(boxes, scores, iou_thr, max_det):
    order = scores.argsort()[::-1]
    keep = []
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.clip(xx2 - xx1, 0, None); h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (a_i + a_r - inter + 1e-9)
        order = rest[iou < iou_thr]
    return keep


def resolve(line: str, root: Path) -> Path:
    if line.startswith("spl/"):
        return root / ("SPLDataset/" + line[4:])
    if line.startswith("our/"):
        return root / ("OurDataset/" + line[4:])
    return root / line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="firmware/BallDetector_N6/model/int8_ptq_qdq.onnx")
    ap.add_argument("--image-list", default="software/ball_detection/splits/val.txt")
    ap.add_argument("--image-root", default="datasets/BALL")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--viz-dir", default=str(Path(__file__).resolve().parent / "n6_viz/local_infer"))
    a = ap.parse_args()

    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    root = Path(a.image_root)

    lines = [l.strip() for l in Path(a.image_list).read_text().splitlines() if l.strip()]
    if a.limit:
        lines = lines[:a.limit]

    for line in lines:
        p = resolve(line, root)
        if not p.exists():
            print(f"  skip (not found): {p}")
            continue
        img = Image.open(p).convert("RGB").resize((MODEL_W, MODEL_H), Image.BILINEAR)
        x = (np.asarray(img, np.float32) / 255.0).transpose(2, 0, 1)[None]   # NCHW [0,1]
        outs = sess.run(None, {iname: x})

        boxes_all, scores_all = [], []
        for r in outs:
            r = np.asarray(r)[0]                                            # (5,H,W)
            st = STRIDE_BY_HW.get((r.shape[-2], r.shape[-1]))
            b, s = decode_head(r, st)
            boxes_all.append(b); scores_all.append(s)
        boxes = np.concatenate(boxes_all); scores = np.concatenate(scores_all)
        m = scores >= CONF_THR
        boxes, scores = boxes[m], scores[m]
        keep = nms(boxes, scores, NMS_IOU, MAX_DET)
        preds = [(float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]),
                  float(boxes[i, 3]), float(scores[i])) for i in keep]

        print(f"{p.name:40s}  {len(preds)} det")
        for (x1, y1, x2, y2, s) in preds:
            print(f"    score={s:.3f}  [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
        out = viz_common.render_detections(p, preds, Path(a.viz_dir) / f"{p.stem}_pc.png",
                                           tag="PC onnx chw")
        print(f"    -> {out}")


if __name__ == "__main__":
    main()
