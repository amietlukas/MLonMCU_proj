"""NumPy decoder for the BallSTYOLONano N6 outputs.

Mirrors `ball_detection/src/training/decode.py` so on-board raw outputs
(or float ONNX runs) produce identical boxes to the training-time eval.
"""
from __future__ import annotations

import numpy as np

STRIDES = (8, 16, 32)
TWTH_CLAMP_MIN = -4.0
TWTH_CLAMP_MAX = 4.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_single_scale(pred: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """pred: [B, 5, H, W] -> boxes [B, H*W, 4] xyxy, scores [B, H*W]."""
    b, _, h, w = pred.shape
    tx, ty, tw, th, tobj = (pred[:, i, :, :] for i in range(5))

    gy, gx = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32),
                         indexing="ij")
    gx = gx[None, ...]
    gy = gy[None, ...]

    cx = (_sigmoid(tx) + gx) * stride
    cy = (_sigmoid(ty) + gy) * stride
    pw = np.exp(np.clip(tw, TWTH_CLAMP_MIN, TWTH_CLAMP_MAX)) * stride
    ph = np.exp(np.clip(th, TWTH_CLAMP_MIN, TWTH_CLAMP_MAX)) * stride

    x1 = cx - 0.5 * pw
    y1 = cy - 0.5 * ph
    x2 = cx + 0.5 * pw
    y2 = cy + 0.5 * ph

    boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(b, -1, 4)
    scores = _sigmoid(tobj).reshape(b, -1)
    return boxes, scores


def decode_all(p8: np.ndarray, p16: np.ndarray, p32: np.ndarray,
               input_h: int = 480, input_w: int = 640
               ) -> tuple[np.ndarray, np.ndarray]:
    bxs, scs = [], []
    for pred, stride in zip((p8, p16, p32), STRIDES):
        b, s = decode_single_scale(pred, stride)
        bxs.append(b)
        scs.append(s)
    boxes = np.concatenate(bxs, axis=1)
    scores = np.concatenate(scs, axis=1)
    boxes[..., 0] = np.clip(boxes[..., 0], 0, input_w)
    boxes[..., 1] = np.clip(boxes[..., 1], 0, input_h)
    boxes[..., 2] = np.clip(boxes[..., 2], 0, input_w)
    boxes[..., 3] = np.clip(boxes[..., 3], 0, input_h)
    return boxes, scores


def nms(boxes: np.ndarray, scores: np.ndarray,
        iou_thresh: float = 0.25, conf_thresh: float = 0.50,
        max_det: int = 3) -> np.ndarray:
    """Returns array of indices into (boxes, scores) that survive NMS."""
    mask = scores >= conf_thresh
    if not mask.any():
        return np.empty((0,), dtype=np.int64)
    idxs = np.nonzero(mask)[0]
    boxes = boxes[idxs]
    scores = scores[idxs]
    order = scores.argsort()[::-1]

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    keep: list[int] = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(int(idxs[i]))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou < iou_thresh]
    return np.array(keep, dtype=np.int64)
