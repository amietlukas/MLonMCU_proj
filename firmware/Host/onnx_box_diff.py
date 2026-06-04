"""Diff the board's per-image boxes (from `bench --dump-boxes`) against the int8
ONNX run on the host, image-by-image, using the SAME decode params as the firmware
(conf 0.05 decode, NMS 0.25, max_det 8). Isolates whether the on-device path
diverges from ONNX at the box level (and by how much).

Run:
  software/venv/bin/python firmware/Host/onnx_box_diff.py \
      --dump /tmp/board_boxes.csv \
      --onnx software/final_models/ball_detection/20260527-035002-smallimgsize_v1/exports/int8_ptq_qdq.onnx \
      --image-root datasets/BALL
"""
from __future__ import annotations
import argparse, csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from host_balldetector_n6 import _iou, MODEL_W, MODEL_H
from local_infer import decode_head, nms, STRIDE_BY_HW

CONF_DECODE = 0.05   # == firmware on-device decode threshold
NMS_IOU, MAX_DET = 0.25, 8


def onnx_boxes(sess, iname, path):
    img = Image.open(path).convert("RGB").resize((MODEL_W, MODEL_H), Image.BILINEAR)
    x = (np.asarray(img, np.float32) / 255.0).transpose(2, 0, 1)[None]
    outs = sess.run(None, {iname: x})
    ball, scall = [], []
    for r in outs:
        r = np.asarray(r)[0]
        st = STRIDE_BY_HW.get((r.shape[-2], r.shape[-1]))
        if st is None:
            continue
        b, s = decode_head(r, st)
        ball.append(b); scall.append(s)
    boxes = np.concatenate(ball); scores = np.concatenate(scall)
    m = scores >= CONF_DECODE
    boxes, scores = boxes[m], scores[m]
    keep = nms(boxes, scores, NMS_IOU, MAX_DET)
    return sorted([(float(boxes[i,0]), float(boxes[i,1]), float(boxes[i,2]),
                    float(boxes[i,3]), float(scores[i])) for i in keep], key=lambda b: -b[4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="board boxes csv from bench --dump-boxes")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image-root", default="../../datasets/BALL")
    ap.add_argument("--image-list", default="../../software/ball_detection/splits/val.txt")
    a = ap.parse_args()

    # board boxes grouped by image name
    board = defaultdict(list)
    for r in csv.DictReader(open(a.dump)):
        board[r["image"]].append((float(r["x1"]), float(r["y1"]), float(r["x2"]),
                                  float(r["y2"]), float(r["score"])))

    # map image basename -> full path via the split list
    name2path = {}
    root = Path(a.image_root)
    for line in Path(a.image_list).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = root / ("SPLDataset/" + line[4:] if line.startswith("spl/")
                    else "OurDataset/" + line[4:] if line.startswith("our/") else line)
        name2path[p.name] = p

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    n_img = 0
    top_iou, top_dscore = [], []           # for the highest-score box per image
    n_board_only = n_onnx_only = n_matched = 0
    examples = []
    for name, bboxes in board.items():
        p = name2path.get(name)
        if p is None or not p.exists():
            continue
        n_img += 1
        oboxes = onnx_boxes(sess, iname, p)

        # greedy IoU pairing board<->onnx
        used = [False] * len(oboxes)
        for bb in bboxes:
            best, biou = -1, 0.30
            for j, ob in enumerate(oboxes):
                if used[j]:
                    continue
                v = _iou(bb[:4], ob[:4])
                if v >= biou:
                    best, biou = j, v
            if best >= 0:
                used[best] = True
                n_matched += 1
            else:
                n_board_only += 1
        n_onnx_only += used.count(False)

        # top-box comparison (the detection that drives most metrics)
        if bboxes and oboxes:
            bb, ob = bboxes[0], oboxes[0]
            top_iou.append(_iou(bb[:4], ob[:4]))
            top_dscore.append(bb[4] - ob[4])
            if len(examples) < 8:
                examples.append((name, bb, ob))

    def stats(v):
        if not v: return "n/a"
        s = sorted(v); n = len(s)
        return f"mean={sum(v)/n:+.4f} med={s[n//2]:+.4f} min={s[0]:+.4f} max={s[-1]:+.4f}"

    print(f"\n=== board vs ONNX boxes ({n_img} images, decode conf {CONF_DECODE}) ===")
    print(f"  paired boxes (IoU>=0.30): {n_matched}   board-only: {n_board_only}   onnx-only: {n_onnx_only}")
    print(f"  TOP-box IoU (board vs onnx): {stats(top_iou)}")
    print(f"  TOP-box score delta (board - onnx): {stats(top_dscore)}")
    print("\n  examples (name | board top [x1,y1,x2,y2,score] | onnx top):")
    for name, bb, ob in examples:
        print(f"    {name:34s} B[{bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f},{bb[4]:.2f}]"
              f"  O[{ob[0]:.0f},{ob[1]:.0f},{ob[2]:.0f},{ob[3]:.0f},{ob[4]:.2f}]")


if __name__ == "__main__":
    main()
