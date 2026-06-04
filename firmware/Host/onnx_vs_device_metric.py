"""Evaluate the int8 ONNX on the host through the EXACT host-bench metric pipeline
(same load_image stretch-resize, same decode, same viz_common GT in model space,
same _ap / _match_image / _metrics_at_conf as host_balldetector_n6.py) and write
the metrics into model_comparison.csv next to the on-device row.

Purpose: separate "eval pipeline" from "device/NPU". Compare the host-ONNX row
against:
  * device (on-device bench row in model_comparison.csv)  -- the same tag w/o _onnx
  * ref    (fp32 training-val metrics, the ref_* columns)

If host-onnx ~= device  -> the gap is the EVAL PIPELINE (GT/metric), device is faithful.
If host-onnx ~= ref      -> the NPU degrades vs onnx (a device/decode/quant issue).

The row is upserted with a `_onnx` suffix tag (e.g. smallimgsize_v1_unpruned_int8_onnx)
so it lands beside the on-device row for a direct diff. Columns/units match the
bench writer; the timing columns are HOST CPU times (onnxruntime), not the NPU.

Run:
  software/venv/bin/python firmware/Host/onnx_vs_device_metric.py \
      --onnx software/final_models/ball_detection/20260527-035002-smallimgsize_v1/exports/int8_ptq_qdq.onnx \
      --model-name smallimgsize_v1_unpruned_int8_onnx
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

import viz_common
from host_balldetector_n6 import (
    _match_image, _ap, _metrics_at_conf, _load_ref_metrics, _pct,
    _upsert_comparison, _gather_paths, MODEL_W, MODEL_H,
)
from local_infer import decode_head, nms, STRIDE_BY_HW  # same decode as firmware

CONF_DECODE = 0.05   # firmware decodes at 0.05 and sends all boxes; metric applies --conf
NMS_IOU = 0.25
MAX_DET = 8


def _onnx_preds(sess, iname, path):
    """Run the int8 ONNX on one image, returning (preds, pre_ms, infer_ms, post_ms).
    preds is sorted desc by score, decode/NMS identical to the on-device path."""
    t0 = time.perf_counter()
    img = Image.open(path).convert("RGB").resize((MODEL_W, MODEL_H), Image.BILINEAR)  # stretch == host load_image
    x = (np.asarray(img, np.float32) / 255.0).transpose(2, 0, 1)[None]
    t1 = time.perf_counter()
    outs = sess.run(None, {iname: x})
    t2 = time.perf_counter()

    boxes_all, scores_all = [], []
    for r in outs:
        r = np.asarray(r)[0]
        st = STRIDE_BY_HW.get((r.shape[-2], r.shape[-1]))
        if st is None:
            continue
        b, s = decode_head(r, st)
        boxes_all.append(b); scores_all.append(s)
    boxes = np.concatenate(boxes_all); scores = np.concatenate(scores_all)
    m = scores >= CONF_DECODE
    boxes, scores = boxes[m], scores[m]
    keep = nms(boxes, scores, NMS_IOU, MAX_DET)
    preds = sorted([(float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]),
                     float(boxes[i, 3]), float(scores[i])) for i in keep], key=lambda b: -b[4])
    t3 = time.perf_counter()
    return preds, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image-list", default="../../software/ball_detection/splits/val.txt")
    ap.add_argument("--image-root", default="../../datasets/BALL")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.45,
                    help="operating conf for P/R/F1 -- 0.45 matches the BallDetector_N6 "
                         "firmware CONF_THR / host bench default, so the row is directly "
                         "comparable to the on-device row.")
    ap.add_argument("--iou", type=float, default=0.5, help="matching IoU for a TP")
    ap.add_argument("--model-name", default="",
                    help="row tag; default = <onnx model dir>_onnx so it sits beside the device row")
    ap.add_argument("--out-dir", default="",
                    help="dir for results_<tag>.csv + metrics.csv ref lookup; "
                         "default = the onnx's model dir (exports/..)")
    ap.add_argument("--comparison-csv",
                    default="../../software/final_models/ball_detection/model_comparison.csv")
    a = ap.parse_args()

    onnx_path = Path(a.onnx)
    out_dir = Path(a.out_dir) if a.out_dir else onnx_path.parent.parent  # exports/ -> model dir
    tag = a.model_name or f"{out_dir.name.split('-', 1)[-1]}_onnx"

    class _A:  # reuse host _gather_paths
        image_list = a.image_list; image_root = a.image_root; images = None; limit = a.limit
    paths = _gather_paths(_A())
    print(f"{len(paths)} images  ->  tag '{tag}'")

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    rows, all_scored, cerr_all = [], [], []
    n_gt_total = tp_t = fp_t = fn_t = 0
    iou_sum = 0.0
    pre_l, inf_l, post_l = [], [], []

    for k, p in enumerate(paths):
        preds, pre_ms, inf_ms, post_ms = _onnx_preds(sess, iname, p)
        gts = viz_common.load_gt_model_space(p)
        scored, sum_iou, cerr, n_fn = _match_image(preds, gts, a.iou)
        tp = sum(t for _, t in scored)
        fp = len(scored) - tp
        tp_t += tp; fp_t += fp; fn_t += n_fn
        iou_sum += sum_iou; cerr_all += cerr
        n_gt_total += len(gts); all_scored += scored
        pre_l.append(pre_ms); inf_l.append(inf_ms); post_l.append(post_ms)
        rows.append((p.name, len(gts), len(preds), tp, fp, n_fn, pre_ms, inf_ms, post_ms))
        if (k + 1) % 50 == 0 or k + 1 == len(paths):
            print(f"  [{k+1}/{len(paths)}] {p.name:32s} gt={len(gts)} pred={len(preds)}")

    n = len(rows)
    op_conf = a.conf
    map50 = _ap(all_scored, n_gt_total)
    precision, recall, f1, fp_per_image = _metrics_at_conf(all_scored, n_gt_total, n, op_conf)
    ref = _load_ref_metrics(out_dir)
    mean_iou = iou_sum / tp_t if tp_t else 0.0
    cerr_sorted = sorted(cerr_all)
    cerr_mean = sum(cerr_all) / len(cerr_all) if cerr_all else 0.0
    inf_sorted = sorted(inf_l)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / f"results_{tag}.csv"
    with open(results_csv, "w") as f:
        f.write("image,n_gt,n_pred,tp,fp,fn,pre_ms,infer_ms,post_ms\n")
        for rw in rows:
            f.write("{},{},{},{},{},{},{:.3f},{:.3f},{:.3f}\n".format(*rw))

    cols = ["model", "n_samples", "conf", "iou_match",
            "map50", "precision", "recall", "f1", "mean_iou",
            "center_err_px_mean", "center_err_px_median", "center_err_px_p95", "fp_per_image",
            "ref_map50", "ref_precision", "ref_recall", "ref_f1", "ref_mean_iou",
            "ref_center_err_px", "ref_fp_per_image",
            "pre_ms_mean", "infer_ms_mean", "post_ms_mean", "infer_ms_p50", "infer_ms_p95",
            "results_csv"]
    row = {
        "model": tag, "n_samples": n, "conf": f"{op_conf}", "iou_match": f"{a.iou}",
        "map50": f"{map50:.4f}", "precision": f"{precision:.4f}", "recall": f"{recall:.4f}",
        "f1": f"{f1:.4f}", "mean_iou": f"{mean_iou:.4f}",
        "center_err_px_mean": f"{cerr_mean:.3f}",
        "center_err_px_median": f"{_pct(cerr_sorted,0.5):.3f}",
        "center_err_px_p95": f"{_pct(cerr_sorted,0.95):.3f}",
        "fp_per_image": f"{fp_per_image:.4f}",
        "pre_ms_mean": f"{sum(pre_l)/n:.3f}" if n else "",
        "infer_ms_mean": f"{sum(inf_l)/n:.3f}" if n else "",
        "post_ms_mean": f"{sum(post_l)/n:.3f}" if n else "",
        "infer_ms_p50": f"{_pct(inf_sorted,0.5):.2f}",
        "infer_ms_p95": f"{_pct(inf_sorted,0.95):.2f}",
        "results_csv": str(results_csv),
    }
    row.update(ref)
    _upsert_comparison(Path(a.comparison_csv), cols, row, tag)

    print(f"\n=== {tag} : {n} images  (conf={op_conf}, iou={a.iou}) ===")
    print(f"  host-onnx:  map50={map50:.4f}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  mIoU={mean_iou:.4f}  fp/img={fp_per_image:.3f}")
    if ref:
        print(f"  training :  map50={ref.get('ref_map50','?')}  P={ref.get('ref_precision','?')}  "
              f"R={ref.get('ref_recall','?')}  F1={ref.get('ref_f1','?')}  mIoU={ref.get('ref_mean_iou','?')}  fp/img={ref.get('ref_fp_per_image','?')}")
    print(f"  center_err_px mean={cerr_mean:.2f} median={_pct(cerr_sorted,0.5):.2f} p95={_pct(cerr_sorted,0.95):.2f}  (train {ref.get('ref_center_err_px','?')})")
    print(f"  host timing (CPU, not NPU): pre={sum(pre_l)/max(n,1):.2f}ms infer={sum(inf_l)/max(n,1):.2f}ms post={sum(post_l)/max(n,1):.3f}ms")
    print(f"  -> {results_csv}\n  -> {a.comparison_csv}")


if __name__ == "__main__":
    main()
