"""Serial driver for the BallDetector_N6 firmware.

Two modes:
  --mode img  : push test images from a folder, draw the returned bboxes.
  --mode cam  : ask the board to stream camera frames + detections, display live.

Wire format is defined in protocol_balldetector_n6.md.
"""
from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial
from PIL import Image

try:
    import cv2  # only needed for --mode cam (JPEG decode + window)
except ImportError:
    cv2 = None

import viz_common  # GT lookup + render (same dir)

MAGIC = b"\xAA\x55\xA5\x5A"

T_HELLO     = 0x01
T_INFO      = 0x02
T_IMG_BEGIN = 0x10
T_IMG_CHUNK = 0x11
T_IMG_END   = 0x12
T_INFER_RAW = 0x20
T_INFER_DEC = 0x21
T_CAM_START = 0x30
T_CAM_STOP  = 0x31
T_CAM_FRAME = 0x32
T_LOG       = 0xFE
T_ACK       = 0xFF

CHUNK = 4096
# Matches the deployed int8 model (firmware/BallDetector_N6/model/int8_ptq_qdq.onnx):
# 384x288 RGB, 4:3 to match the B-CAMS-IMX sensor aspect. The board echoes these
# back in its INFO frame; hello() prints them so a mismatch is visible at connect.
MODEL_W, MODEL_H, MODEL_C = 384, 288, 3


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


def send_frame(ser: serial.Serial, ftype: int, payload: bytes = b"") -> None:
    header = struct.pack("<BI", ftype, len(payload))
    body = header + payload
    crc = crc16_ccitt(body)
    ser.write(MAGIC + body + struct.pack("<H", crc))


def read_frame(ser: serial.Serial, timeout_s: float = 120.0) -> tuple[int, bytes]:
    deadline = time.monotonic() + timeout_s
    window = b""
    while time.monotonic() < deadline:
        b = ser.read(1)
        if not b:
            continue
        window = (window + b)[-4:]
        if window == MAGIC:
            break
    else:
        raise TimeoutError("no frame magic on serial")

    header = ser.read(5)
    if len(header) != 5:
        raise IOError("truncated header")
    ftype, length = struct.unpack("<BI", header)
    payload = ser.read(length)
    if len(payload) != length:
        raise IOError(f"truncated payload (got {len(payload)} of {length})")
    crc_bytes = ser.read(2)
    if len(crc_bytes) != 2:
        raise IOError("truncated crc")
    crc_recv, = struct.unpack("<H", crc_bytes)
    crc_calc = crc16_ccitt(header + payload)
    if crc_recv != crc_calc:
        raise IOError(f"crc mismatch: got 0x{crc_recv:04x} want 0x{crc_calc:04x}")
    return ftype, payload


def load_image(path: Path, layout: str = "chw") -> bytes:
    """RGB888 at model resolution, as the generated network's uint8 input tensor.

    The network was generated with `--input-data-type uint8`, so the NPU input
    buffer expects RAW uint8 pixels (0..255) and applies the QLinear(1/255,-128)
    quantization internally. Do NOT pre-subtract 128 — that double-converts and
    the NPU sees garbage (the long-standing wrong-detections bug). The model is
    NCHW, so 'chw' is planar (the correct layout); 'hwc' is only for A/B testing.
    """
    img = Image.open(path).convert("RGB").resize((MODEL_W, MODEL_H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)           # HWC, raw 0..255
    if layout == "chw":
        arr = np.transpose(arr, (2, 0, 1))          # -> CHW planar
    return np.ascontiguousarray(arr, dtype=np.uint8).tobytes()


def hello(ser: serial.Serial) -> dict:
    send_frame(ser, T_HELLO)
    while True:
        ftype, payload = read_frame(ser)
        if ftype == T_INFO:
            fw_ver, w, h, c, n_out = struct.unpack("<HHHBB", payload[:8])
            return {"fw_ver": fw_ver, "w": w, "h": h, "c": c, "n_outputs": n_out}
        if ftype == T_LOG:
            print(f"[fw] {payload.decode(errors='ignore').rstrip()}")


def push_image(ser: serial.Serial, img_bytes: bytes, chunk_delay: float = 0.0,
               fmt: int = 0, retries: int = 5) -> dict:
    # fmt: 0 = planar CHW (matches the NCHW model), 1 = interleaved HWC
    # The UART link has no flow control. The firmware now rejects any image with a
    # missing chunk ("IMG size mismatch"); we re-send the whole image (up to
    # `retries`) so a dropped chunk can't silently corrupt a frame's pixels.
    for attempt in range(retries):
        ser.reset_input_buffer()
        send_frame(ser, T_IMG_BEGIN, struct.pack("<HHBB", MODEL_W, MODEL_H, MODEL_C, fmt))
        for i in range(0, len(img_bytes), CHUNK):
            chunk = img_bytes[i:i + CHUNK]
            send_frame(ser, T_IMG_CHUNK, struct.pack("<H", i // CHUNK) + chunk)
            if chunk_delay:
                time.sleep(chunk_delay)   # pace the upload (no flow ctrl)
        send_frame(ser, T_IMG_END)

        resend = False
        while True:
            try:
                ftype, payload = read_frame(ser, timeout_s=5.0)
            except TimeoutError:
                resend = True; break               # no reply -> re-send the image
            if ftype == T_LOG:
                msg = payload.decode(errors="ignore").rstrip()
                print(f"[fw] {msg}")
                if "size mismatch" in msg.lower():
                    resend = True; break           # dropped chunk -> re-send
                continue
            if ftype == T_INFER_DEC:
                pre_us, inf_us, post_us, n = struct.unpack("<IIIH", payload[:14])
                boxes = []
                off = 14
                for _ in range(n):
                    x1, y1, x2, y2, s = struct.unpack("<5f", payload[off:off + 20])
                    boxes.append((x1, y1, x2, y2, s))
                    off += 20
                return {"pre_us": pre_us, "inference_us": inf_us, "post_us": post_us,
                        "boxes": boxes}
        if not resend:
            break
    raise IOError(f"image upload failed after {retries} attempts (persistent UART drops)")


def _resolve_list_entry(line: str, root: Path) -> Path:
    """Resolve a BALL-split path (e.g. 'spl/foo/bar.png' or 'our/Kueche/x.jpg')
    against the dataset root. The split files use 'spl/' and 'our/' prefixes
    that map to the SPLDataset/ and OurDataset/ source dirs."""
    mapped = line
    if line.startswith("spl/"):
        mapped = "SPLDataset/" + line[len("spl/"):]
    elif line.startswith("our/"):
        mapped = "OurDataset/" + line[len("our/"):]
    return root / mapped


def _gather_paths(args: argparse.Namespace) -> list[Path]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    if args.image_list:
        root = Path(args.image_root)
        paths = []
        for line in Path(args.image_list).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = _resolve_list_entry(line, root)
            if p.exists():
                paths.append(p)
            else:
                print(f"  skip (not found): {p}", file=sys.stderr)
    else:
        paths = sorted(p for p in Path(args.images).iterdir()
                       if p.suffix.lower() in exts)
    if args.limit:
        paths = paths[:args.limit]
    return paths


def mode_img(args: argparse.Namespace) -> int:
    paths = _gather_paths(args)
    if not paths:
        print("no images to process (check --images/--image-list)", file=sys.stderr)
        return 2

    times_ms: list[float] = []
    n_det_total = 0
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        info = hello(ser)
        print(f"connected, fw=0x{info['fw_ver']:04x}, model {info['w']}x{info['h']}x{info['c']}")
        for p in paths:
            buf = load_image(p, args.layout)
            r = push_image(ser, buf, chunk_delay=args.chunk_delay,
                           fmt=(0 if args.layout == "chw" else 1))
            ms = r["inference_us"] / 1000.0
            times_ms.append(ms)
            n_det_total += len(r["boxes"])
            print(f"{p.name:40s}  {ms:7.1f} ms  {len(r['boxes'])} det")
            for x1, y1, x2, y2, s in r["boxes"]:
                print(f"    score={s:.3f}  [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
            if args.viz_dir:
                out = viz_common.render_detections(
                    p, r["boxes"], Path(args.viz_dir) / f"{p.stem}_mcu.png",
                    tag=f"MCU {args.layout}")
                print(f"    -> {out}")

    # Timing summary over the set (on-board NPU inference only, not UART/transfer).
    if times_ms:
        t = sorted(times_ms)
        n = len(t)
        mean = sum(t) / n
        median = t[n // 2] if n % 2 else (t[n // 2 - 1] + t[n // 2]) / 2
        print("\n--- inference timing over {} images ---".format(n))
        print(f"  mean {mean:.2f} ms   median {median:.2f} ms   "
              f"min {t[0]:.2f} ms   max {t[-1]:.2f} ms")
        if mean > 0:
            print(f"  throughput ~{1000.0 / mean:.1f} inf/s (NPU only)   "
                  f"{n_det_total} detections total")
        else:
            print(f"  throughput unavailable (no completed NPU inference)   "
                  f"{n_det_total} detections total")
    return 0


def mode_cam(args: argparse.Namespace) -> int:
    if cv2 is None:
        print("opencv-python required for --mode cam", file=sys.stderr)
        return 2
    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        info = hello(ser)
        print(f"connected, fw=0x{info['fw_ver']:04x}")
        send_frame(ser, T_CAM_START, struct.pack("<H", args.period_ms))
        try:
            while True:
                ftype, payload = read_frame(ser, timeout_s=2.0)
                if ftype == T_LOG:
                    print(f"[fw] {payload.decode(errors='ignore').rstrip()}")
                    continue
                if ftype != T_CAM_FRAME:
                    continue
                frame_idx, jlen = struct.unpack("<II", payload[:8])
                jpeg = payload[8:8 + jlen]
                rest = payload[8 + jlen:]
                inf_us, n = struct.unpack("<IH", rest[:6])
                boxes = []
                off = 6
                for _ in range(n):
                    x1, y1, x2, y2, s = struct.unpack("<5f", rest[off:off + 20])
                    boxes.append((x1, y1, x2, y2, s))
                    off += 20

                img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                for x1, y1, x2, y2, s in boxes:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f"{s:.2f}", (int(x1), max(0, int(y1) - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(img, f"#{frame_idx}  {inf_us/1000:.1f} ms",
                            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("BallDetector_N6", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            send_frame(ser, T_CAM_STOP)
            cv2.destroyAllWindows()
    return 0


# ---------------------------------------------------------------------------
# Benchmark mode: run a val set, compute detection metrics + per-stage timing,
# write a per-model results_<tag>.csv and upsert a model_comparison.csv row.
# Boxes are in 384x288 model space; GT comes from viz_common.load_gt_model_space
# (same stretch-to-model scaling as the image we send). Single class (ball).
# ---------------------------------------------------------------------------
def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _match_image(preds, gts, iou_thr):
    """preds sorted desc by score. Greedy highest-score-first GT matching.
    Returns (scored=[(score,is_tp)], sum_iou_of_TP, [center_err_px], n_fn)."""
    used = [False] * len(gts)
    scored, sum_iou, cerr = [], 0.0, []
    for (x1, y1, x2, y2, s) in preds:
        best, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if used[j]:
                continue
            v = _iou((x1, y1, x2, y2), g)
            if v >= best_iou:
                best, best_iou = j, v
        if best >= 0:
            used[best] = True
            scored.append((s, 1))
            sum_iou += best_iou
            gx, gy = (gts[best][0]+gts[best][2])/2.0, (gts[best][1]+gts[best][3])/2.0
            cerr.append((((x1+x2)/2.0 - gx) ** 2 + ((y1+y2)/2.0 - gy) ** 2) ** 0.5)
        else:
            scored.append((s, 0))
    return scored, sum_iou, cerr, used.count(False)


def _ap(all_scored, n_gt) -> float:
    """All-point AP at the match IoU (single class)."""
    if n_gt == 0 or not all_scored:
        return 0.0
    s = sorted(all_scored, key=lambda x: -x[0])
    tp = fp = 0
    rec, prec = [], []
    for _, istp in s:
        tp += istp
        fp += (1 - istp)
        rec.append(tp / n_gt)
        prec.append(tp / (tp + fp))
    for i in range(len(prec) - 2, -1, -1):        # monotonic precision envelope
        prec[i] = max(prec[i], prec[i + 1])
    ap = prev = 0.0
    for i in range(len(rec)):
        ap += prec[i] * (rec[i] - prev)
        prev = rec[i]
    return ap


def _metrics_at_conf(all_scored, n_gt, n_img, conf):
    """Precision/recall/F1/fp_per_image counting only detections with score>=conf
    (matches training's eval at cfg.eval.conf_threshold). is_tp was assigned by a
    greedy score-sorted match, so filtering by score is equivalent to re-matching."""
    tp = sum(1 for s, istp in all_scored if s >= conf and istp)
    fp = sum(1 for s, istp in all_scored if s >= conf and not istp)
    fn = n_gt - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, (fp / n_img if n_img else 0.0)


def _load_ref_metrics(model_dir):
    """Pull the training-time val metrics (best-val_map50 epoch) from the model's
    metrics.csv so on-device vs training sit side-by-side in the table. These are
    at the same conf_threshold (0.5) / nms (0.25) / matching IoU (0.5)."""
    p = Path(model_dir) / "metrics.csv"
    if not p.is_file():
        return {}
    try:
        rows = list(csv.DictReader(open(p, newline="")))
    except Exception:
        return {}
    if not rows:
        return {}
    def fnum(r, k):
        try:
            return float(r.get(k, "") or "nan")
        except ValueError:
            return float("nan")
    best = max(rows, key=lambda r: (fnum(r, "val_map50") if fnum(r, "val_map50") == fnum(r, "val_map50") else -1.0))
    def fmt(k, nd=4):
        v = best.get(k, "")
        try:
            return f"{float(v):.{nd}f}"
        except (ValueError, TypeError):
            return ""
    return {
        "ref_map50":         fmt("val_map50"),
        "ref_precision":     fmt("val_precision"),
        "ref_recall":        fmt("val_recall"),
        "ref_f1":            fmt("val_f1"),
        "ref_mean_iou":      fmt("val_mean_iou"),
        "ref_center_err_px": fmt("val_center_error_px", 3),
        "ref_fp_per_image":  fmt("val_fp_per_image"),
    }


def _pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _upsert_comparison(cmp_csv: Path, cols, row, tag):
    """Replace/insert the row for `tag`; leave other models' rows untouched."""
    existing = []
    extra_cols = []  # columns present in the file but not in `cols`
    if cmp_csv.is_file():
        lines = cmp_csv.read_text().splitlines()
        if lines:
            hdr = lines[0].split(",")
            extra_cols = [c for c in hdr if c not in cols]  # e.g. params/macc/flash_KB/ram_KB/energy_mJ
            for ln in lines[1:]:
                if not ln.strip():
                    continue
                d = dict(zip(hdr, ln.split(",")))
                if d.get("model") != tag:
                    existing.append(d)
    existing.append(row)
    # Preserve any extra columns (e.g. the static/energy cols added by
    # ball_add_static.py) so a bench upsert doesn't wipe them. The freshly
    # upserted row leaves them blank -- re-run ball_add_static.py to fill it.
    out_cols = list(cols) + extra_cols
    cmp_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(cmp_csv, "w") as f:
        f.write(",".join(out_cols) + "\n")
        for d in existing:
            f.write(",".join(str(d.get(c, "")) for c in out_cols) + "\n")


def mode_bench(args) -> int:
    paths = _gather_paths(args)
    if not paths:
        print("no images (check --image-list/--image-root)", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model_name
    results_csv = out_dir / f"results_{tag}.csv"

    rows, all_scored, cerr_all = [], [], []
    dump_rows = []
    n_gt_total = tp_t = fp_t = fn_t = 0
    iou_sum = 0.0
    pre_l, inf_l, post_l = [], [], []

    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        info = hello(ser)
        print(f"connected fw=0x{info['fw_ver']:04x} model {info['w']}x{info['h']}x{info['c']}")
        if (info["w"], info["h"]) != (MODEL_W, MODEL_H):
            print(f"WARNING: board model {info['w']}x{info['h']} != host MODEL {MODEL_W}x{MODEL_H}; "
                  f"GT scaling uses the host MODEL_* constants -- set them equal for correct metrics.",
                  file=sys.stderr)
        for k, p in enumerate(paths):
            buf = load_image(p, "hwc")           # HWC -> board transposes (pre stage)
            r = push_image(ser, buf, chunk_delay=args.chunk_delay, fmt=1)
            preds = sorted(r["boxes"], key=lambda b: -b[4])   # all boxes; conf applied to metrics below
            if args.dump_boxes:
                for (bx1, by1, bx2, by2, bsc) in preds:
                    dump_rows.append((p.name, bx1, by1, bx2, by2, bsc))
            gts = viz_common.load_gt_model_space(p)
            scored, sum_iou, cerr, n_fn = _match_image(preds, gts, args.iou)
            tp = sum(t for _, t in scored)
            fp = len(scored) - tp
            tp_t += tp; fp_t += fp; fn_t += n_fn
            iou_sum += sum_iou; cerr_all += cerr
            n_gt_total += len(gts); all_scored += scored
            pre_l.append(r["pre_us"] / 1000.0)
            inf_l.append(r["inference_us"] / 1000.0)
            post_l.append(r["post_us"] / 1000.0)
            rows.append((p.name, len(gts), len(preds), tp, fp, n_fn,
                         r["pre_us"]/1000.0, r["inference_us"]/1000.0, r["post_us"]/1000.0))
            if (k + 1) % 25 == 0 or k + 1 == len(paths):
                print(f"  [{k+1}/{len(paths)}] {p.name:32s} gt={len(gts)} pred={len(preds)} "
                      f"infer={r['inference_us']/1000:.1f}ms")

    n = len(rows)
    if args.dump_boxes:
        with open(args.dump_boxes, "w") as f:
            f.write("image,x1,y1,x2,y2,score\n")
            for rw in dump_rows:
                f.write("{},{:.3f},{:.3f},{:.3f},{:.3f},{:.4f}\n".format(*rw))
        print(f"  dumped {len(dump_rows)} board boxes -> {args.dump_boxes}")
    op_conf = args.conf
    map50 = _ap(all_scored, n_gt_total)                       # AP over all boxes (~training ap_conf)
    precision, recall, f1, fp_per_image = _metrics_at_conf(all_scored, n_gt_total, n, op_conf)
    ref = _load_ref_metrics(args.out_dir)                     # training metrics, side-by-side
    mean_iou = iou_sum / tp_t if tp_t else 0.0
    cerr_sorted = sorted(cerr_all)
    cerr_mean = sum(cerr_all) / len(cerr_all) if cerr_all else 0.0
    inf_sorted = sorted(inf_l)

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
        "model": tag, "n_samples": n, "conf": f"{op_conf}", "iou_match": f"{args.iou}",
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
    row.update(ref)   # ref_map50 / ref_precision / ... (blank if no metrics.csv)
    _upsert_comparison(Path(args.comparison_csv), cols, row, tag)

    print(f"\n=== {tag} : {n} images  (conf={op_conf}, iou={args.iou}) ===")
    print(f"  on-device:  map50={map50:.4f}  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  mIoU={mean_iou:.4f}  fp/img={fp_per_image:.3f}")
    if ref:
        print(f"  training :  map50={ref.get('ref_map50','?')}  P={ref.get('ref_precision','?')}  "
              f"R={ref.get('ref_recall','?')}  F1={ref.get('ref_f1','?')}  mIoU={ref.get('ref_mean_iou','?')}  fp/img={ref.get('ref_fp_per_image','?')}")
    print(f"  center_err_px mean={cerr_mean:.2f} median={_pct(cerr_sorted,0.5):.2f} p95={_pct(cerr_sorted,0.95):.2f}  (train {ref.get('ref_center_err_px','?')})")
    print(f"  timing: pre={sum(pre_l)/max(n,1):.2f}ms infer={sum(inf_l)/max(n,1):.1f}ms post={sum(post_l)/max(n,1):.3f}ms")
    print(f"  -> {results_csv}\n  -> {args.comparison_csv}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=4000000)   # fast image upload (firmware USART1 @ 4 Mbaud)
    sub = p.add_subparsers(dest="mode", required=True)

    pi = sub.add_parser("img", help="feed images from disk")
    src = pi.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="folder of .jpg/.png images")
    src.add_argument("--image-list", help="text file of image paths, one per line "
                     "(e.g. software/ball_detection/splits/val.txt)")
    pi.add_argument("--image-root", default="datasets/BALL",
                    help="root the --image-list paths resolve against (spl/->SPLDataset, our/->OurDataset)")
    pi.add_argument("--limit", type=int, default=0,
                    help="process at most N images (0 = all); handy for a quick timing run")
    pi.add_argument("--chunk-delay", type=float, default=0.0,
                    help="seconds to sleep between IMG_CHUNKs; try 0.002 if the board "
                         "drops bytes on the upload (UART has no flow control)")
    pi.add_argument("--layout", choices=["chw", "hwc"], default="chw",
                    help="pixel layout sent to the NPU input buffer: chw=planar "
                         "(matches the NCHW model, default), hwc=interleaved (A/B test)")

    pb = sub.add_parser("bench", help="benchmark over a val set: metrics + timing")
    bsrc = pb.add_mutually_exclusive_group(required=True)
    bsrc.add_argument("--images", help="folder of images")
    bsrc.add_argument("--image-list", help="text file of image paths (e.g. splits/val.txt)")
    pb.add_argument("--image-root", default="datasets/BALL")
    pb.add_argument("--limit", type=int, default=0)
    pb.add_argument("--chunk-delay", type=float, default=0.0)
    pb.add_argument("--conf", type=float, default=0.45,
                    help="operating confidence for P/R/F1 -- 0.45 == BallDetector_N6 firmware "
                         "CONF_THR (was 0.5). NMS(0.25)/max_det(8) already match N6 on-device.")
    pb.add_argument("--iou", type=float, default=0.5, help="matching IoU for a TP")
    pb.add_argument("--model-name", required=True)
    pb.add_argument("--out-dir", required=True)
    pb.add_argument("--comparison-csv",
                    default="software/final_models/ball_detection/model_comparison.csv")
    pb.add_argument("--dump-boxes", default="",
                    help="write the board's per-image boxes to this CSV (for ONNX A/B diff)")

    pc = sub.add_parser("cam", help="live camera view")
    pc.add_argument("--period-ms", type=int, default=0,
                    help="0 = free-run, otherwise frame period")

    args = p.parse_args()
    args.mode = args.mode  # already set by subparser
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "img":
        return mode_img(args)
    if args.mode == "bench":
        return mode_bench(args)
    if args.mode == "cam":
        return mode_cam(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
