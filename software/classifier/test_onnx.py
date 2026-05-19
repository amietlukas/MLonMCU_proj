"""
test_onnx.py - host-side ONNX evaluator (CPU).

Mirrors firmware/Host/host.py end-to-end EXCEPT:
  - no MCU/serial: inference runs locally with onnxruntime
  - no timing: pre_ms / infer_ms / post_ms / total_ms are left blank in the CSV
  - fast: tight inference loop, runs the full HAGRID test set by default

Produces, for the chosen --model tag:
  <run>/metrics/results_<tag>.csv
  <run>/metrics/confusion_matrix_<tag>.csv
  <run>/metrics/metrics_<tag>.csv
  ... and updates the matching row of firmware/Host/model_comparison.csv
  (all other rows in that CSV are preserved unchanged).

Static info (params, MACC, weights_KB, acts_KB) is sourced from the same
hardcoded STATIC_INFO table that firmware/Host/compare_models.py uses.
Tags absent from the table get blank cells (same behaviour as compare_models.py).

Usage:
    python software/classifier/test_onnx.py --model bignet_fp32
    python software/classifier/test_onnx.py --model bignet_int8 --n-samples 5000
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


REPO_ROOT = Path("/mnt/core/MLonMCU_proj")

# Match firmware/Host/host.py exactly.
DATASET_PATH = REPO_ROOT / "datasets" / "HAGRID" / "hagrid_full_qqvga_resize" / "test"
MODEL_W, MODEL_H, MODEL_C = 160, 120, 1
CLASS_NAMES = ["palm", "rock", "pinkie", "one", "fist", "others"]
CLASS_TO_IDX = {n: i for i, n in enumerate(CLASS_NAMES)}

MODEL_RUNS = {
    "smallnet_fp32":      "software/classifier/runs/smallnet_greyscale-20260504-184641_BASELINE_SMALL",
    "smallnet_int8":      "software/classifier/runs/smallnet_greyscale-20260504-184641_BASELINE_SMALL",
    "bignet_fp32":        "software/classifier/runs/bignet-20260505-163101_BASELINE",
    "bignet_int8":        "software/classifier/runs/bignet-20260505-163101_BASELINE",
    "bignet_pruned_fp32": "software/classifier/runs/bignet_pruned-20260505-225640_BASELINE_PRUNED",
    "bignet_pruned_int8": "software/classifier/runs/bignet_pruned-20260505-225640_BASELINE_PRUNED",
}

# Same source-of-truth as firmware/Host/compare_models.py.
# Values come from `stedgeai analyze --target stm32u5` on each ONNX.
STATIC_INFO = {
    "smallnet_fp32":      {"params":  93446, "macc": 184_666_598, "weights_B": 373_784, "acts_B": 707_108},
    "smallnet_int8":      {"params":  93446, "macc": 183_668_710, "weights_B":  95_160, "acts_B": 184_100},
    "bignet_fp32":        {"params": 190182, "macc":  72_482_390, "weights_B": 760_728, "acts_B": 353_252},
    "bignet_int8":        {"params": 190182, "macc":  71_911_382, "weights_B": 192_232, "acts_B":  93_888},
    "bignet_pruned_fp32": {"params":  17417, "macc":   7_614_225, "weights_B":  69_668, "acts_B": 109_976},
    "bignet_pruned_int8": {"params":  17417, "macc":   7_437_357, "weights_B":  18_052, "acts_B":  27_640},
}

MODEL_COMPARISON_CSV = REPO_ROOT / "firmware" / "Host" / "model_comparison.csv"
COMPARISON_FIELDS = [
    "model", "n_samples",
    "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "pre_ms_mean", "infer_ms_mean", "post_ms_mean",
    "infer_ms_p50", "infer_ms_p95",
    "params", "macc", "flash_KB", "ram_KB",
    "energy_mJ",
    "results_csv",
]


def find_onnx(run_dir: Path, tag: str) -> Path:
    suffix = "_int8_qdq_op13.onnx" if tag.endswith("_int8") else "_fp32_op13.onnx"
    onnx_dir = run_dir / "checkpoints" / "onnx"
    cands = sorted(onnx_dir.glob(f"*{suffix}"))
    if not cands:
        sys.exit(f"[ERROR] no ONNX matching *{suffix} in {onnx_dir}")
    return cands[0]


def load_dataset_paths():
    samples = []
    for cls in CLASS_NAMES:
        folder = DATASET_PATH / cls
        for img_path in sorted(glob.glob(str(folder / "*.jpg"))):
            samples.append((img_path, CLASS_TO_IDX[cls]))
    return samples


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fmt(v, spec=""):
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:
            return ""
        return format(v, spec)
    return str(v)


def update_comparison_csv(tag: str, n_samples: int, acc: float,
                          macro_p: float, macro_r: float, macro_f1: float,
                          results_csv_path: Path) -> None:
    """Replace just the row for `tag` in model_comparison.csv. Other rows untouched.

    Timing-related columns (pre/infer/post and energy) are left blank — they
    are only meaningful for firmware-measured runs from firmware/Host/host.py.
    """
    st = STATIC_INFO.get(tag, {})
    new_row = {
        "model":           tag,
        "n_samples":       str(n_samples),
        "accuracy":        fmt(acc, ".4f"),
        "macro_precision": fmt(macro_p, ".4f"),
        "macro_recall":    fmt(macro_r, ".4f"),
        "macro_f1":        fmt(macro_f1, ".4f"),
        "pre_ms_mean":     "",
        "infer_ms_mean":   "",
        "post_ms_mean":    "",
        "infer_ms_p50":    "",
        "infer_ms_p95":    "",
        "params":          fmt(st.get("params")),
        "macc":            fmt(st.get("macc")),
        "flash_KB":        fmt(st["weights_B"] / 1024, ".1f") if st.get("weights_B") else "",
        "ram_KB":          fmt(st["acts_B"] / 1024, ".1f")    if st.get("acts_B")    else "",
        "energy_mJ":       "",
        "results_csv":     str(results_csv_path),
    }

    rows = []
    replaced = False
    if MODEL_COMPARISON_CSV.exists():
        with MODEL_COMPARISON_CSV.open() as f:
            for r in csv.DictReader(f):
                if r.get("model") == tag:
                    rows.append(new_row)
                    replaced = True
                else:
                    rows.append(r)
    if not replaced:
        rows.append(new_row)

    MODEL_COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_COMPARISON_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPARISON_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COMPARISON_FIELDS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_RUNS.keys()),
                        help="Model variant tag. Selects run dir and ONNX file.")
    parser.add_argument("--onnx", default=None,
                        help="Override the auto-located ONNX path.")
    parser.add_argument("--n-samples", type=int, default=0,
                        help="If >0, draw N random samples WITH replacement (matches host.py "
                             "N_INFERENCES). Default 0 = full test set, each image once (shuffled).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = REPO_ROOT / MODEL_RUNS[args.model]
    out_dir = run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.model
    results_csv   = out_dir / f"results_{tag}.csv"
    confusion_csv = out_dir / f"confusion_matrix_{tag}.csv"
    metrics_csv   = out_dir / f"metrics_{tag}.csv"

    onnx_path = Path(args.onnx) if args.onnx else find_onnx(run_dir, tag)
    print(f"[INFO] model = {tag}")
    print(f"[INFO] onnx  = {onnx_path}")
    print(f"[INFO] out   = {out_dir}")

    samples = load_dataset_paths()
    print(f"[INFO] loaded {len(samples)} images from {DATASET_PATH}")
    random.seed(args.seed)
    if args.n_samples and args.n_samples > 0:
        plan = [random.choice(samples) for _ in range(args.n_samples)]
        print(f"[INFO] running {len(plan)} random samples (with replacement)")
    else:
        plan = list(samples)
        random.shuffle(plan)
        print(f"[INFO] running full test set: {len(plan)} images")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    inp_name = inp.name
    if tuple(inp.shape) != (1, 1, MODEL_H, MODEL_W):
        sys.exit(f"[ERROR] unexpected input shape {inp.shape}, expected (1,1,{MODEL_H},{MODEL_W})")

    correct = 0
    total = 0
    all_gt: list[int] = []
    all_pred: list[int] = []

    f_csv = results_csv.open("w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["image", "gt", "pred", "confidence",
                     "pre_ms", "infer_ms", "post_ms", "total_ms", "correct"])

    print_every = max(1, len(plan) // 20)
    for i, (img_path, gt) in enumerate(plan):
        arr = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
        # ONNX expects NCHW [1,1,120,160] float in [0,255] (normalize is fused).
        x = arr.reshape(1, 1, MODEL_H, MODEL_W)
        logits = sess.run(None, {inp_name: x})[0][0]
        pred = int(np.argmax(logits))
        conf = float(softmax(logits)[pred])

        is_correct = int(pred == gt)
        correct += is_correct
        total += 1
        all_gt.append(gt)
        all_pred.append(pred)

        writer.writerow([os.path.basename(img_path), CLASS_NAMES[gt], CLASS_NAMES[pred],
                         f"{conf:.4f}", "", "", "", "", is_correct])

        if (i + 1) % print_every == 0 or (i + 1) == len(plan):
            print(f"  {i+1}/{len(plan)}  acc={correct/total:.4f}")

    f_csv.close()
    print(f"[INFO] results   -> {results_csv}")

    if total == 0:
        return

    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for g, p in zip(all_gt, all_pred):
        cm[g][p] += 1

    with confusion_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GT \\ Pred"] + CLASS_NAMES)
        for i, row in enumerate(cm):
            w.writerow([CLASS_NAMES[i]] + row.tolist())

    precisions, recalls, f1s, supports = [], [], [], []
    for i in range(n):
        tp = cm[i][i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1); supports.append(support)

    macro_p  = float(np.mean(precisions))
    macro_r  = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    acc = correct / total

    with metrics_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for i in range(n):
            w.writerow([CLASS_NAMES[i],
                        f"{precisions[i]:.4f}", f"{recalls[i]:.4f}",
                        f"{f1s[i]:.4f}", supports[i]])
        w.writerow([])
        w.writerow(["macro_avg",
                    f"{np.mean(precisions):.4f}", f"{np.mean(recalls):.4f}",
                    f"{macro_f1:.4f}", total])
        w.writerow(["accuracy", "", "", f"{acc:.4f}", total])

    print(f"[INFO] confusion -> {confusion_csv}")
    print(f"[INFO] metrics   -> {metrics_csv}")
    print(f"[RESULT] {tag}: n={total}  acc={acc:.4f}  "
          f"P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}")

    update_comparison_csv(tag, total, acc, macro_p, macro_r, macro_f1, results_csv)
    print(f"[INFO] model_comparison.csv row '{tag}' updated -> {MODEL_COMPARISON_CSV}")


if __name__ == "__main__":
    main()
