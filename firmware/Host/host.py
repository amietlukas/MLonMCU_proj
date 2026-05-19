import os
import json
import serial
import struct
import numpy as np
from PIL import Image
from glob import glob
import random
import time
import csv
import argparse

# =========================
# CONFIG
# =========================
PORT = "/dev/ttyACM0"   # adjust (Linux)
BAUD = 921600
CPU_FREQ_HZ = 160_000_000  # 160 MHz (MSI 4MHz * PLLN 80 / PLLR 2)

DATASET_PATH = "/mnt/core/MLonMCU_proj/datasets/HAGRID/hagrid_full_qqvga_resize/test"
ANNOT_PATH   = "/mnt/core/MLonMCU_proj/datasets/HAGRID/hagrid_full_qqvga_resize/annotations/test"

# Model input is landscape 160x120 (W x H), 1-channel grayscale, post-letterbox.
MODEL_W = 160
MODEL_H = 120
MODEL_C = 1

# Number of inferences to run.
#   None  -> full test set, each image exactly once (shuffled)
#   int N -> N random samples (with replacement)
N_INFERENCES = None

CLASS_NAMES = ["palm", "rock", "pinkie", "one", "fist", "others"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Known model run directories. Pick one of these tags with --model.
MODEL_RUNS = {
    "smallnet_fp32":     "software/classifier/runs/smallnet_greyscale-20260504-184641_BASELINE_SMALL",
    "smallnet_int8":     "software/classifier/runs/smallnet_greyscale-20260504-184641_BASELINE_SMALL",
    "bignet_fp32":       "software/classifier/runs/bignet-20260505-163101_BASELINE",
    "bignet_int8":       "software/classifier/runs/bignet-20260505-163101_BASELINE",
    "bignet_pruned_fp32": "software/classifier/runs/bignet_pruned-20260505-225640_BASELINE_PRUNED",
    "bignet_pruned_int8": "software/classifier/runs/bignet_pruned-20260505-225640_BASELINE_PRUNED",
}
REPO_ROOT = "/mnt/core/MLonMCU_proj"


# =========================
# LOAD IMAGE PATHS
# =========================
def load_dataset_paths():
    samples = []
    for cls in CLASS_NAMES:
        folder = os.path.join(DATASET_PATH, cls)
        images = glob(os.path.join(folder, "*.jpg"))
        for img_path in images:
            samples.append((img_path, CLASS_TO_IDX[cls]))
    return samples


# =========================
# LOAD IMAGE
# =========================
def load_image(path):
    """Grayscale, native QQVGA resize (already 160x120 in dataset)."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8).flatten()


# =========================
# HANDSHAKE
# =========================
def wait_for(ser, keyword):
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("[MCU]", line)
            if line == keyword:
                return True
            if line == "STOP":
                return False


# =========================
# INFERENCE
# =========================
def run_inference(ser, img_flat):
    print("[HOST] Waiting for MCU...", flush=True)
    if not wait_for(ser, "READY_IN"):
        return None

    time.sleep(0.01)  # let MCU enter HAL_UART_Receive before we blast data

    byte_buffer = img_flat.tobytes()
    print(f"[HOST] Transmitting {len(byte_buffer)} bytes...", flush=True)
    ser.write(byte_buffer)
    ser.flush()
    print("[HOST] Image transmitted. Waiting for MCU...", flush=True)

    if not wait_for(ser, "READY_OUT"):
        return None

    out_data = ser.read(24)
    if len(out_data) < 24:
        return None

    pred_class, confidence, t_pre, t_infer, t_post, t_all = struct.unpack("<ifIIII", out_data)

    t_pre_ms   = t_pre   / CPU_FREQ_HZ * 1000
    t_infer_ms = t_infer / CPU_FREQ_HZ * 1000
    t_post_ms  = t_post  / CPU_FREQ_HZ * 1000
    t_all_ms   = t_all   / CPU_FREQ_HZ * 1000

    return pred_class, confidence, t_pre_ms, t_infer_ms, t_post_ms, t_all_ms


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_RUNS.keys()),
                        help="Active model variant on the MCU. Determines the output run dir + CSV tag.")
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = os.path.join(REPO_ROOT, MODEL_RUNS[args.model])
    out_dir = os.path.join(run_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.model
    results_csv   = os.path.join(out_dir, f"results_{tag}.csv")
    confusion_csv = os.path.join(out_dir, f"confusion_matrix_{tag}.csv")
    metrics_csv   = os.path.join(out_dir, f"metrics_{tag}.csv")

    print(f"[HOST] model={args.model}  out_dir={out_dir}")
    print(f"[HOST] writing: {results_csv}")

    random.seed(args.seed)
    samples = load_dataset_paths()
    print(f"[HOST] Loaded {len(samples)} images")

    if N_INFERENCES is None:
        plan = list(samples)
        random.shuffle(plan)
        print(f"[HOST] running full test set: {len(plan)} images")
    else:
        plan = [random.choice(samples) for _ in range(int(N_INFERENCES))]
        print(f"[HOST] running {len(plan)} random samples (N_INFERENCES={N_INFERENCES})")

    ser = serial.Serial(args.port, BAUD, timeout=5)
    time.sleep(2)
    ser.reset_input_buffer()
    ser.write(b"\x01")
    ser.flush()

    correct = 0
    total = 0
    all_gt, all_pred = [], []

    csvfile = open(results_csv, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(["image", "gt", "pred", "confidence", "pre_ms", "infer_ms", "post_ms", "total_ms", "correct"])

    for i, (img_path, gt_label) in enumerate(plan):
        print(f"\n=== {i+1}/{len(plan)} ===")
        print("Image:", os.path.basename(img_path))

        img = load_image(img_path)
        res = run_inference(ser, img)
        if res is None:
            break

        pred, conf, t_pre, t_inf, t_post, t_all = res
        all_gt.append(gt_label)
        all_pred.append(pred)

        is_correct = int(pred == gt_label)
        if is_correct:
            correct += 1
        total += 1

        print(f"GT: {CLASS_NAMES[gt_label]}")
        print(f"Pred: {CLASS_NAMES[pred]} (Conf: {conf:.2f})")
        print(f"Timings - Pre: {t_pre:.2f}ms, Infer: {t_inf:.2f}ms, Post: {t_post:.2f}ms, Total: {t_all:.2f}ms")
        print(f"Accuracy so far: {correct}/{total} = {correct/total:.3f}")

        writer.writerow([os.path.basename(img_path), CLASS_NAMES[gt_label], CLASS_NAMES[pred],
                         f"{conf:.4f}", f"{t_pre:.3f}", f"{t_inf:.3f}", f"{t_post:.3f}", f"{t_all:.3f}", is_correct])
        csvfile.flush()

    csvfile.close()
    ser.close()
    print(f"\nResults saved to {results_csv}")

    if total == 0:
        return

    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for gt, pr in zip(all_gt, all_pred):
        cm[gt][pr] += 1

    with open(confusion_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GT \\ Pred"] + CLASS_NAMES)
        for i, row in enumerate(cm):
            w.writerow([CLASS_NAMES[i]] + row.tolist())

    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        precisions, recalls, f1s, supports = [], [], [], []
        for i in range(n):
            tp = cm[i][i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            support = cm[i, :].sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            precisions.append(precision); recalls.append(recall); f1s.append(f1); supports.append(support)
            w.writerow([CLASS_NAMES[i], f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", support])

        w.writerow([])
        w.writerow(["macro_avg", f"{np.mean(precisions):.4f}", f"{np.mean(recalls):.4f}", f"{np.mean(f1s):.4f}", total])
        w.writerow(["accuracy", "", "", f"{correct/total:.4f}", total])

    print(f"\n{'='*40}\nFINAL RESULTS ({total} samples)\n{'='*40}")
    print(f"Accuracy: {correct}/{total} = {correct/total:.3f}")
    print(f"Confusion matrix saved to {confusion_csv}")
    print(f"Metrics saved to {metrics_csv}")

    header = "GT \\ Pred"
    print(f"\n{header:>10}", end="")
    for name in CLASS_NAMES:
        print(f"{name:>8}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:>10}", end="")
        for val in row:
            print(f"{val:>8}", end="")
        print()

    print(f"\n{'class':>10} {'prec':>8} {'recall':>8} {'f1':>8} {'support':>8}")
    for i in range(n):
        print(f"{CLASS_NAMES[i]:>10} {precisions[i]:>8.3f} {recalls[i]:>8.3f} {f1s[i]:>8.3f} {supports[i]:>8}")
    print(f"{'macro_avg':>10} {np.mean(precisions):>8.3f} {np.mean(recalls):>8.3f} {np.mean(f1s):>8.3f} {total:>8}")


if __name__ == "__main__":
    main()
