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

# =========================
# CONFIG
# =========================
PORT = "/dev/ttyACM0"   # adjust (Linux)
BAUD = 921600
CPU_FREQ_HZ = 160_000_000  # 160 MHz (MSI 4MHz * PLLN 80 / PLLR 2)
RESULTS_CSV        = "results.csv"
CONFUSION_CSV      = "confusion_matrix.csv"
METRICS_CSV        = "metrics.csv"

DATASET_PATH = "/mnt/core/MLonMCU_proj/datasets/HAGRID/hagrid_balanced_classification/test"
ANNOT_PATH   = "/mnt/core/MLonMCU_proj/datasets/HAGRID/hagrid_balanced_classification/annotations/test"

MODEL_SIZE = 128

N_INFERENCES = 1000

CLASS_NAMES = ["palm", "rock", "pinkie", "one", "fist", "others"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


# =========================
# LOAD ANNOTATIONS
# =========================
def load_annotations():
    """Load bounding box annotations for all classes."""
    annots = {}
    for cls in CLASS_NAMES:
        annot_file = os.path.join(ANNOT_PATH, f"{cls}.json")
        with open(annot_file) as f:
            data = json.load(f)
        for img_id, entry in data.items():
            annots[img_id] = entry["bboxes"][0]  # first hand bbox
    return annots


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
# LETTERBOX TO 128x128 (matches training exactly)
# =========================
def letterbox_square(img, size, fill=0):
    """Resize preserving aspect ratio, pad to square."""
    w, h = img.size
    s = size / max(w, h)
    nw, nh = int(round(w * s)), int(round(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    padded = Image.new("RGB", (size, size), (fill, fill, fill))
    padded.paste(img, ((size - nw) // 2, (size - nh) // 2))
    return padded


def load_image(path):
    """Load image, letterbox to 128x128, return as flat uint8 RGB HWC."""
    img = Image.open(path).convert("RGB")
    img = letterbox_square(img, MODEL_SIZE)
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

    # Wait for MCU to signal it's ready
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

    # Convert DWT cycle counts to milliseconds
    t_pre_ms   = t_pre   / CPU_FREQ_HZ * 1000
    t_infer_ms = t_infer / CPU_FREQ_HZ * 1000
    t_post_ms  = t_post  / CPU_FREQ_HZ * 1000
    t_all_ms   = t_all   / CPU_FREQ_HZ * 1000

    return pred_class, confidence, t_pre_ms, t_infer_ms, t_post_ms, t_all_ms


# =========================
# MAIN
# =========================
def main():

    samples = load_dataset_paths()
    print(f"[HOST] Loaded {len(samples)} images")

    ser = serial.Serial(PORT, BAUD, timeout=5)
    time.sleep(2)  # wait for MCU to boot / USB CDC to stabilize
    ser.reset_input_buffer()  # drain any init messages from MCU

    correct = 0
    total = 0
    all_gt = []
    all_pred = []

    csvfile = open(RESULTS_CSV, "w", newline="")
    writer = csv.writer(csvfile)
    writer.writerow(["image", "gt", "pred", "confidence", "pre_ms", "infer_ms", "post_ms", "total_ms", "correct"])

    for i in range(N_INFERENCES):

        img_path, gt_label = random.choice(samples)

        print(f"\n=== {i+1} ===")
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
    print(f"\nResults saved to {RESULTS_CSV}")

    # === EVALUATION METRICS ===
    if total == 0:
        return

    n = len(CLASS_NAMES)

    # Confusion matrix (rows = ground truth, cols = predicted)
    cm = np.zeros((n, n), dtype=int)
    for gt, pr in zip(all_gt, all_pred):
        cm[gt][pr] += 1

    with open(CONFUSION_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GT \\ Pred"] + CLASS_NAMES)
        for i, row in enumerate(cm):
            w.writerow([CLASS_NAMES[i]] + row.tolist())

    # Per-class precision, recall, F1
    with open(METRICS_CSV, "w", newline="") as f:
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

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)

            w.writerow([CLASS_NAMES[i], f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", support])

        # Macro averages
        w.writerow([])
        w.writerow(["macro_avg", f"{np.mean(precisions):.4f}", f"{np.mean(recalls):.4f}", f"{np.mean(f1s):.4f}", total])
        w.writerow(["accuracy", "", "", f"{correct/total:.4f}", total])

    print(f"\n{'='*40}")
    print(f"FINAL RESULTS ({total} samples)")
    print(f"{'='*40}")
    print(f"Accuracy: {correct}/{total} = {correct/total:.3f}")
    print(f"\nConfusion matrix saved to {CONFUSION_CSV}")
    print(f"Metrics saved to {METRICS_CSV}")

    # Print confusion matrix to console
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

    # Print per-class metrics
    print(f"\n{'class':>10} {'prec':>8} {'recall':>8} {'f1':>8} {'support':>8}")
    for i in range(n):
        print(f"{CLASS_NAMES[i]:>10} {precisions[i]:>8.3f} {recalls[i]:>8.3f} {f1s[i]:>8.3f} {supports[i]:>8}")
    print(f"{'macro_avg':>10} {np.mean(precisions):>8.3f} {np.mean(recalls):>8.3f} {np.mean(f1s):>8.3f} {total:>8}")


if __name__ == "__main__":
    main()
