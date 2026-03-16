"""
Real-time hand gesture classification from webcam using a trained checkpoint.

Usage:
    python webcam_classify.py --checkpoint runs/<run_id>/checkpoints/best.pt

Keys:
    q / ESC  –  quit
"""

import argparse
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np

from src.data import CLASS_NAMES, LetterboxSquare, _parse_norm
from src.model import BaselineCNN


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Webcam gesture classifier")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to best.pt checkpoint")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device: 'cuda' or 'cpu'")
    p.add_argument("--cam-id", type=int, default=0,
                   help="OpenCV camera index")
    p.add_argument("--conf-threshold", type=float, default=0.4,
                   help="Min confidence to display a prediction")
    return p.parse_args()


# ── build the same eval transform used during training ───────────────────────
def build_inference_transform(cfg: dict) -> T.Compose:
    """Reproduce the eval-time transform from a saved config snapshot."""
    size = int(cfg["data"]["input_size"])
    pp = cfg.get("preprocess", {})
    fill = int(pp.get("fill", 0))
    in_channels = int(cfg["model"]["in_channels"])
    norm_cfg = pp.get("normalize", {})
    norm_mean, norm_std = _parse_norm(norm_cfg, in_channels)

    spatial = LetterboxSquare(size=size, fill=fill)

    if in_channels == 1:
        channel_tf = [T.Grayscale(num_output_channels=1)]
    else:
        channel_tf = []

    return T.Compose(
        channel_tf + [
            spatial,
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std),
        ]
    )


# ── overlay helpers ──────────────────────────────────────────────────────────
GESTURE_ACTIONS = {
    "palm":   "STOP",
    "rock":   "DRIVE FORWARD",
    "pinkie": "STEER RIGHT",
    "one":    "STEER LEFT",
    "fist":   "DRIVE BACKWARD",
    "others": "–",
}

BAR_COLORS = [
    (50, 205, 50),   # palm   – green
    (255, 140, 0),   # rock   – orange
    (255, 50, 50),   # pinkie – red
    (0, 191, 255),   # one    – blue
    (180, 105, 255), # fist   – pink
    (160, 160, 160), # others – grey
]


def draw_overlay(frame: np.ndarray, probs: list[float], pred_idx: int,
                 conf_threshold: float) -> np.ndarray:
    """Draw a confidence bar chart + predicted label on the frame."""
    h, w = frame.shape[:2]
    bar_x = 10
    bar_y_start = 30
    bar_h = 22
    bar_gap = 6
    max_bar_w = 220

    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
        y = bar_y_start + i * (bar_h + bar_gap)
        bw = int(prob * max_bar_w)
        color = BAR_COLORS[i] if i == pred_idx else (100, 100, 100)

        # background bar
        cv2.rectangle(frame, (bar_x, y), (bar_x + max_bar_w, y + bar_h),
                       (40, 40, 40), -1)
        # filled bar
        cv2.rectangle(frame, (bar_x, y), (bar_x + bw, y + bar_h), color, -1)
        # label
        cv2.putText(frame, f"{name} {prob:.0%}", (bar_x + max_bar_w + 8, y + bar_h - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # large prediction text
    conf = probs[pred_idx]
    if conf >= conf_threshold:
        label = CLASS_NAMES[pred_idx]
        action = GESTURE_ACTIONS.get(label, "")
        text = f"{label.upper()}  ({action})  {conf:.0%}"
        cv2.putText(frame, text, (bar_x, h - 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, BAR_COLORS[pred_idx], 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "low confidence", (bar_x, h - 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1, cv2.LINE_AA)

    return frame


# ── main loop ────────────────────────────────────────────────────────────────
@torch.no_grad()
def main():
    args = parse_args()

    # load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    # build model
    model = BaselineCNN(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[INFO] Loaded model from {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    print(f"[INFO] Device: {device}")

    # build transform
    transform = build_inference_transform(cfg)

    # open webcam
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.cam_id}")
    print("[INFO] Press 'q' or ESC to quit.")

    from PIL import Image

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB PIL -> transform -> batch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        x = transform(pil_img).unsqueeze(0).to(device)  # [1, C, H, W]

        # forward
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        pred_idx = int(logits.argmax(dim=1).item())

        # draw overlay on original frame
        frame = draw_overlay(frame, probs, pred_idx, args.conf_threshold)

        cv2.imshow("Gesture Classifier", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
