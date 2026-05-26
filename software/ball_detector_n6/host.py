"""Serial driver for the BallDetector_N6 firmware.

Two modes:
  --mode img  : push test images from a folder, draw the returned bboxes.
  --mode cam  : ask the board to stream camera frames + detections, display live.

Wire format is defined in protocol.md.
"""
from __future__ import annotations

import argparse
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
MODEL_W, MODEL_H, MODEL_C = 640, 480, 3


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


def read_frame(ser: serial.Serial, timeout_s: float = 5.0) -> tuple[int, bytes]:
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


def load_image_rgb888_hwc(path: Path) -> bytes:
    img = Image.open(path).convert("RGB").resize((MODEL_W, MODEL_H), Image.BILINEAR)
    return np.array(img, dtype=np.uint8).tobytes()  # HWC packed


def hello(ser: serial.Serial) -> dict:
    send_frame(ser, T_HELLO)
    while True:
        ftype, payload = read_frame(ser)
        if ftype == T_INFO:
            fw_ver, w, h, c, n_out = struct.unpack("<HHHBB", payload[:8])
            return {"fw_ver": fw_ver, "w": w, "h": h, "c": c, "n_outputs": n_out}
        if ftype == T_LOG:
            print(f"[fw] {payload.decode(errors='ignore').rstrip()}")


def push_image(ser: serial.Serial, img_bytes: bytes) -> dict:
    send_frame(ser, T_IMG_BEGIN, struct.pack("<HHBB", MODEL_W, MODEL_H, MODEL_C, 1))
    for i in range(0, len(img_bytes), CHUNK):
        chunk = img_bytes[i:i + CHUNK]
        send_frame(ser, T_IMG_CHUNK, struct.pack("<H", i // CHUNK) + chunk)
    send_frame(ser, T_IMG_END)

    while True:
        ftype, payload = read_frame(ser)
        if ftype == T_LOG:
            print(f"[fw] {payload.decode(errors='ignore').rstrip()}")
            continue
        if ftype == T_INFER_DEC:
            inf_us, n = struct.unpack("<IH", payload[:6])
            boxes = []
            off = 6
            for _ in range(n):
                x1, y1, x2, y2, s = struct.unpack("<5f", payload[off:off + 20])
                boxes.append((x1, y1, x2, y2, s))
                off += 20
            return {"inference_us": inf_us, "boxes": boxes}


def mode_img(args: argparse.Namespace) -> int:
    image_dir = Path(args.images)
    paths = sorted(p for p in image_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if not paths:
        print(f"no images in {image_dir}", file=sys.stderr)
        return 2

    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        info = hello(ser)
        print(f"connected, fw=0x{info['fw_ver']:04x}, model {info['w']}x{info['h']}x{info['c']}")
        for p in paths:
            buf = load_image_rgb888_hwc(p)
            r = push_image(ser, buf)
            print(f"{p.name:40s}  {r['inference_us']/1000:7.1f} ms  "
                  f"{len(r['boxes'])} det")
            for x1, y1, x2, y2, s in r["boxes"]:
                print(f"    score={s:.3f}  [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=921600)
    sub = p.add_subparsers(dest="mode", required=True)

    pi = sub.add_parser("img", help="feed images from disk")
    pi.add_argument("--images", required=True, help="folder of .jpg/.png images")

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
    if args.mode == "cam":
        return mode_cam(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
