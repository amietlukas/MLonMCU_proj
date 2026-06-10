#!/usr/bin/env python3
"""
host_u5.py — viewer for the Classifier_U5 hand-gesture firmware.

The U5 board free-runs: it captures a QQVGA grayscale frame, classifies it
on-device, and streams the image + result over its UART (ST-LINK VCP). This
script is receive-only — it just listens, prints the prediction, and (by
default) shows the grayscale frame with the predicted gesture overlaid.

Per frame the MCU sends:
    "FRAME\\r\\n" + 19200 bytes grayscale (H=120, W=160) + 24 bytes "<ifIIII"

Usage:
    python3 host_u5.py --port /dev/ttyACM0          # window + console
    python3 host_u5.py --port /dev/ttyACM0 --no-preview   # console only

Tip: if you Ctrl+C and re-launch and it hangs at "looking for FRAME header",
press the board's NRST button so it restarts its stream cleanly.
"""

import argparse
import struct
import time

import numpy as np
import serial

# ---- protocol / model constants (must match the U5 firmware) ----
BAUD        = 921600
CPU_FREQ_HZ = 160_000_000          # for cycle-count -> ms
MODEL_W, MODEL_H = 160, 120
FRAME_BYTES  = MODEL_W * MODEL_H   # 19200 grayscale bytes
RESULT_BYTES = 24                  # "<ifIIII"
HEADER       = b"FRAME\r\n"

CLASS_NAMES = ["palm", "rock", "pinkie", "one", "fist", "other"]


def is_log_line(s: str) -> bool:
    """True if s looks like a deliberate MCU text line (e.g. 'BOOT'), not
    image bytes that happened to contain a newline."""
    return bool(s) and len(s) <= 120 and all(32 <= ord(c) < 127 for c in s)


def drain_to_header(ser, timeout=10.0):
    """Read one byte at a time until HEADER appears as a contiguous suffix.
    Byte-by-byte is deliberate: chunked reads would swallow the payload tail
    after the match and leave us misaligned. Prints any clean log lines seen
    along the way; image bytes are silently dropped so they can't corrupt the
    terminal."""
    deadline = time.monotonic() + timeout
    buf, line = bytearray(), bytearray()
    while time.monotonic() < deadline:
        b = ser.read(1)
        if not b:
            continue
        buf.append(b[0])
        if len(buf) > len(HEADER):
            del buf[0]
        line.append(b[0])
        if b[0] == 0x0A:
            s = line.decode(errors="ignore").strip()
            if is_log_line(s):
                print("[MCU]", s, flush=True)
            line.clear()
        if len(buf) == len(HEADER) and bytes(buf) == HEADER:
            return True
    return False


def read_exact(ser, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def read_frame_after_header(ser):
    img_bytes = read_exact(ser, FRAME_BYTES)
    if img_bytes is None:
        return None
    res_bytes = read_exact(ser, RESULT_BYTES)
    if res_bytes is None:
        return None
    pred, conf, t_pre, t_inf, t_post, t_all = struct.unpack("<ifIIII", res_bytes)
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(MODEL_H, MODEL_W)
    return {
        "img":        img,
        "pred":       pred,
        "conf":       conf,
        "t_infer_ms": t_inf / CPU_FREQ_HZ * 1000.0,
        "t_all_ms":   t_all / CPU_FREQ_HZ * 1000.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=BAUD)
    ap.add_argument("--no-preview", action="store_true",
                    help="console only, no image window")
    args = ap.parse_args()

    use_cv = False
    if not args.no_preview:
        try:
            import cv2
            cv2.namedWindow("U5 inference", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("U5 inference", 640, 480)
            use_cv = True
        except ImportError:
            print("[HOST] opencv-python not installed; running console-only", flush=True)

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    time.sleep(0.5)
    ser.reset_input_buffer()

    print("[HOST] looking for FRAME header (press NRST if this hangs)...", flush=True)
    if not drain_to_header(ser, timeout=10.0):
        print("[HOST] no clean FRAME header in 10s — press the board's NRST and retry.",
              flush=True)
        ser.close()
        return

    print("[HOST] synced.", flush=True)
    idx, last, fps = 0, time.time(), 0.0
    try:
        while True:
            if idx > 0 and not drain_to_header(ser, timeout=5.0):
                print("[HOST] lost sync — exiting", flush=True)
                break
            frame = read_frame_after_header(ser)
            if frame is None:
                print("[HOST] short read mid-frame — exiting", flush=True)
                break

            now = time.time()
            dt = now - last
            fps = (0.9 * fps + 0.1 / dt) if (fps and dt > 0) else (1.0 / dt if dt > 0 else 0.0)
            last = now

            p = frame["pred"]
            cls = CLASS_NAMES[p] if 0 <= p < len(CLASS_NAMES) else f"?{p}"
            print(f"[{idx:5d}] pred={cls:<7s} conf={frame['conf']:.2f}  "
                  f"infer={frame['t_infer_ms']:.1f}ms  total={frame['t_all_ms']:.1f}ms  "
                  f"link_fps={fps:4.1f}", flush=True)

            if use_cv:
                import cv2
                vis = cv2.cvtColor(frame["img"], cv2.COLOR_GRAY2BGR)
                vis = cv2.resize(vis, (640, 480), interpolation=cv2.INTER_NEAREST)
                cv2.putText(vis, f"{cls} ({frame['conf']*100:.0f}%)  {fps:.1f}fps",
                            (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("U5 inference", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            idx += 1
    except KeyboardInterrupt:
        print("\n[HOST] interrupted", flush=True)
    finally:
        ser.close()
        if use_cv:
            import cv2
            cv2.destroyAllWindows()
        print(f"[HOST] processed {idx} frames", flush=True)


if __name__ == "__main__":
    main()
