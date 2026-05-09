"""
host_cam.py — terminal viewer for Classifier_Cam (B-CAMS-OMV).

The MCU streams a captured-and-inferred frame on every iteration:
   "FRAME\r\n" + 19200 bytes grayscale (HxW=120x160) + 24 bytes "<ifIIII"

Important: press the NRST button on the board before launching this script
if you've Ctrl+C'd a previous run. Otherwise the MCU is still mid-stream
and the host has no clean way to find the FRAME header.
"""

import argparse
import os
import struct
import sys
import time

import numpy as np
import serial


# ============================================================
# CONFIG
# ============================================================
PORT        = "/dev/ttyACM0"
BAUD        = 921600
CPU_FREQ_HZ = 160_000_000

MODEL_W = 160
MODEL_H = 120
FRAME_BYTES = MODEL_W * MODEL_H            # 19200 grayscale bytes
RESULT_BYTES = 24                          # "<ifIIII"
HEADER = b"FRAME\r\n"

CLASS_NAMES = ["palm", "rock", "pinkie", "one", "fist", "others"]


# ============================================================
# Robust framing
# ============================================================
def is_printable_line(s):
    """Heuristic — true if s looks like a deliberate MCU log line, not
    image bytes that happened to contain a \n."""
    if not s or len(s) > 120:
        return False
    return all(32 <= ord(c) < 127 for c in s)


def drain_to_header(ser, timeout=10.0):
    """Read bytes one at a time until we see HEADER as a contiguous
    suffix of our rolling buffer. Returns True on success, False on
    timeout. Reading byte-by-byte is critical: if we read in chunks,
    the unprocessed tail of the chunk after we match HEADER is silently
    lost to pyserial's internal buffer, leaving us byte-misaligned for
    the subsequent read_exact() of the image+result payload. Prints
    only well-formed log lines we picked up along the way; image bytes
    are silently swallowed so they can't corrupt the terminal."""
    deadline = time.monotonic() + timeout
    buf = bytearray()
    line = bytearray()
    while time.monotonic() < deadline:
        b = ser.read(1)
        if not b:
            continue
        b = b[0]
        buf.append(b)
        if len(buf) > len(HEADER):
            del buf[0]
        line.append(b)
        if b == 0x0A:
            s = line.decode(errors="ignore").strip()
            if is_printable_line(s):
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


def read_one_frame_after_header(ser):
    img_bytes = read_exact(ser, FRAME_BYTES)
    if img_bytes is None or len(img_bytes) != FRAME_BYTES:
        return None
    res_bytes = read_exact(ser, RESULT_BYTES)
    if res_bytes is None or len(res_bytes) != RESULT_BYTES:
        return None
    pred, conf, t_pre, t_inf, t_post, t_all = struct.unpack("<ifIIII", res_bytes)
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(MODEL_H, MODEL_W)
    return {
        "img":        img,
        "pred":       pred,
        "conf":       conf,
        "t_pre_ms":   t_pre  / CPU_FREQ_HZ * 1000,
        "t_infer_ms": t_inf  / CPU_FREQ_HZ * 1000,
        "t_post_ms":  t_post / CPU_FREQ_HZ * 1000,
        "t_all_ms":   t_all  / CPU_FREQ_HZ * 1000,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--baud", type=int, default=BAUD)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--save", default=None,
                        help="Save each frame as PNG into this directory")
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    time.sleep(0.5)
    ser.reset_input_buffer()
    # If MCU is still in BOOT handshake, this unblocks it. If it's already
    # mid-stream, it gets ignored (MCU isn't reading). Either way fine.
    ser.write(b"\x01")
    ser.flush()

    print("[HOST] draining and looking for FRAME header (press NRST if this hangs)...",
          flush=True)
    if not drain_to_header(ser, timeout=10.0):
        print("[HOST] couldn't find a clean FRAME header in 10s. "
              "Press the black NRST button on the board and retry.",
              flush=True)
        ser.close()
        return

    print("[HOST] synced.", flush=True)

    fig = ax = im = txt = None
    if not args.no_preview:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(np.zeros((MODEL_H, MODEL_W), dtype=np.uint8),
                       cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        txt = ax.set_title("starting…", fontsize=11)
        fig.tight_layout()

    save_dir = None
    if args.save:
        save_dir = args.save
        os.makedirs(save_dir, exist_ok=True)
        print(f"[HOST] saving frames to {save_dir}/", flush=True)

    frame_idx = 0
    last_t = time.time()
    fps = 0.0

    try:
        while True:
            # First frame: header was already consumed by drain_to_header.
            # Subsequent frames: re-find header (MCU sends it before each).
            if frame_idx > 0:
                if not drain_to_header(ser, timeout=5.0):
                    print("[HOST] lost sync — exiting", flush=True)
                    break

            frame = read_one_frame_after_header(ser)
            if frame is None:
                print("[HOST] short read mid-frame — exiting", flush=True)
                break

            now = time.time()
            dt = now - last_t
            fps = (0.9 * fps + 0.1 / dt) if (fps and dt > 0) else (1.0 / dt if dt > 0 else 0.0)
            last_t = now

            cls = (CLASS_NAMES[frame["pred"]] if 0 <= frame["pred"] < len(CLASS_NAMES)
                   else f"?{frame['pred']}")
            print(
                f"[{frame_idx:5d}] pred={cls:<7s} conf={frame['conf']:.2f}  "
                f"pre={frame['t_pre_ms']:.1f}ms  "
                f"inf={frame['t_infer_ms']:.1f}ms  "
                f"post={frame['t_post_ms']:.2f}ms  "
                f"total={frame['t_all_ms']:.1f}ms  "
                f"link_fps={fps:4.1f}",
                flush=True,
            )

            if im is not None:
                im.set_data(frame["img"])
                txt.set_text(f"{cls} ({frame['conf']*100:.1f}%) — {frame['t_all_ms']:.0f} ms")
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            if save_dir:
                from PIL import Image
                Image.fromarray(frame["img"]).save(
                    f"{save_dir}/frame_{frame_idx:06d}_{cls}_c{int(frame['conf']*100):02d}.png")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[HOST] interrupted by user")
    finally:
        ser.close()
        print(f"[HOST] processed {frame_idx} frames")


if __name__ == "__main__":
    main()
