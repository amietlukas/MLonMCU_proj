#!/usr/bin/env python3
"""
host_n6.py — UVC viewer for the BallDetector_N6 firmware.

The N6 board runs fully standalone: camera → NPU ball detection → bounding
boxes drawn on the frame → streamed out as a **USB/UVC video device**. The host
does not send anything and does not run any inference — it just opens the UVC
camera device and shows the video the board produces.

Plug the board's USB OTG port (CN8) into the PC. When the UVC device enumerates,
this script opens it and displays the live, annotated feed. Press 'q' to quit.

Usage:
    python3 host_n6.py                 # auto-detect the UVC device
    python3 host_n6.py --device 2      # force /dev/video2 (or camera index 2)
    python3 host_n6.py --list          # list candidate video devices and exit

Requires opencv-python (`pip install opencv-python`).
"""

import argparse
import sys
import time

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required: pip install opencv-python")


def list_devices(max_index=10):
    """Probe camera indices 0..max_index and report which ones open."""
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append((i, w, h, ok))
        cap.release()
    return found


def open_device(index, timeout_s=10.0):
    """Open the UVC device, waiting up to timeout_s for it to enumerate and
    deliver a first frame (it may not be ready the instant the board boots)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
        time.sleep(0.5)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=None,
                    help="video device index (e.g. 0 for /dev/video0). "
                         "Default: first device that opens.")
    ap.add_argument("--list", action="store_true",
                    help="list candidate video devices and exit")
    args = ap.parse_args()

    if args.list:
        devs = list_devices()
        if not devs:
            print("no video devices found")
        for i, w, h, ok in devs:
            print(f"  index {i}: {w}x{h} {'(reads frames)' if ok else '(opens, no frame)'}")
        return 0

    index = args.device
    if index is None:
        devs = list_devices()
        cams = [i for i, _, _, ok in devs if ok]
        if not cams:
            print("[HOST] no UVC video device found. Is the board plugged into CN8 "
                  "and booted from flash? Try --list.", file=sys.stderr)
            return 2
        index = cams[-1]   # the board usually enumerates after built-in webcams
        print(f"[HOST] auto-selected video device index {index} "
              f"(use --device to override, --list to see all)")

    print(f"[HOST] opening UVC device {index} (waiting for the board to stream)...")
    cap = open_device(index)
    if cap is None:
        print(f"[HOST] device {index} never delivered a frame. Check the board is "
              f"running from flash and the USB cable is on CN8.", file=sys.stderr)
        return 2

    print("[HOST] streaming — press 'q' in the window to quit.")
    cv2.namedWindow("BallDetector_N6 (UVC)", cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[HOST] lost the UVC stream (board reset/unplugged?) — exiting")
                break
            cv2.imshow("BallDetector_N6 (UVC)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n[HOST] interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
