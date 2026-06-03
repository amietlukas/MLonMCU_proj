#!/usr/bin/env python3
"""
Receive a raw camera frame dumped by BallDetector_0D over the ST-Link VCP and
save it (plus rotated/transposed variants) so we can pick the correct, distortion
-free orientation before baking a transform into firmware.

Firmware wire framing:
    b"\\nIMG:" <W:u16le> <H:u16le> <C:u8> <W*H*C raw RGB888 bytes> b"\\nEND\\n"

Usage:
    python3 recv_frame_n6.py [--port /dev/ttyACM0] [--baud 115200] [--trigger]
The firmware also emits one frame automatically a moment after boot; pass
--trigger to actively request a fresh frame by sending 'c'.
"""
import argparse, os, sys, time

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial is required:  pip install pyserial")

try:
    from PIL import Image
except ImportError:
    Image = None  # raw .bin is still written; PNGs are skipped

MARKER = b"\nIMG:"
OUTDIR = "/tmp/n6_frames"


def read_exact(ser, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise TimeoutError(f"timed out after {len(buf)}/{n} bytes")
        buf += chunk
    return bytes(buf)


def find_marker(ser):
    """Slide a window over the byte stream until MARKER appears."""
    window = b""
    while True:
        b = ser.read(1)
        if not b:
            continue
        window = (window + b)[-len(MARKER):]
        if window == MARKER:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--trigger", action="store_true",
                    help="send 'c' to request a fresh frame")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    ser = serial.Serial(args.port, args.baud, timeout=2)
    print(f"[*] {args.port} @ {args.baud}  -> {OUTDIR}")

    if args.trigger:
        time.sleep(0.2)
        ser.reset_input_buffer()
        ser.write(b"c")
        print("[*] trigger 'c' sent")

    print("[*] waiting for frame marker (this can take ~30 s per frame @ 115200)...")
    find_marker(ser)
    w = int.from_bytes(read_exact(ser, 2), "little")
    h = int.from_bytes(read_exact(ser, 2), "little")
    c = read_exact(ser, 1)[0]
    print(f"[*] header: {w}x{h}x{c}")
    if not (0 < w <= 4096 and 0 < h <= 4096 and c in (1, 3)):
        sys.exit(f"[!] implausible header {w}x{h}x{c} — out of sync, retry")

    n = w * h * c
    t0 = time.time()
    data = bytearray()
    last = time.time()
    while len(data) < n and (time.time() - last) < 5.0:
        chunk = ser.read(min(65536, n - len(data)))
        if chunk:
            data += chunk
            last = time.time()
    if len(data) < n:
        print(f"[!] partial: {len(data)}/{n} bytes after {time.time()-t0:.1f}s "
              f"(rendering what arrived, padded)")
        data += bytes(n - len(data))
    else:
        print(f"[*] received {n} bytes in {time.time()-t0:.1f}s")
    data = bytes(data)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(OUTDIR, f"frame_{stamp}_{w}x{h}.bin")
    with open(raw_path, "wb") as f:
        f.write(data)
    print(f"[+] raw : {raw_path}")

    if Image is None:
        print("[!] Pillow not installed; skipping PNGs (pip install pillow)")
        return

    img = Image.frombytes("RGB", (w, h), data)
    variants = {
        "asis":          img,
        "rot90_cw":      img.transpose(Image.ROTATE_270),  # PIL ROTATE_270 == 90 CW
        "rot90_ccw":     img.transpose(Image.ROTATE_90),
        "transpose":     img.transpose(Image.TRANSPOSE),   # true matrix transpose
        "flip_v":        img.transpose(Image.FLIP_TOP_BOTTOM),
        "flip_h":        img.transpose(Image.FLIP_LEFT_RIGHT),
    }
    for name, im in variants.items():
        p = os.path.join(OUTDIR, f"frame_{stamp}_{name}.png")
        im.save(p)
        print(f"[+] {name:10s}: {p}  ({im.width}x{im.height})")

    print("\nOpen the PNGs and tell me which orientation looks upright + undistorted.")
    print(f"  xdg-open {OUTDIR}")


if __name__ == "__main__":
    main()
