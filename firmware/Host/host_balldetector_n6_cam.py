"""Serial driver for the BallDetector_N6_cam firmware (live camera variant).

Two modes:
  img  : push test images from a folder, draw the returned bboxes (same as the
         non-cam build — kept for accuracy validation on the same board).
  cam  : ask the board to capture from the B-CAMS-IMX (IMX335), run the NPU, and
         stream a low-res RGB preview + decoded boxes per frame; display live.

CAM_FRAME wire format (this build): frame_idx:u32, prev_w:u16, prev_h:u16,
rgb[prev_w*prev_h*3] (RGB888), immediately followed by an INFER_DEC frame whose
boxes are in MODEL space (384x288). The board subsamples the model frame to the
preview to fit the 921600-baud link; see protocol_balldetector_n6.md.
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
               fmt: int = 0) -> dict:
    # fmt: 0 = planar CHW (matches the NCHW model), 1 = interleaved HWC
    send_frame(ser, T_IMG_BEGIN, struct.pack("<HHBB", MODEL_W, MODEL_H, MODEL_C, fmt))
    for i in range(0, len(img_bytes), CHUNK):
        chunk = img_bytes[i:i + CHUNK]
        send_frame(ser, T_IMG_CHUNK, struct.pack("<H", i // CHUNK) + chunk)
        if chunk_delay:
            time.sleep(chunk_delay)   # pace the upload if the board drops bytes (no flow ctrl)
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


def _read_infer_dec(ser: serial.Serial) -> dict:
    """Read frames until the INFER_DEC that follows a CAM_FRAME (skipping logs)."""
    while True:
        ftype, payload = read_frame(ser, timeout_s=2.0)
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


def mode_cam(args: argparse.Namespace) -> int:
    # Live display via matplotlib (cv2's Qt GUI is broken in some venvs:
    # "Cannot find font directory" / "Point size <= 0"). PIL handles --save.
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    show = not args.no_window
    fig = ax = im_artist = title = None
    if show:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_xticks([]); ax.set_yticks([])
        title = ax.set_title("starting…")

    fps, last_t = 0.0, time.time()
    # 12 s read timeout: a full 384x288 raw frame (diagnostic, scale 1) is
    # 331 KB ~= 3.6 s over UART; the small preview is well under this.
    with serial.Serial(args.port, args.baud, timeout=12) as ser:
        info = hello(ser)
        print(f"connected, fw=0x{info['fw_ver']:04x}, model {info['w']}x{info['h']}")
        print("streaming — close the window or Ctrl+C to stop")
        send_frame(ser, T_CAM_START, struct.pack("<H", args.period_ms))
        try:
            while True:
                if show and not plt.fignum_exists(fig.number):
                    break
                try:
                    ftype, payload = read_frame(ser, timeout_s=5.0)
                except TimeoutError:
                    print("[host] no CAM_FRAME within 5 s — board likely failed to start "
                          "the camera (see [fw] CAM:/i2c lines above). Exiting.", file=sys.stderr)
                    break
                except OSError as e:
                    # CRC/truncation (UART has no flow control) — drop + resync.
                    print(f"[host] frame desync ({e}) — resyncing", file=sys.stderr)
                    continue
                if ftype == T_LOG:
                    print(f"[fw] {payload.decode(errors='ignore').rstrip()}")
                    continue
                if ftype != T_CAM_FRAME:
                    continue

                frame_idx, prev_w, prev_h = struct.unpack("<IHH", payload[:8])
                rgb = np.frombuffer(payload[8:8 + prev_w * prev_h * 3], dtype=np.uint8)
                if rgb.size != prev_w * prev_h * 3:
                    print("[host] short preview payload — skipping", file=sys.stderr)
                    continue
                rgb = rgb.reshape(prev_h, prev_w, 3)          # RGB (matplotlib/PIL native)

                # The boxes for this frame arrive in the very next INFER_DEC.
                try:
                    det = _read_infer_dec(ser)
                except OSError as e:
                    print(f"[host] det desync ({e}) — resyncing", file=sys.stderr)
                    continue
                boxes, inf_us = det["boxes"], det["inference_us"]

                now = time.time(); dt = now - last_t
                fps = (0.9 * fps + 0.1 / dt) if (fps and dt > 0) else (1.0 / dt if dt > 0 else 0.0)
                last_t = now
                print(f"#{frame_idx:<6d} {inf_us/1000:6.1f} ms  {fps:4.1f} fps  {len(boxes)} det" +
                      "".join(f"  [{s:.2f}]" for *_, s in boxes))

                # Boxes are in MODEL space (384x288) -> preview pixels.
                sx, sy = prev_w / MODEL_W, prev_h / MODEL_H

                if show:
                    if im_artist is None:
                        im_artist = ax.imshow(rgb, interpolation="nearest")
                    else:
                        im_artist.set_data(rgb)
                    for p in list(ax.patches):
                        p.remove()
                    for x1, y1, x2, y2, s in boxes:
                        ax.add_patch(mpatches.Rectangle(
                            (x1 * sx, y1 * sy), (x2 - x1) * sx, (y2 - y1) * sy,
                            fill=False, edgecolor="red", linewidth=1.5))
                    title.set_text(f"#{frame_idx}  {inf_us/1000:.0f} ms  {fps:.1f} fps  {len(boxes)} det")
                    fig.canvas.draw_idle(); fig.canvas.flush_events()

                if save_dir:
                    up = Image.fromarray(rgb).resize(
                        (prev_w * args.scale, prev_h * args.scale), Image.NEAREST)
                    up.save(save_dir / f"cam_{frame_idx:06d}.png")
        except KeyboardInterrupt:
            print("\n[host] interrupted")
        finally:
            send_frame(ser, T_CAM_STOP)
            if show:
                import matplotlib.pyplot as plt
                plt.ioff(); plt.close("all")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--baud", type=int, default=921600)
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
    pi.add_argument("--viz-dir", default=str(Path(__file__).resolve().parent / "n6_viz"),
                    help="folder to save annotated <name>_mcu.png (GT green, pred red) "
                         "after each inference; set '' to disable")

    pc = sub.add_parser("cam", help="live camera view")
    pc.add_argument("--period-ms", type=int, default=0,
                    help="0 = free-run, otherwise frame period")
    pc.add_argument("--scale", type=int, default=4,
                    help="integer upscale of the preview for the display window (default 4)")
    pc.add_argument("--save", default=None,
                    help="folder to save each raw preview as cam_<idx>.png via PIL "
                         "(e.g. firmware/Host/n6_viz/cam) — reliable, no GUI needed")
    pc.add_argument("--no-window", action="store_true",
                    help="don't open the live matplotlib window (headless; use with --save)")

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
