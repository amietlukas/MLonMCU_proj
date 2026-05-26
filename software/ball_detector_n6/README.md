# ball_detector_n6 — host side

PC-side tooling to drive the **STM32 Nucleo-N657X0-Q + B-CAMS-IMX** ball
detector. Counterpart to `firmware/BallDetector_N6/`.

Mirrors the pattern of `software/classifier` + `firmware/Host/host.py`: the
board doesn't have an LCD, so we stream pictures and/or detections over UART
and visualise on the PC.

## What lives here

| File | Purpose |
| --- | --- |
| `prepare_model.py` | Validate the trained int8 ONNX, copy it to a stable path, and emit the `user_neuralart.json` sanity report. Run before `stedgeai generate`. |
| `yolo_decode.py`   | NumPy port of `ball_detection/src/training/decode.py`. Decodes raw `p8/p16/p32` head outputs into xyxy boxes + scores. |
| `host.py`          | Serial driver. Two modes: `--mode img` feeds JPEG/PNG files from disk; `--mode cam` displays live camera frames from the board. |
| `protocol.md`      | UART framing (host ↔ firmware). Source of truth shared with firmware. |

## Source model

Pulled from the existing training pipeline:

```
software/ball_detection/runs/20260521-195948-simota_diou_v2_pruned_30/exports/int8_ptq_qdq.onnx
```

- Input: `input` [1, 3, 480, 640] (NCHW float, QDQ-wrapped int8)
- Outputs: `p8 [1,5,60,80]`, `p16 [1,5,30,40]`, `p32 [1,5,15,20]`
- 5 channels per cell: `[tx, ty, tw, th, tobj]`
- Strides: 8 / 16 / 32

## Setup

Use the existing repo venv:

```
source /mnt/core/MLonMCU_proj/software/venv/bin/activate
pip install pyserial opencv-python  # if missing
```

`onnx`, `numpy`, `Pillow` are already in the existing venv.

## Typical flow

1. `python prepare_model.py` — sanity-check the ONNX, drop a copy at
   `model/int8_ptq_qdq.onnx` for `stedgeai` to consume.
2. (In `firmware/BallDetector_N6/`) `scripts/stedgeai_compile.sh` →
   generates `network.c/.h` + weight blob.
3. (In `firmware/BallDetector_N6/`) build + flash via CubeIDE.
4. `python host.py --mode img --images <dir>` — push test images, verify
   bbox output matches host-side prediction.
5. `python host.py --mode cam` — live view from B-CAMS-IMX.
