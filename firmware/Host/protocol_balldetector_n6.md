# UART protocol — BallDetector_N6 ↔ host.py

Baud: **921600 8N1**, no flow control. Port on the Nucleo is the ST-LINK VCP
(`/dev/ttyACM0` on Linux). Multi-byte fields are **little-endian**.

## Framing

All frames start with a 4-byte magic and a 1-byte type:

```
0xAA 0x55 0xA5 0x5A  <type>  <len:u32>  <payload[len]>  <crc16:u16>
```

`crc16` is CRC-16/CCITT-FALSE over `<type><len><payload>`.

## Frame types

| `type` | Direction | Name        | Payload |
| ------ | --------- | ----------- | --- |
| `0x01` | H → F     | `HELLO`     | empty — board replies with `INFO` |
| `0x02` | F → H     | `INFO`      | `fw_ver:u16, model_w:u16, model_h:u16, model_c:u8, n_outputs:u8` |
| `0x10` | H → F     | `IMG_BEGIN` | `width:u16, height:u16, channels:u8, fmt:u8` (fmt: 0=RGB888 planar NCHW, 1=RGB888 packed HWC) |
| `0x11` | H → F     | `IMG_CHUNK` | `chunk_idx:u16, bytes[]` — up to 4096 bytes payload |
| `0x12` | H → F     | `IMG_END`   | empty |
| `0x20` | F → H     | `INFER_RAW` | `inference_us:u32, p8_bytes[36*48*5*sizeof(int8)], p16_bytes[18*24*5*int8], p32_bytes[9*12*5*int8], scales+zero_points` |
| `0x21` | F → H     | `INFER_DEC` | `inference_us:u32, n_boxes:u16, box[n] = {x1:f32, y1:f32, x2:f32, y2:f32, score:f32}` (after NPU + on-board NMS) |
| `0x30` | H → F     | `CAM_START` | `period_ms:u16` (0 = free-run) |
| `0x31` | H → F     | `CAM_STOP`  | empty |
| `0x32` | F → H     | `CAM_FRAME` | `frame_idx:u32, prev_w:u16, prev_h:u16, rgb[prev_w*prev_h*3]` (RGB888 low-res preview) — immediately followed by a separate `INFER_DEC` frame whose boxes are in MODEL space (384×288) |
| `0xFE` | F → H     | `LOG`       | UTF-8 string |
| `0xFF` | both      | `ACK/NACK`  | `status:u8` (0=ok, !=0=error code) |

## Modes

- **IMG mode**: host sends `IMG_BEGIN` → `IMG_CHUNK*` → `IMG_END`. Board
  runs inference, replies with `INFER_DEC` (or `INFER_RAW` if host requested
  raw via a future config flag). Use this for accuracy validation against
  the eval set.
- **CAM mode** (implemented in `BallDetector_N6_cam`): host sends `CAM_START`.
  The board brings up the B-CAMS-IMX (IMX335) via the Camera Middleware,
  captures RGB888 frames on DCMIPP_PIPE1 (ISP debayer → 384×288), transposes
  HWC→CHW into the NPU input, runs the NPU, and per frame streams a `CAM_FRAME`
  (a small RGB preview, integer-downscaled from the model frame) followed by an
  `INFER_DEC` (boxes in model space). Host sends `CAM_STOP` to end. A full
  384×288×3 frame is 331 KB ≈ 3.6 s at 921600 baud, so only the downscaled
  preview is streamed (default 128×96 ≈ 2.5 fps; see `PREVIEW_SCALE` in
  `constants.h`). On-board NMS keeps `INFER_DEC` tiny.
