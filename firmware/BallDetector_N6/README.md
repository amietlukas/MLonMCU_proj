# BallDetector_N6 — on-board YOLO ball detection (B-CAMS-IMX / IMX335)

YOLO ball detector on the **STM32 Nucleo-N657X0-Q + B-CAMS-IMX**, no LCD:
the IMX335 feeds the DCMIPP NN pipe, the Neural-ART NPU runs a pruned int8
STYOLO-nano at **384×288**, the three raw YOLO heads are decoded + NMS'd in
C on-chip, and the decoded boxes (and optionally JPEG frames) stream to the
PC over UART. The host script (`firmware/Host/host_balldetector_n6.py`) draws
the boxes live.

Built from a **fresh CubeMX project** that pulls X-CUBE-AI from the locally
installed pack — deliberately **not** layered on ST's modelzoo app (that path
died on a middleware version clash, see "Why LL_ATON-direct" below).

## What's in this folder (tracked — survives clean rebuilds)

| File | Purpose |
|------|---------|
| `Core/Inc/constants.h`        | Detection thresholds, UART config, protocol caps. Model dims come from `yolo_postproc.h` / `network.h` (not duplicated). |
| `Core/Inc/uart_protocol.h`    | Wire-format contract (frame types, CRC, framer API). |
| `Core/Src/uart_protocol.c`    | HELLO/INFO/IMG/CAM/INFER framing over the VCP UART. |
| `Core/Inc/balldetector_app.h` | App entry points + `g_frame_ready` flag. |
| `Core/Src/balldetector_app.c` | Main loop: frame → NPU (LL_ATON) → decode → UART. |
| `Core/Inc/yolo_postproc.h`    | Single-class STYOLO decode + NMS API. |
| `Core/Src/yolo_postproc.c`    | Decoder impl (cross-checked against `yolo_decode.py`). |
| `Core/Src/main.c`             | CubeMX-generated: clocks, DCMIPP, UART, AI runtime init. |
| `scripts/prepare_model.py`    | ONNX sanity-check + copy into `model/`. |
| `scripts/patch_onnx_input_size.py` | Re-shape the FCN ONNX to a new fixed input size. |
| `scripts/stedgeai_compile.sh` | ONNX → `X-CUBE-AI/App/network.{c,h}` + weight blob. |
| `scripts/build.sh`            | Headless CubeIDE build → `Debug/*.elf,*.bin`. |
| `scripts/flash.sh`            | Sign + flash app + weights to external OctoFlash. |
| `scripts/all.sh`              | prepare → compile → build → flash. |
| `tests/test_postproc.py`      | C-vs-Python decoder cross-check (must PASS before flashing). |
| `model/int8_ptq_qdq.onnx`     | The deployed quantized model. |

What this folder **does not** contain (CubeMX-managed / generated — `.gitignore`d):

- `Drivers/` — STM32N6 HAL + BSP (DCMIPP, IMX335 sensor, xSPI2)
- `Middlewares/` — the X-CUBE-AI **LL_ATON** runtime
- `X-CUBE-AI/App/network.{c,h}` + `network_atonbuf.xSPI2.raw` (weights),
  emitted by `stedgeai_compile.sh`
- `Debug/` build outputs, `st_ai_ws/` stedgeai scratch

## Why LL_ATON-direct (and not stai_network)

On this install (X-CUBE-AI **10.2.0**, ST Edge AI Core v2.2.0), the N6
Neural-ART code generator emits **`network.{c,h}` (the LL_ATON C API)** and
**not** `stai_network.{c,h}` — verified empirically with `stedgeai generate
--c-api st-ai`. So the app calls **LL_ATON directly** (`#include
"ll_aton_runtime.h"`, the `LL_ATON_DEFAULT_*` I/O descriptors in
`network.h`). This keeps a single, consistent X-CUBE-AI version end-to-end —
the previous modelzoo-based attempt failed because its bundled middleware
(atonn-v1.1.3) clashed with the installed runtime (atonn-v1.1.1).

## Setup steps (one time)

### 1. Generate the project in CubeMX (in STM32CubeIDE)

There is no standalone CubeMX here; drive it from CubeIDE's wizard.
**File → New → STM32 Project → Board Selector → NUCLEO-N657X0-Q** (load the
default board config so pinmux + clock tree + BSP are auto-filled). Set the
project name to `BallDetector_N6` and the location to `firmware/` so output
lands in-place. Then:

- **Software Packs → X-CUBE-AI**: enable the **Neural-ART** runtime. Add a
  network named `network`, load `model/int8_ptq_qdq.onnx`, target
  **STM32N6 / Neural-ART**, memory pool **`stm32n6.mpool`** (profile
  `n6-allmems-O3` — on-chip only, no hyperRAM), **input uint8 / output
  int8**. (CubeMX runs `stedgeai` for you; `scripts/stedgeai_compile.sh`
  re-rolls it after a model swap.)
- **Multimedia → DCMIPP** + the **IMX335** sensor on the B-CAMS-IMX FPC —
  CubeMX routes the MIPI-CSI2 pins. Configure the NN downscale pipe to
  **384×288 RGB888** (sensor is 2592×1944 4:3, so no crop/letterbox).
- **Connectivity → USART** routed to the **ST-LINK VCP**, **921600 8N1**.
- Motor/servo PWM: **deferred** — not configured yet. When added, claim
  timer channels that don't collide with DCMIPP/UART (pinmux only).
- **Project Manager → Toolchain: STM32CubeIDE**. Generate.

> Edits to CubeMX-generated files (step 2) go inside `USER CODE` blocks so
> they survive future `.ioc` regenerations.

### 2. Wire the app into the X-CUBE-AI lifecycle

In `X-CUBE-AI/App/app_x-cube-ai.c` (USER CODE blocks):

```c
/* USER CODE BEGIN includes */
#include "balldetector_app.h"
/* USER CODE END includes */

void MX_X_CUBE_AI_Init(void)    { /* USER CODE BEGIN 5 */ balldetector_init();    /* USER CODE END 5 */ }
void MX_X_CUBE_AI_Process(void) { /* USER CODE BEGIN 6 */ balldetector_run();     /* USER CODE END 6 */ }  /* never returns */
```

The DCMIPP frame-ready callback flips `g_frame_ready` (declared in
`balldetector_app.h`); the capture loop consumes it.

### 3. Generate the network code

```
scripts/stedgeai_compile.sh        # -> X-CUBE-AI/App/network.{c,h} + weights
```

### 4. Build (headless — no GUI window)

```
scripts/build.sh                   # -> Debug/BallDetector_N6.{elf,bin}
```

### 5. Sign + flash to external OctoFlash (xSPI2)

```
scripts/flash.sh                   # app(signed)@0x70000000, weights@0x70380000
```

### 6. Run the host

```
source software/venv/bin/activate
python firmware/Host/host_balldetector_n6.py --port /dev/ttyACM0 cam
```

Or the whole chain: `scripts/all.sh`.

## Memory budget (fits entirely on-chip)

From `stedgeai generate` on the deployed model (profile `n6-allmems-O3`,
`stm32n6.mpool`):

```
input        : uint8(1x3x288x384)  324.00 KiB  QLinear(s=1/255, zp=0)   → activations
output p8    : int8 (1x5x36x48)      8.44 KiB  QLinear(s=0.256579876, zp=41)
output p16   : int8 (1x5x18x24)      2.11 KiB  QLinear(s=0.204285681, zp=0)
output p32   : int8 (1x5x9x12)        540 B    QLinear(s=0.146335885, zp=0)
params       : 1,013,727 items
weights      : 1,029,089 B (1004.97 KiB, 1 segment)  →  xSPI2 OctoFlash
activations  : 1,396,224 B (1.33 MiB, 3 segments)    →  on-chip SRAM
```

Per-bank placement (no external PSRAM — the Nucleo-N657X0-Q has none):

| Bank      | Capacity | Used      | %     |
|-----------|---------:|----------:|------:|
| flexMEM / cpuRAM1  |  —     |     0 B   |  0%   |
| cpuRAM2   | 1.000 MB | 594.0 kB  | 58.0% |
| npuRAM3   |  448 kB  |     0 B   |  0%   (free) |
| npuRAM4   |  448 kB  | 324.0 kB  | 72.3% |
| npuRAM5   |  448 kB  | 445.5 kB  | 99.4% |
| npuRAM6   |  448 kB  |     0 B   |  0%   (free for CPU / future growth) |
| octoFlash | 112 MB   | 1004.97 kB|  0.9% (weights only) |

The per-head `scale`/`zero_point` above are baked into `yolo_postproc.c`'s
`HEADS[]` table. Re-run `scripts/stedgeai_compile.sh` (it prints the same
summary) after any re-quantization and update that table + `yolo_decode.py`
to match, then re-run the cross-check.

## Post-processing

The NPU emits **raw YOLO head tensors only** — decode + NMS run on the CPU:

- **`Core/{Inc,Src}/yolo_postproc.{c,h}`** — single-class STYOLO decode + NMS.
  Self-contained (`<math.h>`, `<stdint.h>`, `<string.h>`); pass the three
  int8 head buffers from the LL_ATON output and get `yolo_box_t[]`
  (conf 0.50, NMS IoU 0.25, ≤8 boxes). Each of the **2268** grid cells
  (36·48 + 18·24 + 9·12) is rejected with a single int8 compare before any
  float math, so only a handful reach dequant + sigmoid + exp.
- **`tests/test_postproc.py`** — builds `yolo_postproc.c` into a `.so`, runs
  it via ctypes on a seeded synthetic input, and asserts the boxes match
  `firmware/Host/yolo_decode.py` within 1e-3. **Currently PASSES (8/8).**
  Re-run after any change to the C decoder or its quant table:

  ```
  python firmware/BallDetector_N6/tests/test_postproc.py
  ```

Where NMS runs decides the wire format:

| postproc runs | UART frame  | bytes/inference | notes |
|---------------|-------------|-----------------|-------|
| Firmware (default) | `INFER_DEC` | ~60–160 (≤8 boxes) | real-time CAM mode |
| Host (debug)       | `INFER_RAW` | ~11,340 (8.6k+2.2k+0.5k) | compare on-board vs host decode |

## Common gotchas

- **Weights address mismatch**: `stedgeai`'s mdesc models xSPI2 at
  `0x71000000`, but the board flash map programs weights at `0x70380000`
  (app at `0x70000000`). The CubeMX-generated linker script reconciles the
  two — confirm `flash.sh`'s `WEIGHTS_ADDR` matches it before trusting a
  flash.
- **LL_ATON I/O via cached pointers**: read the input/output buffer
  addresses from the `network.h` descriptors (`LL_ATON_DEFAULT_IN_1_*` /
  `OUT_*`) each run; don't cache a stale pointer. See
  `feedback_xcubeai_inputs_get_gotcha`.
- **Garbage detections after a retrain**: the per-head scale/zp in
  `yolo_postproc.c` and `yolo_decode.py` are hard-coded. Re-run
  `stedgeai_compile.sh`, copy the new quant params into both decoders, and
  confirm `test_postproc.py` passes before flashing.
