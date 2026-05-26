# BallDetector_N6

YOLO ball detector on **STM32 Nucleo-N657X0-Q + B-CAMS-IMX**, no LCD —
results stream to PC over UART (`software/ball_detector_n6/host.py`).

Approach: take ST's reference object-detection app for STM32N6 as the
base (it already wires DCMIPP camera capture + Neural-ART NPU inference),
strip the LCD display layer, replace it with the UART protocol defined
in `software/ball_detector_n6/protocol.md`.

## You need to do (one-time)

1. **Install X-CUBE-AI 10.x with the N6 Neural-ART add-on.**
   - CubeIDE → Help → Manage Embedded Software Packages → STMicroelectronics
   - Install `X-CUBE-AI` ≥ 10.0.0
   - Confirm `stedgeai --version` works on the CLI and supports
     `--target stm32n6 --st-neural-art`.

2. **Install STM32CubeProgrammer + the N6 signing tool.**
   - N6 boots from external flash; weights/binary need to be signed with
     `STM32_SigningTool_CLI` (bundled with CubeProgrammer).
   - Update the `SIGNING_TOOL` path at the top of
     `scripts/flash_n6_signed.sh`.

3. **Clone ST's reference app into this directory.**
   ```
   cd firmware/BallDetector_N6
   git clone https://github.com/STMicroelectronics/stm32ai-modelzoo-services.git
   cp -r stm32ai-modelzoo-services/object_detection/STM32N6/Application/* .
   ```
   *(Repo URL / path may differ across releases — open the modelzoo
   readme and look for the `STM32N6` object-detection application. Pick
   the variant for **Nucleo-N657X0-Q** if multiple exist.)*

4. **Strip the LCD/display layer.**
   In the cloned app, remove or stub out:
   - `Src/display_*.c`, LTDC init in `Src/main.c`
   - any `BSP_LCD_*` or `osd_*` calls in the inference loop
   - the OSD/bbox-drawing code (we draw on PC instead)

5. **Add the UART transport.**
   - Enable a USART/UART instance routed to the ST-LINK VCP at 921600 8N1.
   - Add a `comm/uart_protocol.c` implementing the framing in
     `software/ball_detector_n6/protocol.md` (HELLO/IMG/CAM/INFER frames).
   - In the main inference loop, replace the LCD draw with one of:
     - **IMG mode**: poll UART for `IMG_BEGIN/CHUNK/END`, copy the
       received RGB888 buffer straight into the NPU input tensor (skip
       the camera path).
     - **CAM mode**: keep the camera path, JPEG-encode the captured
       frame (use the hardware JPEG codec on N6), and send
       `CAM_FRAME` + `INFER_DEC` per inference.

6. **Generate the network code.** From this directory:
   ```
   scripts/stedgeai_compile.sh
   ```
   Produces `X-CUBE-AI/App/network.c/.h` and the weight blob.

7. **Build + flash from the terminal** (no GUI). Either run the whole chain:
   ```
   scripts/all.sh
   ```
   or step through it manually:
   ```
   scripts/build_n6.sh         # headless CubeIDE build -> Debug/*.elf,*.bin
   scripts/flash_n6_signed.sh  # sign + flash to external OSPI
   ```

8. **Run the host.** From the repo root:
   ```
   source software/venv/bin/activate
   python software/ball_detector_n6/host.py --port /dev/ttyACM0 cam
   ```

## Terminal-only build details

The headless build uses STM32CubeIDE's bundled Eclipse with the CDT
`headlessbuild` application — no window opens. On this machine the
toolchain lives inside the CubeIDE install (no separate CubeProgrammer
install needed):

```
export CUBEIDE=/opt/st/stm32cubeide_2.1.1/headless-build.sh
export STEDGEAI=$HOME/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Utilities/linux/stedgeai
export CUBEPROG_BIN=/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_2.2.400.202601091506/tools/bin
export SIGNING_TOOL=$CUBEPROG_BIN/STM32_SigningTool_CLI
export PROG_TOOL=$CUBEPROG_BIN/STM32_Programmer_CLI
export EXT_LOADER=$CUBEPROG_BIN/ExternalLoader/MX25UM51245G_STM32N6570-NUCLEO.stldr
```

stedgeai for N6 uses `--target stm32n6 --memory-pool <file.mpool>`.
Pick the layout via the `MPOOL` env var (default: ext PSRAM for
activations, which we need for the 921 KB 640×480 input). Override:
```
MPOOL=$HOME/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/scripts/N6_scripts/my_mpools/stm32n6.mpool \
  scripts/stedgeai_compile.sh
```

Raw commands behind the scripts:
```
# stedgeai (verified working on this model)
$STEDGEAI generate \
  --target stm32n6 \
  --model software/ball_detector_n6/model/int8_ptq_qdq.onnx \
  --name ball_n6 \
  --memory-pool $HOME/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/scripts/N6_scripts/my_mpools/stm32n6__extRam.mpool \
  --binary --address 0x71000000 \
  --output firmware/BallDetector_N6/X-CUBE-AI/App

# build (no GUI)
$CUBEIDE -data /tmp/cubeide_ws_ball_n6 \
  -import firmware/BallDetector_N6 \
  -cleanBuild BallDetector_N6/Debug

# sign + flash (app at 0x70000000, weights at 0x71000000)
$SIGNING_TOOL -bin Debug/BallDetector_N6.bin -nk -t ssbl -hv 2.3 \
              -o Debug/BallDetector_N6-signed.bin
$PROG_TOOL -c port=SWD mode=HOTPLUG -el $EXT_LOADER \
           -w Debug/BallDetector_N6-signed.bin       0x70000000 \
           -w X-CUBE-AI/App/ball_n6_data.bin         0x71000000 \
           -hardRst
```

## What stedgeai analyze reports for this model

```
input        : uint8(1x3x480x640)  900.00 KiB  scale=1/255  zp=0      → activations
output p8    : uint8(1x5x60x80)     23.44 KiB  scale=0.234  zp=149
output p16   : uint8(1x5x30x40)      5.86 KiB  scale=0.236  zp=124
output p32   : uint8(1x5x15x20)      1.46 KiB  scale=0.221  zp=155
macc         : 1,732,443,223  (~1.7 GMAC)
weights      : 997 KiB (1 segment)
activations  : 1.95 MiB (2 segments) — xSPI1 hyperRAM + AXISRAM3 npuRAM3
```

Two host-side implications baked into `yolo_decode.py` later:
- inputs are **uint8 NCHW** scaled by 1/255 (i.e. /255 of RGB888)
- outputs are **uint8** — dequantize with the per-head `(u8 - zp) * scale`
  before running the existing sigmoid/exp decode

## Model

Pulled from `software/ball_detection`:
- 1.1 MB int8 QDQ ONNX, 640×480 RGB input, 1 class ("Ball"), strides 8/16/32.
- Pre-flight check: `python software/ball_detector_n6/prepare_model.py`
  (copies the chosen checkpoint to `software/ball_detector_n6/model/`).

## Memory budget (rough — verify with stedgeai analyze)

- Input tensor: 640·480·3·1 B = 921 600 B (fits in PSRAM, not in NPU SRAM)
- Outputs: 60·80·5 + 30·40·5 + 15·20·5 = 31 500 B (int8) → trivial
- Weights: 1.1 MB → external flash, signed binary
- Activations: ~MB scale, will need PSRAM. Configure in `scripts/user_neuralart.json`.

## Post-processing

The model emits **raw YOLO head tensors only** — no decode/NMS on the
NPU. We do that ourselves:

- **`app/yolo_postproc.{c,h}`** — single-class STYOLO decode + NMS in C.
  Self-contained (only `<math.h>`, `<stdint.h>`, `<string.h>`); pass the
  three uint8 head buffers from `ai_run`/`stai_run` and get `yolo_box_t[]`.
  Per-head quant params (scale/zero_point) are hard-coded from the
  `stedgeai analyze` report — re-update them if the model is re-quantized.
- **Fast path**: each cell is rejected with a single uint8 compare
  before any float math. For the typical busy scene only ~tens of
  cells survive to the dequant + sigmoid + exp stage out of 6300.
- **`app/test_postproc.py`** — cross-check harness. Builds the C file
  into a `.so`, calls it via ctypes on a seeded synthetic input, runs
  the same input through `software/ball_detector_n6/yolo_decode.py`,
  asserts boxes match within 1e-3. Currently **PASSES** with 8/8 boxes.
  Re-run after any change to the C decoder or its quant table.

Where the NMS runs (firmware vs host) decides the wire format:

| Where postproc runs | UART frame | Bytes / inference | Notes |
| --- | --- | --- | --- |
| Firmware (default)  | `INFER_DEC` | ~60–160 (≤8 boxes) | Real-time CAM mode |
| Host                | `INFER_RAW` | ~31 500 (24k+6k+1.5k) | ~30 ms at 921600 baud — caps to ~30 fps |

Default plan: firmware always runs `yolo_postprocess()` and sends
`INFER_DEC`. Keep `INFER_RAW` as a debug path (toggle via a future
config frame) so we can compare on-board decode against host decode on
real captures, not just synthetic data.

## Layout (what survives clean rebuilds)

```
firmware/BallDetector_N6/
  README.md                 <- this file
  scripts/
    stedgeai_compile.sh     <- ONNX -> NPU code (verified working)
    build_n6.sh             <- headless CubeIDE build
    flash_n6_signed.sh      <- sign + flash via STM32CubeProgrammer
    all.sh                  <- model prep -> stedgeai -> build -> flash
  app/
    yolo_postproc.h         <- single-class YOLO decode + NMS API
    yolo_postproc.c         <- impl (verified against host yolo_decode.py)
    test_postproc.py        <- cross-check harness (PASS)
  (Application/, X-CUBE-AI/, Drivers/, Middlewares/  cloned from ST app)
```
