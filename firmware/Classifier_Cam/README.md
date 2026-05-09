# Classifier_Cam — on-board capture + inference (B-CAMS-OMV / OV5640)

This is the camera-fed sibling of the `Classifier` project. The MCU
captures a frame from the OV5640 on the B-CAMS-OMV at **QQVGA
(160x120) YUV422** — the same resolution and luma representation
the `bignet_pruned` model was trained on (dataset:
`hagrid_full_qqvga_resize`). The Y channel of YUV422 is the
grayscale image, so the model input is a strided extract from the
frame buffer — no resize, no RGB→Y conversion. The MCU then runs
inference and streams the image + prediction to the host over UART.
The host script (`firmware/Host/host_cam.py`) prints a running
prediction line and (optionally) shows a live preview.

## What's in this folder

| File | Purpose |
|------|---------|
| `Core/Inc/constants.h`            | Model + camera defines (resolutions, quant, buffer sizes). |
| `Core/Inc/classifier_cam_app.h`   | App entry-point declarations + `g_frame_ready` flag. |
| `Core/Src/classifier_cam_app.c`   | Capture loop, RGB565→gray, AI inference, UART streaming. |
| `Core/Src/main.c`                 | Boots clocks, peripherals, AI, then jumps into the loop. |

What this folder **does not** contain (and you have to add yourself,
because they're board-/CubeMX-managed):

- `Drivers/STM32U5xx_HAL_Driver/{Inc,Src}/stm32u5xx_hal_dcmi*.{h,c}`
- `Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_camera.{c,h}`
- `Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_bus.{c,h}`
- `Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_conf_template.h` →
  rename to `b_u585i_iot02a_conf.h`
- `Drivers/BSP/Components/ov5640/*`
- `Drivers/BSP/Components/Common/camera.h`
- Generated `dcmi.c/.h`, `i2c.c/.h` from CubeMX

## Setup steps (one time)

### 1. Clone the existing project

In STM32CubeIDE: **File → Copy** the `Classifier` project, paste it as
`Classifier_Cam`. This gives you a working baseline (X-CUBE-AI already
configured, model weights already present, USART1 already on the
ST-LINK VCP).

### 2. Enable camera peripherals in CubeMX

Open `Classifier_Cam.ioc`, then:

- **Pinout & Configuration → Connectivity → DCMI**: enable as
  **Slave 8 bits External Synchro** (i.e. hardware sync via dedicated
  HSYNC/VSYNC pins, NOT embedded BT.656 codes). Pins for CN2 on
  B-U585I-IOT02A: PI6/5/7, PC11, PB6, PH12/10/9/8/5, PE0 — CubeMX
  routes them automatically.
- **Connectivity → I2C2**: enable. 100 kHz is fine for OV5640 SCCB.
- **System Core → DMA / GPDMA1 → Channel 12**: Standard request mode,
  request = `DCMI_PSSI`, peripheral→memory, normal mode, word/word,
  source increment fixed, dest incremented, burst length 1.
- **NVIC**: enable `GPDMA1 Channel 12 global interrupt`. The
  DCMI/PSSI IRQ does NOT need to be enabled here — BSP unmasks it
  programmatically inside `BSP_CAMERA_Init`.

Save and **regenerate code**. CubeMX will create `dcmi.c/h`,
`i2c.c/h`, `gpdma.c/h`, and the HAL files for DCMI under
`Drivers/STM32U5xx_HAL_Driver/`.

> **Heads-up**: regen wipes hand-edits made outside `USER CODE`
> blocks. The four source-file edits in step 5 below are all inside
> USER CODE blocks so they survive future regens; the BSP-side
> renames in step 4 sit in vendor files that you won't regenerate.

### 3. Drop in ST's BSP

From the **STM32CubeU5** firmware package
(`~/STM32Cube/Repository/STM32Cube_FW_U5_V*/Drivers/BSP/`), copy
into your project's `Drivers/BSP/`:

- `B-U585I-IOT02A/b_u585i_iot02a.[ch]`
- `B-U585I-IOT02A/b_u585i_iot02a_camera.[ch]`
- `B-U585I-IOT02A/b_u585i_iot02a_bus.[ch]`
- `B-U585I-IOT02A/b_u585i_iot02a_errno.h`
- `B-U585I-IOT02A/b_u585i_iot02a_conf_template.h`
  → rename to `b_u585i_iot02a_conf.h` and place in `Core/Inc/`
- `Components/Common/camera.h`
- `Components/ov5640/*` (sensor driver: `ov5640.c`, `ov5640.h`,
  `ov5640_reg.c/h`)

Then add include paths in **Project → Properties → C/C++ Build →
Settings → MCU GCC Compiler → Include paths (-I)**:

- `../Drivers/BSP/B-U585I-IOT02A`
- `../Drivers/BSP/Components/ov5640`
- `../Drivers/BSP/Components/Common`

### 4. Resolve BSP↔CubeMX name collisions

Both the BSP and CubeMX define `MX_DCMI_Init` and `MX_I2C2_Init` with
*different signatures* (BSP takes a handle pointer; CubeMX takes
`void`). The result is a multiple-definition link error.

Fix by renaming the BSP versions only (one-time, vendor-side edit):

```bash
sed -i 's/\bMX_I2C1_Init\b/BSP_MX_I2C1_Init/g;
        s/\bMX_I2C2_Init\b/BSP_MX_I2C2_Init/g' \
  Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_bus.{c,h}
sed -i 's/\bMX_DCMI_Init\b/BSP_MX_DCMI_Init/g' \
  Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_camera.{c,h}
```

Also remove the `static` from BSP's DMA handle so the IRQ handler in
`stm32u5xx_it.c` can reach it:

```bash
sed -i 's/^static DMA_HandleTypeDef hdma_handler;$/DMA_HandleTypeDef hdma_handler;/' \
  Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_camera.c
```

### 5. Swap in the four application files

- *Add* `Core/Inc/classifier_cam_app.h`
- *Add* `Core/Src/classifier_cam_app.c`
- *Replace* `Core/Inc/constants.h` with the version from this folder
- *Remove* `Core/Inc/classifier_app.h` and `Core/Src/classifier_app.c`

### 6. Patch CubeMX-generated files inside USER CODE blocks

These edits all sit inside `USER CODE` blocks, so they survive future
`.ioc` regenerations.

**`Core/Src/main.c`** (USER CODE BEGIN 2): add the USART1 init that
CubeMX's regen omits.

```c
/* USER CODE BEGIN 2 */
MX_USART1_UART_Init();
CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
DWT->CYCCNT = 0;
DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
/* USER CODE END 2 */
```

**`Core/Src/usart.c`** (line ~71): change USART1 baud.

```c
huart1.Init.BaudRate = 921600;   /* was 115200 */
```

**`Core/Src/usart.c`** (USER CODE BEGIN 0 area): wrap the `_write`
function in `#if 0 ... #endif` to avoid colliding with the same
symbol in `X-CUBE-AI/App/aiTestUtility.c`.

**`X-CUBE-AI/App/app_x-cube-ai.c`**: hook our app into the
X-CUBE-AI lifecycle.

```c
/* USER CODE BEGIN includes */
#include "classifier_cam_app.h"
/* USER CODE END includes */

void MX_X_CUBE_AI_Init(void) {
    /* USER CODE BEGIN 5 */
    classifier_cam_init();
    /* USER CODE END 5 */
}
void MX_X_CUBE_AI_Process(void) {
    /* USER CODE BEGIN 6 */
    classifier_cam_process();   /* never returns */
    /* USER CODE END 6 */
}
```

**`Core/Src/stm32u5xx_it.c`** — two patches:

1. Redirect `GPDMA1_Channel12_IRQHandler` (USER CODE BEGIN
   GPDMA1_Channel12_IRQn 0) to use BSP's DMA handle:
```c
extern DMA_HandleTypeDef hdma_handler;
HAL_DMA_IRQHandler(&hdma_handler);
return;
```
2. Add a `DCMI_PSSI_IRQHandler` (USER CODE BEGIN 1, end of file):
```c
extern DCMI_HandleTypeDef hcamera_dcmi;
void DCMI_PSSI_IRQHandler(void) {
    HAL_DCMI_IRQHandler(&hcamera_dcmi);
}
```

### 7. Build + flash

Same as `Classifier`. The MCU will print `BOOT\r\n` over USART1
(ST-LINK VCP) and wait for one sync byte, then start streaming
frames.

## Wire protocol (per frame, repeats forever)

```
MCU -> "FRAME\r\n"
MCU -> 19200 bytes  uint8 grayscale, row-major HW (H=120, W=160)
MCU -> 24 bytes     "<ifIIII":
                       i32 pred_class
                       f32 confidence
                       u32 t_pre_cycles
                       u32 t_infer_cycles
                       u32 t_post_cycles
                       u32 t_all_cycles
```

Bandwidth: ~19.2 KB/frame × ~10 fps target ≈ 200 KB/s. Comfortable
at 921600 baud (~92 KB/s of raw payload after overhead → ~4–5 fps
sustained on the link). If you want more fps, drop the image stream
and send only the 24-byte result.

## Running the host

```
cd firmware/Host
python host_cam.py                    # live preview + terminal
python host_cam.py --no-preview       # terminal only
python host_cam.py --save out_frames  # also save each frame as PNG
```

## Common gotchas

- **`CREATE_FAIL`/`INIT_FAIL` over UART**: the AI runtime didn't
  bind. Same root cause as in the host-fed Classifier — make sure
  the multi-network registry and `--allocate-inputs/-outputs` are
  consistent with what the generated code expects.
  See: `feedback_xcubeai_inputs_get_gotcha`.
- **`CAM_INIT_FAIL`**: I2C2 isn't reaching the OV5640. Verify B-CAMS-OMV
  is seated on CN2, the camera-power LDO is enabled (board-dependent
  GPIO), and SCL/SDA pins match what BSP expects.
- **Garbled image (chroma in the grayscale)**: byte order in YUV422
  is sensor-dependent. OV5640 default is `YUYV` (Y0 U Y1 V), so even
  bytes are Y. If your BSP build sets it to `UYVY`, swap to
  `gray[i] = cam[2*i + 1]` in `yuv422_extract_y`.
- **`CAMERA_R160x120` rejected with `BSP_ERROR_FEATURE_NOT_SUPPORTED`**:
  some BSP variants only enumerate QVGA+. Either fall back to
  QVGA YUV422 and decimate Y by 2, or patch the OV5640 driver's
  resolution table to expose the QQVGA timing block (it's a
  documented OV5640 mode, not a hardware limitation).
- **Predictions look terrible vs. host-fed Classifier**: framing.
  The training dataset is hand-centric close-ups; if the B-CAMS-OMV
  is mounted such that the hand fills only a small fraction of the
  frame, the model will struggle. Verify by saving a few frames with
  `host_cam.py --save out_frames` and running them through the
  existing `host.py` path — if accuracy is fine there, it's a
  framing problem, not a preprocessing problem.

## Why no DCMI driver in the source tree?

`STM32U585` has DCMI peripheral support, but the HAL driver for it
is only pulled in when you enable DCMI in the `.ioc`. We deliberately
don't ship copies of those HAL files here — keeping them in CubeMX's
control means you stay one regenerate away from a clean upgrade.
