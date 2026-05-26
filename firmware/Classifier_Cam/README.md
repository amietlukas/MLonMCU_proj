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

**USART3 pin reassignment + baud (Arduino-header pins for HC-05)**:
the inherited Classifier `.ioc` routes USART3 to PA7/PA5, which sit on
the on-board STMod+ area, not the Arduino headers. Reassign USART3 to
the Arduino D0/D1 pins **PD9 (RX) / PD8 (TX)** so an HC-05 plugged into
the Arduino socket "just works", and drop the baud to 9600.

In the `.ioc`: clear PA7/PA5, set **PD8 = USART3_TX**, **PD9 =
USART3_RX**, regenerate. Then in `Core/Src/usart.c`:

```c
/* MX_USART3_UART_Init, line ~115 */
huart3.Init.BaudRate = 9600;     /* was 115200 — HC-05/06 default is 9600 */

/* HAL_UART_MspInit USART3 branch, line ~210 — CubeMX will write this
 * for you after the pin reassignment + regen; shown here for review:  */
__HAL_RCC_GPIOD_CLK_ENABLE();
GPIO_InitStruct.Pin       = GPIO_PIN_8 | GPIO_PIN_9;   /* PD8=TX, PD9=RX */
GPIO_InitStruct.Mode      = GPIO_MODE_AF_PP;
GPIO_InitStruct.Pull      = GPIO_NOPULL;
GPIO_InitStruct.Speed     = GPIO_SPEED_FREQ_LOW;
GPIO_InitStruct.Alternate = GPIO_AF7_USART3;
HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);
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

## Bluetooth (HC-05 on U585 → HC-06 on Arduino)

The capture loop sends one ASCII byte per inference over **USART3** to
drive the `paul_car.ino` Arduino sketch. The U585 carries the HC-05
(master) and pairs with the HC-06 (slave) on the Arduino side.

### Board wiring (B-U585I-IOT02A Arduino header)

USART3 is routed to the Arduino UNO R3 UART pins **D0 (PD9) = RX** and
**D1 (PD8) = TX** via `GPIO_AF7_USART3`. On the B-U585I-IOT02A the
Arduino digital header is the row labelled CN13 — D0/D1 are the two
pins closest to the corner of the board, silk-screened "D0" and "D1".

| HC-05 pin | B-U585I-IOT02A pin           | Notes |
|-----------|------------------------------|-------|
| VCC       | `5V` on the Arduino power header (CN8) | HC-05 breakouts need 5V; on-board regulator drops it to 3.3V for the radio. |
| GND       | any `GND` on CN8             | Common ground with the U585. |
| **RXD**   | **D1** = PD8 (USART3_TX)     | 3.3V TX from PD8 is fine for HC-05 RXD (3.3V-tolerant on most breakouts). |
| TXD       | **D0** = PD9 (USART3_RX)     | Optional — only needed if you want to read AT responses or replies from the Arduino. |
| EN / KEY  | (leave floating)             | Only pulled high to enter AT-command mode for pairing/baud changes. |
| STATE     | (leave floating)             | Optional connection-state indicator. |

Do **not** swap RXD/TXD — HC-05 RXD is an input, so it goes to **D1
(PD8, U585 TX)**. HC-05 TXD goes to **D0 (PD9, U585 RX)**.

### Arduino side

Standard HC-06 wiring: VCC→5V, GND→GND, HC-06 RXD→Arduino TX, HC-06
TXD→Arduino RX. The `paul_car.ino` sketch reads single chars and
already maps `'0'..'5'` to STOP/FWD/FWD-RIGHT/FWD-LEFT/BACKWARD/OTHER.

### Pairing HC-05 (master) to HC-06 (slave) — one time

The HC-05 needs to be told which HC-06 to dial. With the HC-05's `EN`
pin held high at power-up it enters AT mode at **38400 baud**. From a
USB-UART:

```
AT+ROLE=1            // master
AT+CMODE=0           // connect to a specific address only
AT+BIND=xxxx,xx,xxxxxx   // HC-06's MAC, comma-separated
AT+INIT              // initialize SPP profile
AT+LINK=xxxx,xx,xxxxxx   // optional, force-connect now
```

Find the HC-06 MAC by powering it up and running `AT+ADDR?` on it
(HC-06 AT mode is always-on at 9600 baud as long as it's not paired).

After pairing, both modules sit at 9600 8N1 and the link is transparent
— the U585's `HAL_UART_Transmit(&huart3, ...)` byte pops out of the
Arduino's hardware UART RX.

### Gesture → command map

Defined in `prediction_to_cmd()` in `classifier_cam_app.c`. Edit there
to taste:

| Class | Gesture | Byte | Meaning      |
|-------|---------|------|--------------|
| 0     | palm    | `'0'`| STOP         |
| 1     | rock    | `'1'`| FORWARD      |
| 2     | pinkie  | `'3'`| FWD-LEFT     |
| 3     | one     | `'2'`| FWD-RIGHT    |
| 4     | fist    | `'4'`| BACKWARD     |
| 5     | others  | `'0'`| STOP         |

Predictions below `CMD_MIN_CONF` (default 0.5) are forced to `'0'`
(STOP) to keep a flickery low-confidence frame from slamming the car
between directions.

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
