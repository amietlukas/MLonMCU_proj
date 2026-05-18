# Dataset_Gatherer — VGA JPEG over WiFi (B-U585I-IOT02A + B-CAMS-OMV)

Captures 640×480 JPEG frames from the OV5640 on B-CAMS-OMV and pushes
them to a PC over WiFi (TCP). The PC runs `receive_images.py`. Designed
for ball-detection dataset gathering.

## What's in this folder

| File | Purpose |
|------|---------|
| `Core/Inc/constants.h`            | WiFi creds, PC IP/port, camera config. **Edit these.** |
| `Core/Inc/dataset_gather_app.h`   | App entry-point declarations. |
| `Core/Src/dataset_gather_app.c`   | Capture loop + WiFi/TCP transport. |
| `Core/Src/main.c`                 | Reference boot sequence. |

## Setup

Easiest path: copy the `Classifier_Cam` STM32CubeIDE project as a
starting point (its README documents all the camera-side CubeMX setup),
then add the WiFi pieces below.

### 1. Configure peripherals in CubeMX (`.ioc`)

In addition to everything `Classifier_Cam`'s README enables (DCMI, I2C2,
GPDMA1 channel for DCMI, USART1):

- **Connectivity → SPI2** (or whichever SPI is wired to the MXCHIP on your
  board revision — check the schematic; Rev C uses SPI2): enable as Full-
  Duplex Master, NSS hardware output, Motorola, 8-bit, MSB first, 4 MBit/s.
  Pinout is fixed by the board layout.
- **GPIO**: route the MXCHIP control pins (NSS, NOTIFY, FLOW, RESET) per
  the U585 BSP wiring (`b_u585i_iot02a_conf.h` defines them).
- **NVIC**: enable EXTI line for the MXCHIP NOTIFY pin.

### 2. Pull in the MXCHIP WiFi BSP + driver

From STM32Cube_FW_U5 (`~/STM32Cube/Repository/STM32Cube_FW_U5_V*/`):

- `Drivers/BSP/Components/mx_wifi/*` → copy into `Drivers/BSP/Components/`
- `Drivers/BSP/B-U585I-IOT02A/b_u585i_iot02a_conf_template.h` → already
  renamed for camera; ensure the `USE_BSP_COM_FEATURE` and
  `MX_WIFI_*` macros are set.
- Add include paths: `../Drivers/BSP/Components/mx_wifi`,
  `../Drivers/BSP/Components/mx_wifi/core`.
- Make sure the generated `mx_wifi_conf.h` is present in `Core/Inc/`
  (copy from `mx_wifi_conf_template.h` and pick the bare-OS variant for a
  RTOS-less project).

### 3. Drop the four application files

- *Add* `Core/Inc/constants.h` (or replace the Classifier_Cam version)
- *Add* `Core/Inc/dataset_gather_app.h`
- *Add* `Core/Src/dataset_gather_app.c`
- Keep CubeMX-generated `main.c`; just insert the two USER-CODE-2 lines
  (`dataset_gather_init();` then `dataset_gather_process();`) — the
  `main.c` in this folder is a reference for what those calls look like.

### 4. Edit credentials in `constants.h`

```c
#define WIFI_SSID      "your_wifi_ssid"
#define WIFI_PASSWORD  "your_wifi_password"
#define HOST_IP        "192.168.1.100"   /* PC IP on the WiFi network */
#define HOST_PORT      8888
```

Find your PC's IP with `ip addr` (Linux/macOS) or `ipconfig` (Windows).
The PC and the U585 must be on the same WiFi network.

### 5. Build and flash

The MCU prints status to the ST-LINK VCP (USART1):

```
BOOT
CAM: VGA JPEG OK
WIFI: joining AP...
WIFI: ip=192.168.1.42
TCP: connected to 192.168.1.100:8888
CAM: started
FRAME #0 len=42313 err=0
FRAME #30 len=39810 err=0
...
```

## Wire protocol (per frame)

```
MCU -> 4 bytes  big-endian uint32 length L
MCU -> L bytes  JPEG bytestream (ends with FFD9)
```

Repeats forever. Frame rate depends on JPEG size and link quality —
expect 5–15 fps on a clean 2.4 GHz network at VGA.

## Common gotchas

- **`CAM_INIT_FAIL` on `CAMERA_PF_JPEG`**: ST's stock OV5640 BSP exposes
  JPEG via `CAMERA_PF_JPEG` but not all builds program the sensor's JPEG
  register block. If init fails, patch `ov5640.c` to add the JPEG-mode
  init sequence (the OV5640 datasheet / OmniVision app-note documents
  the register sequence under "JPEG output mode").
- **`WIFI_BUS_FAIL`**: SPI to the MXCHIP isn't reaching the module.
  Verify the SPI peripheral CubeMX picked matches your board revision
  (Rev C ships SPI2; older revs differ) and that NSS/NOTIFY/FLOW pins
  match `b_u585i_iot02a_conf.h`.
- **`WIFI_CONNECT_FAIL`**: SSID/password wrong, AP on 5 GHz only (EMW3080B
  is 2.4 GHz only), or hidden SSID. Use a 2.4 GHz, broadcast SSID.
- **`SOCK_CONNECT_FAIL`**: PC firewall is blocking inbound TCP. On Linux:
  `sudo ufw allow 8888/tcp`. On Windows: allow Python through the firewall
  when prompted.
- **No `EOI marker` warnings**: the JPEG buffer overflowed (frame was
  bigger than `CAM_JPEG_BUFFER_BYTES = 128 KB`). Either bump the buffer
  (verify it still fits in SRAM3) or lower JPEG quality in the OV5640
  driver.
- **Garbled JPEGs received on PC**: byte-order mismatch between the U585
  and PC parser is unlikely (TCP is bytestream-correct), but check that
  `recv_exact` on the PC reads exactly L bytes per frame. A length-prefix
  desync usually shows as one bogus frame followed by a `bogus frame
  length` message in the Python receiver.
