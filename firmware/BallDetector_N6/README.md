# BallDetector_N6
On-device ball detection on the NUCLEO-N657X0-Q. The board runs a STEdgeAI-generated
YOLO-style detector on the STM32N6 NPU, draws bounding boxes on the camera frame, and
streams the annotated video to the PC as a USB/UVC webcam. The detected ball position
pans a camera servo and drives an RC car (motor commands arrive over Bluetooth).

Derived from ST's [STM32N6 Object Detection Getting Started](https://github.com/STMicroelectronics/STM32N6-GettingStarted-ObjectDetection)
example: camera/ISP/boot/flash infrastructure is reused, the model, post-processing,
tracking/actuation and deploy tooling are our own. ST licensing is kept intact.

## Structure

```text
BallDetector_N6/
├── Application/    # app sources (main.c, yolo_postproc, ball_tracker, motor/servo)
├── Model/          # balldet_int8.onnx + generate/deploy scripts + neural-art config
├── Drivers/ Middlewares/ FSBL/   # ST HAL, LL_ATON runtime, first-stage bootloader
├── deploy.sh       # generate -> build -> sign -> (optional) flash
└── STM32Cube_FW_N6/
```

## Hardware

    Board:     NUCLEO-N657X0-Q (power + ST-LINK over USB-C, CN9)
    Camera:    IMX335 module
    Video out: USB/UVC on the OTG port (CN8) -> appears as a webcam on the PC
    Bluetooth: HC-06 on USART3 (PD8/PD9, 9600) for RC drive/steer commands
    Actuators: L298N H-bridge (motors) + SG90 servo (PA3/TIM16, camera pan)

Power: one 7.4V 2S LiPo feeds the L298N and (optionally) STM32 VIN. The servo runs
off the L298N's 5V regulator, NOT the board 5V rail (which browns out under the
servo's current). One common ground.

## Build and Flash

`deploy.sh` runs the full path from a quantized ONNX to a signed, board-ready image.
Toolchain paths are auto-detected from PATH or set via env vars (GCC, STEDGEAI, CUBEPROG).

    cd firmware/BallDetector_N6
    ./deploy.sh Model/balldet_int8.onnx           # generate + build + sign
    ./deploy.sh Model/balldet_int8.onnx --flash    # ...and flash (board in DEV mode)

The STM32N6 has no internal flash, so the app + weights are written to external flash
and signed (`STM32_SigningTool_CLI`, SSBL header — `-nk`/no key, so this is the bootable
format + integrity hash, not full authenticated secure boot). Set the board to
boot-from-flash (jumpers JP1/JP2, see the N6 manual) and power-cycle to run.

## View the output

The board is standalone — it streams finished, annotated video over UVC. `host_n6.py`
(in `firmware/Host/`) opens the UVC device and shows it; any webcam app works too.

    cd ../Host
    python3 host_n6.py            # auto-detect the board's UVC device ('q' to quit)
    python3 host_n6.py --list     # list video devices if auto-detect picks wrong

## Notes

- Use the STM32CubeIDE GCC 12.3 toolchain (ST's build needs `-fcyclomatic-complexity`).
- Input is fixed to 384×288; a different size builds/flashes but decodes wrong.
- Model files were generated with STEdgeAI 4.0.0; another version may error with
  `Possible mismatch in ll_aton library used`.

## Tools Version
Use these versions or the build won't work

    STM32CubeIDE (v1.17.0)
    STM32CubeProgrammer (v2.18.0)
    STEdgeAI (v4.0.0)
