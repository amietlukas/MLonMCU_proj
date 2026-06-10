# Classifier_U5
On-device hand-gesture classification on the B-U585I-IOT02A. The board captures
QQVGA (160×120) grayscale frames from the OV5640 camera, runs a 6-class gesture
classifier on-device, and sends one drive command per inference to an RC car over
Bluetooth. It also streams the image + result over UART for optional monitoring.

## Structure

```text
Classifier_U5/
├── Core/           # application (classifier_cam_app.c) + CubeMX HAL
├── X-CUBE-AI/      # generated network (X-CUBE-AI 10.2.0)
├── Drivers/        # ST HAL + BSP
├── Debug/          # make-based build output
└── stm32.ioc       # CubeMX project
```

## Hardware

    Board:     B-U585I-IOT02A (STM32U585AI)
    Camera:    B-CAMS-OMV (OV5640)
    Bluetooth: HC-05 on USART3/PA7 (master) -> pairs with the HC-06 on the RC car
    Gestures:  palm=STOP rock=FWD pinkie=FWD-RIGHT one=FWD-LEFT fist=BACK other=STOP

## Build and Flash

Open the project in STM32CubeIDE and Build + Run, or from the command line:

    cd Debug
    make -j8
    STM32_Programmer_CLI -c port=SWD mode=UR -d Classifier_Cam.elf -hardRst

## Monitor (optional)

The board drives the car on its own; the UART stream is just for watching it.
`host_u5.py` (in `firmware/Host/`) prints each prediction and shows the frame:

    cd ../Host
    python3 host_u5.py --port /dev/ttyACM0

## Tools Version
Use these versions or the build won't work

    STM32CubeIDE (v1.17.0)
    STM32CubeProgrammer (v2.18.0)
    X-CUBE-AI (v10.2.0)
