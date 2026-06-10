# Firmware
This folder holds the firmware that can build and later flashed onto the MCUs.

## Structure

```text
firmware/
├── BallDetector_N6/     # Ball detection pipeline
├── Classifier_U5/       # Hand Gesture classification pipeline
└── README.md
```

## Getting Started

Install the tools below, then pick a project and follow its README:

- **[BallDetector_N6](BallDetector_N6/)** — ball detection on the NUCLEO-N657X0-Q.
  Run `./deploy.sh Model/balldet_int8.onnx --flash`, then view the live feed with
  `python3 ../Host/host_n6.py`.
- **[Classifier_U5](Classifier_U5/)** — hand-gesture classification on the
  B-U585I-IOT02A. Build + flash from STM32CubeIDE, then monitor with
  `python3 ../Host/host_u5.py --port /dev/ttyACM0`.

The host-side viewer scripts live in [Host/](Host/) and need Python with
`pyserial`, `numpy`, and `opencv-python`.

## Tools Version
Use these versions or the build won't work

    STM32CubeIDE (v1.17.0)
    STM32CubeProgrammer (v2.18.0)
    STEdgeAI (v4.0.0)
