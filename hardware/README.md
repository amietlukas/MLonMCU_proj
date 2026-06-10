# Hardware
The physical components used across the project: two STM32 boards (one per
firmware pipeline) plus the RC car, cameras, wireless link and actuators.

## Boards

| Board | Used by | Role |
| :---- | :------ | :--- |
| [NUCLEO-N657X0-Q](https://www.st.com/en/evaluation-tools/nucleo-n657x0-q.html) (STM32N6) | `firmware/BallDetector_N6` | On-device ball detection on the NPU; drives the car + camera servo |
| [B-U585I-IOT02A](https://www.st.com/en/evaluation-tools/b-u585i-iot02a.html) (STM32U5) | `firmware/Classifier_U5` | On-device hand-gesture classification; sends drive commands |

## Cameras

| Camera | On | Notes |
| :----- | :- | :---- |
| IMX335 module | NUCLEO-N657X0-Q | ships with the board; used for ball detection |
| [B-CAMS-OMV](https://www.st.com/en/evaluation-tools/b-cams-omv.html) (OV5640) | B-U585I-IOT02A | captures QQVGA grayscale for gesture input |

## Wireless link (Bluetooth)

The gesture board and the car talk over a Bluetooth serial pair (9600 baud):

    HC-05 (master)  on the B-U585I-IOT02A (USART3)  --(Bluetooth)-->  HC-06 (slave) on the car

## RC car

The base RC car (chassis, motors, wheels) we bought — parts list, photos and
reference material:

https://www.dropbox.com/scl/fo/mgbtbakgduv7u3dstyja3/AHI4W15Y3G3kvc49oE3PCbg?rlkey=4e7v5k8yfv53zf3w6pt4gedie&e=1&dl=0

## Additional Parts

- **7.4V 2S LiPo** 
- **HC-06**