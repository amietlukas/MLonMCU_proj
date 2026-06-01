#ifndef CONSTANTS_H
#define CONSTANTS_H

/* ==================================================================
 * BallDetector_N6 — compile-time configuration
 *
 * YOLO ball detector on STM32 Nucleo-N657X0-Q + B-CAMS-IMX (IMX335),
 * no LCD: frames and/or decoded boxes stream to the PC over UART.
 *
 * Model (firmware/BallDetector_N6/model/int8_ptq_qdq.onnx):
 *   Input  : uint8(1,3,288,384) NCHW RGB, QLinear(s=1/255, zp=0)
 *   Outputs: 3 int8 YOLO heads p8/p16/p32, strides 8/16/32, 1 class
 *
 * Model-shape constants are owned by yolo_postproc.h (YOLO_INPUT_W/H,
 * YOLO_NUM_HEADS, ...) and the X-CUBE-AI-generated network.h
 * (LL_ATON_DEFAULT_IN_/OUT_*_SIZE_BYTES). We reference those rather than
 * redefine them, so the deployed model stays the single source of truth.
 * ================================================================== */

#include "yolo_postproc.h"   /* YOLO_INPUT_W/H, YOLO_NUM_HEADS, YOLO_MAX_DET */

/* RGB888 from the DCMIPP NN pipe. */
#define MODEL_W            YOLO_INPUT_W                 /* 384 */
#define MODEL_H            YOLO_INPUT_H                 /* 288 */
#define MODEL_C            3
#define MODEL_INPUT_BYTES  (MODEL_W * MODEL_H * MODEL_C) /* 331776, == network.h IN_1 */
#define MODEL_NUM_CLASSES  1                            /* "Ball" */

/* ==================================================================
 * Detection / post-processing (see Core/Src/yolo_postproc.c).
 * ================================================================== */
#define DET_CONF_THRESH    0.45f   /* objectness threshold (post-sigmoid); real balls
                                    * score >=0.57, low-conf false positives <0.40 — tune as needed */
#define DET_NMS_IOU        0.25f   /* NMS IoU threshold                   */
#define DET_MAX_BOXES      YOLO_MAX_DET   /* cap reported boxes (<= 8)    */

/* ==================================================================
 * UART transport (ST-LINK VCP). Wire format:
 * firmware/Host/protocol_balldetector_n6.md.
 * ================================================================== */
#define UART_BAUD          921600
#define UART_TX_TIMEOUT_MS  1000
#define UART_RX_TIMEOUT_MS  1000

/* Firmware version reported in the INFO frame (BCD-ish: 0x0100 = v1.00). */
#define FW_VERSION         0x0100

/* Largest IMG_CHUNK payload the host sends (must match host.py CHUNK). */
#define UART_MAX_CHUNK     4096

#endif /* CONSTANTS_H */
