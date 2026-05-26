#ifndef CONSTANTS_H
#define CONSTANTS_H

/* ==================================================================
 * Classifier_Cam constants
 *
 * Same int8 bignet_pruned model as the host-fed Classifier project,
 * but the input image now comes from the on-board OV5640 (B-CAMS-OMV)
 * instead of being streamed from the host.
 *
 * Model:
 *   Input  : int8(1,1,120,160) NCHW grayscale, QLinear(s=1.0, zp=-128)
 *   Output : int8(1,6),        QLinear(s=0.22004639, zp=-39)
 *
 * The exported ONNX has mean/std normalization fused into the first Conv,
 * so the model expects raw [0,255] FLOAT pixel values. With INPUT_SCALE=1.0
 * and INPUT_ZP=-128 the on-MCU quantization is just (uint8 u) -> int8(u-128).
 * ================================================================== */

#define NUM_CLASSES        6

#define MODEL_H            120
#define MODEL_W            160
#define MODEL_C            1
#define MODEL_INPUT_BYTES  (MODEL_H * MODEL_W * MODEL_C)    /* 19200 */
#define MODEL_OUTPUT_BYTES NUM_CLASSES                      /* 6 int8 logits */

#define INPUT_SCALE        1.0f
#define INPUT_ZP           (-128)

#define OUTPUT_SCALE       0.22004639f
#define OUTPUT_ZP          (-39)

/* ==================================================================
 * Camera capture configuration (B-CAMS-OMV / OV5640).
 *
 * Capture at QQVGA (160x120) YUV422 — exactly the resolution the model
 * was trained on (dataset: hagrid_full_qqvga_resize). The Y channel of
 * YUV422 is the luminance / grayscale value, so producing the model's
 * input is a flat byte-strided extract: Y0 U Y1 V Y2 U Y3 V ... -> the
 * even bytes ARE the grayscale image. No resize, no RGB->Y math, no
 * float weights.
 *
 * Frame buffer: 160 * 120 * 2 = 38400 bytes (well under the 786 KB SRAM
 * on STM32U585).
 * ================================================================== */

#define CAM_W              MODEL_W
#define CAM_H              MODEL_H
#define CAM_BPP            2                                /* YUV422: 2 bytes/pixel */
#define CAM_FRAME_BYTES    (CAM_W * CAM_H * CAM_BPP)        /* 38400 */

#define UART_TIMEOUT       HAL_MAX_DELAY

/* ==================================================================
 * Host-side debug stream
 *   STREAM_FRAMES_TO_HOST=1 -> emit the 19.2 KB grayscale frame +
 *     FRAME/result protocol on USART1 (consumed by host_cam.py).
 *   STREAM_FRAMES_TO_HOST=0 -> quiet mode: USART1 only emits the
 *     "PRED ... -> BT 'X' (...)" status line per inference.
 *     Best for picocom-style debugging.
 * ================================================================== */
#define STREAM_FRAMES_TO_HOST   0

/* When set, skip camera/AI entirely and run a transparent USART1 <->
 * USART3 byte bridge so the ST-LINK VCP (picocom on /dev/ttyACM0) talks
 * directly to the HC-05 wired on PD8/PD9. USART3 is re-initialised to
 * 38400 baud (HC-05 AT-mode baud) at startup. Use this to configure
 * HC-05 once (AT+ROLE/AT+BIND/...), then set back to 0 and reflash. */
#define BT_AT_BRIDGE            0

/* DWT cycles -> ms conversion. 160 MHz HCLK on B-U585I-IOT02A. */
#define CPU_FREQ_HZ_GUESS       160000000u

#endif
