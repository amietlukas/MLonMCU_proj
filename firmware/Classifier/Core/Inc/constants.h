#ifndef CONSTANTS_H
#define CONSTANTS_H

/* =============================================================
 * Active model selection
 * -------------------------------------------------------------
 * Set USE_INT8_MODEL to:
 *   1  -> int8 quantized variant (small_net_int8 entry)
 *   0  -> fp32 variant            (small_net_fp32 entry)
 * Both networks are compiled in via the X-CUBE-AI multi-network
 * registry; this flag just picks which one to instantiate at boot.
 * =============================================================
 */
#define USE_INT8_MODEL     1

/* =============================================================
 * Current model: smallnet_greyscale baseline (HAGRID 6-class)
 *
 * The ONNX export bakes the full inference preprocessing into the
 * first Conv (both /255 and (x-mu)/sigma), so the graph expects
 * raw [0, 255] FLOAT pixel values:
 *   - fp32 model: cast each uint8 pixel to float, feed directly.
 *   - int8 model: INPUT_SCALE=1.0, INPUT_ZP=-128 so the on-MCU
 *                 quantization reduces to int8 q = (int8)(u - 128).
 *
 * Output dequant params come from the DequantizeLinear at the
 * int8 model's output (see small_net_int8_generate_report.txt).
 * =============================================================
 */
#define NUM_CLASSES        6

#define MODEL_H            120
#define MODEL_W            160
#define MODEL_C            1
#define MODEL_INPUT_BYTES  (MODEL_H * MODEL_W * MODEL_C)    /* 19200 host bytes */
#define MODEL_OUTPUT_BYTES NUM_CLASSES

/* int8 input/output quant (smallnet_greyscale int8, fused export). */
#define INPUT_SCALE        1.0f
#define INPUT_ZP           (-128)
#define OUTPUT_SCALE       0.231398404f   /* bignet int8 */
#define OUTPUT_ZP          (-46)

#define UART_TIMEOUT       HAL_MAX_DELAY

#endif
