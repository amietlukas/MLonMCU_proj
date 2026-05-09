#ifndef CONSTANTS_H
#define CONSTANTS_H

/* Model: bignet_pruned int8 (HAGRID 6-class gesture classifier, grayscale)
 * Input  : int8(1,1,120,160) NCHW (H=120, W=160 — landscape),
 *          QLinear(s=1.0, zp=-128)
 * Output : int8(1,6),         QLinear(s=0.22004639, zp=-39)
 *
 * The exported ONNX has the per-channel mean/std normalization fused into the
 * first Conv, so the model expects raw [0,255] FLOAT pixel values. With
 * INPUT_SCALE=1.0 and INPUT_ZP=-128 the on-MCU quantization reduces to:
 *     int8 q = (int8_t)(u - 128)         for each uint8 pixel u
 * i.e. a flat uint8 -> int8 reinterpretation. No mean/std math on the MCU.
 *
 * Host sends post-letterbox uint8 grayscale HW bytes (H=120, W=160 = 19200 B).
 * Layout is row-major HW, which matches NCHW for C=1.
 */

#define NUM_CLASSES        6

#define MODEL_H            120
#define MODEL_W            160
#define MODEL_C            1
#define MODEL_INPUT_BYTES  (MODEL_H * MODEL_W * MODEL_C)    /* 19200 */
#define MODEL_OUTPUT_BYTES NUM_CLASSES                       /* 6 int8 logits */

/* Input quantization (from the ONNX QuantizeLinear at the model input) */
#define INPUT_SCALE        1.0f
#define INPUT_ZP           (-128)

/* Output quantization (from the ONNX DequantizeLinear at the model output) */
#define OUTPUT_SCALE       0.22004639f
#define OUTPUT_ZP          (-39)

#define UART_TIMEOUT       HAL_MAX_DELAY

#endif
