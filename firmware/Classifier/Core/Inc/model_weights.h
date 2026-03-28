#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <stdint.h>

/* ======================================================================
 * Auto-generated from ONNX model by extract_weights.py
 * Model: letsgooo_v2-20260310-010152_best-epoch113_int8_qdq_op13.onnx
 * ====================================================================== */

/* Input quantization */
#define MODEL_INPUT_SCALE    0.0131675974f
#define MODEL_INPUT_ZP       (-37)

/* Output quantization */
#define MODEL_OUTPUT_SCALE   0.1351625323f
#define MODEL_OUTPUT_ZP      (-43)

/* Per-layer activation quantization (scale, zero_point) */
#define CONV0_OUT_SCALE      0.0352561511f
#define CONV0_OUT_ZP         (-128)
#define CONV1_OUT_SCALE      0.0252937451f
#define CONV1_OUT_ZP         (-128)
#define CONV2_OUT_SCALE      0.0106701776f
#define CONV2_OUT_ZP         (-128)
#define CONV3_OUT_SCALE      0.1946646571f
#define CONV3_OUT_ZP         (-128)
#define AVGPOOL_OUT_SCALE    0.0066707809f
#define AVGPOOL_OUT_ZP       (-128)

/* Weight and quantization parameter arrays */
extern const int8_t conv0_weights[432];
extern const int32_t conv0_bias[16];
extern const int32_t conv0_output_mult[16];
extern const int32_t conv0_output_shift[16];

extern const int8_t conv1_weights[4608];
extern const int32_t conv1_bias[32];
extern const int32_t conv1_output_mult[32];
extern const int32_t conv1_output_shift[32];

extern const int8_t conv2_weights[18432];
extern const int32_t conv2_bias[64];
extern const int32_t conv2_output_mult[64];
extern const int32_t conv2_output_shift[64];

extern const int8_t conv3_weights[73728];
extern const int32_t conv3_bias[128];
extern const int32_t conv3_output_mult[128];
extern const int32_t conv3_output_shift[128];

extern const int8_t fc_weights[768];
extern const int32_t fc_bias[6];
extern const int32_t fc_output_mult[6];
extern const int32_t fc_output_shift[6];

#endif /* MODEL_WEIGHTS_H */
