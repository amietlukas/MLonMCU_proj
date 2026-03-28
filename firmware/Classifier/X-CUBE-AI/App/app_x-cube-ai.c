/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @brief   AI inference application — multi-backend
  *
  *          Supports three backends selected at compile time:
  *            USE_BACKEND_FP32        — X-CUBE-AI FP32  (default)
  *            USE_BACKEND_INT8_XCUBE  — X-CUBE-AI INT8
  *            USE_BACKEND_INT8_CMSIS  — CMSIS-NN  INT8  (fastest)
  *
  *          Set exactly ONE of the above as a preprocessor define.
  *          If none is set, FP32 is used by default.
  ******************************************************************************
  */

#ifdef __cplusplus
extern "C" {
#endif

/* ── Includes ──────────────────────────────────────────────────────────── */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "constants.h"

/* ── Default backend ───────────────────────────────────────────────────── */
#if !defined(USE_BACKEND_FP32) && !defined(USE_BACKEND_INT8_XCUBE) && !defined(USE_BACKEND_INT8_CMSIS)
#define USE_BACKEND_FP32
#endif

/* ── Backend-specific includes ─────────────────────────────────────────── */
#if defined(USE_BACKEND_FP32)
  #include "small_net_fp32.h"
  #include "small_net_fp32_data.h"
  #define AI_NET_CREATE       ai_small_net_fp32_create_and_init
  #define AI_NET_RUN          ai_small_net_fp32_run
  #define AI_NET_INPUTS_GET   ai_small_net_fp32_inputs_get
  #define AI_NET_OUTPUTS_GET  ai_small_net_fp32_outputs_get
  #define AI_NET_GET_REPORT   ai_small_net_fp32_get_report
  #define AI_NET_DESTROY      ai_small_net_fp32_destroy
  #define AI_ACTIVATIONS_SIZE AI_SMALL_NET_FP32_DATA_ACTIVATIONS_SIZE
  #define AI_IN_SIZE_BYTES    AI_SMALL_NET_FP32_IN_1_SIZE_BYTES
  #define AI_OUT_SIZE_BYTES   AI_SMALL_NET_FP32_OUT_1_SIZE_BYTES
  #define BACKEND_NAME        "X-CUBE-AI FP32"

#elif defined(USE_BACKEND_INT8_XCUBE)
  #include "small_net_int8.h"
  #include "small_net_int8_data.h"
  #define AI_NET_CREATE       ai_small_net_int8_create_and_init
  #define AI_NET_RUN          ai_small_net_int8_run
  #define AI_NET_INPUTS_GET   ai_small_net_int8_inputs_get
  #define AI_NET_OUTPUTS_GET  ai_small_net_int8_outputs_get
  #define AI_NET_GET_REPORT   ai_small_net_int8_get_report
  #define AI_NET_DESTROY      ai_small_net_int8_destroy
  #define AI_ACTIVATIONS_SIZE AI_SMALL_NET_INT8_DATA_ACTIVATIONS_SIZE
  #define AI_IN_SIZE_BYTES    AI_SMALL_NET_INT8_IN_1_SIZE_BYTES
  #define AI_OUT_SIZE_BYTES   AI_SMALL_NET_INT8_OUT_1_SIZE_BYTES
  #define BACKEND_NAME        "X-CUBE-AI INT8"

#elif defined(USE_BACKEND_INT8_CMSIS)
  #include "arm_nnfunctions.h"
  #include "model_weights.h"
  #define BACKEND_NAME        "CMSIS-NN INT8"
#endif

/* ── UART handle (defined in main.c) ──────────────────────────────────── */
extern UART_HandleTypeDef huart1;

/* ── Timer variables ──────────────────────────────────────────────────── */
volatile uint32_t timer_pre_start = 0, timer_pre_end = 0;
volatile uint32_t timer_infer_start = 0, timer_infer_end = 0;
volatile uint32_t timer_post_start = 0, timer_post_end = 0;

/* ════════════════════════════════════════════════════════════════════════
 *  X-CUBE-AI backends (FP32 or INT8)
 * ════════════════════════════════════════════════════════════════════════ */
#if defined(USE_BACKEND_FP32) || defined(USE_BACKEND_INT8_XCUBE)

/* X-CUBE-AI network handle & buffers */
static ai_handle network = AI_HANDLE_NULL;

AI_ALIGNED(4)
static ai_u8 activations[AI_ACTIVATIONS_SIZE];

/* Raw UART receive buffer (always uint8 RGB HWC) */
static uint8_t raw_input[MODEL_INPUT_SIZE];

/* --------------------------------------------------------------------------
 * Preprocessing: receive uint8 RGB HWC image over UART
 * -------------------------------------------------------------------------- */
static int acquire_and_process_data(ai_buffer *ai_input)
{
    /* Clear pending UART errors */
    __HAL_UART_CLEAR_OREFLAG(&huart1);
    __HAL_UART_CLEAR_FEFLAG(&huart1);
    __HAL_UART_CLEAR_NEFLAG(&huart1);

    /* Signal host */
    uint8_t ready_in_msg[] = "READY_IN\r\n";
    HAL_UART_Transmit(&huart1, ready_in_msg, sizeof(ready_in_msg) - 1, HAL_MAX_DELAY);

    /* Receive 128x128x3 uint8 RGB image */
    HAL_StatusTypeDef status = HAL_UART_Receive(&huart1, raw_input,
                                                (uint16_t)MODEL_INPUT_SIZE, HAL_MAX_DELAY);
    if (status != HAL_OK) return -1;

    timer_pre_start = DWT->CYCCNT;

#if defined(USE_BACKEND_FP32)
    /* FP32: normalise uint8 HWC -> float CHW (X-CUBE-AI expects CHW) */
    float *in_data = (float *)ai_input->data;
    const int total_pixels = OUT_SIZE * OUT_SIZE;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < total_pixels; i++) {
            float val = raw_input[i * 3 + c] / 255.0f;
            in_data[c * total_pixels + i] = (val - norm_mean[c]) / norm_std[c];
        }
    }
#else /* USE_BACKEND_INT8_XCUBE */
    /* INT8 X-CUBE-AI: quantize uint8 HWC -> int8 CHW (X-CUBE-AI expects CHW) */
    int8_t *in_data = (int8_t *)ai_input->data;
    static const float combined_scale[3] = {
        1.0f / (255.0f * 0.3270f * 0.0131675974f),
        1.0f / (255.0f * 0.3131f * 0.0131675974f),
        1.0f / (255.0f * 0.3042f * 0.0131675974f),
    };
    static const float combined_offset[3] = {
        (-0.3912f / (0.3270f * 0.0131675974f)) + (-37.0f),
        (-0.3612f / (0.3131f * 0.0131675974f)) + (-37.0f),
        (-0.3425f / (0.3042f * 0.0131675974f)) + (-37.0f),
    };
    const int total_pixels = OUT_SIZE * OUT_SIZE;
    for (int c = 0; c < 3; c++) {
        for (int i = 0; i < total_pixels; i++) {
            float val = raw_input[i * 3 + c] * combined_scale[c] + combined_offset[c];
            int32_t q = (int32_t)roundf(val);
            if (q < -128) q = -128;
            if (q > 127)  q = 127;
            in_data[c * total_pixels + i] = (int8_t)q;
        }
    }
#endif

    timer_pre_end = DWT->CYCCNT;
    return 0;
}

/* --------------------------------------------------------------------------
 * Postprocessing: argmax + softmax confidence
 * -------------------------------------------------------------------------- */
static int post_process(ai_buffer *ai_output)
{
    timer_post_start = DWT->CYCCNT;

    int pred_class = 0;

#if defined(USE_BACKEND_FP32)
    /* FP32 output: find argmax on floats */
    float *out_data = (float *)ai_output->data;
    float max_val = out_data[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (out_data[i] > max_val) {
            max_val = out_data[i];
            pred_class = i;
        }
    }
    /* Softmax confidence */
    float sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        sum += expf(out_data[i] - max_val);
    }
    float confidence = 1.0f / sum;

#else /* USE_BACKEND_INT8_XCUBE */
    /* INT8 output: argmax on int8, softmax on dequantized */
    int8_t *out_data = (int8_t *)ai_output->data;
    int8_t max_q = out_data[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (out_data[i] > max_q) {
            max_q = out_data[i];
            pred_class = i;
        }
    }
    /* Dequantize and softmax */
    float dequant[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        dequant[i] = ((float)out_data[i] - (-43.0f)) * 0.1351625323f;
    }
    float max_f = dequant[pred_class];
    float sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        sum += expf(dequant[i] - max_f);
    }
    float confidence = 1.0f / sum;
#endif

    timer_post_end = DWT->CYCCNT;

    uint32_t timer_pre   = timer_pre_end   - timer_pre_start;
    uint32_t timer_infer = timer_infer_end - timer_infer_start;
    uint32_t timer_post  = timer_post_end  - timer_post_start;
    uint32_t timer_all   = timer_post_end  - timer_pre_start;

    printf("DEBUG: Post-Process done. Sending output to HOST.\r\n");

    uint8_t ready_out_msg[] = "READY_OUT\r\n";
    HAL_UART_Transmit(&huart1, ready_out_msg, 11, HAL_MAX_DELAY);

    HAL_UART_Transmit(&huart1, (uint8_t *)&pred_class,  sizeof(int),      HAL_MAX_DELAY);
    HAL_UART_Transmit(&huart1, (uint8_t *)&confidence,  sizeof(float),    HAL_MAX_DELAY);
    HAL_UART_Transmit(&huart1, (uint8_t *)&timer_pre,   sizeof(uint32_t), HAL_MAX_DELAY);
    HAL_UART_Transmit(&huart1, (uint8_t *)&timer_infer, sizeof(uint32_t), HAL_MAX_DELAY);
    HAL_UART_Transmit(&huart1, (uint8_t *)&timer_post,  sizeof(uint32_t), HAL_MAX_DELAY);
    HAL_UART_Transmit(&huart1, (uint8_t *)&timer_all,   sizeof(uint32_t), HAL_MAX_DELAY);

    return 0;
}

/* --------------------------------------------------------------------------
 * Entry points — X-CUBE-AI backends
 * -------------------------------------------------------------------------- */
void MX_X_CUBE_AI_Init(void)
{
    printf("\r\n%s Model - initialization\r\n", BACKEND_NAME);

    const ai_handle act_addr[] = { activations };
    ai_error err = AI_NET_CREATE(&network, act_addr, NULL);
    if (err.type != AI_ERROR_NONE) {
        printf("ERROR: ai create failed (type=%lu, code=%lu)\r\n",
               (unsigned long)err.type, (unsigned long)err.code);
    }
}

void MX_X_CUBE_AI_Process(void)
{
    int res = -1;

    printf("%s - run - main loop\r\n", BACKEND_NAME);

    ai_buffer *ai_input  = AI_NET_INPUTS_GET(network, NULL);
    ai_buffer *ai_output = AI_NET_OUTPUTS_GET(network, NULL);

    do {
        /* 1 — acquire and pre-process input data */
        res = acquire_and_process_data(ai_input);

        /* 2 — run inference */
        if (res == 0) {
            printf("DEBUG: Starting Inference...\r\n");
            timer_infer_start = DWT->CYCCNT;

            ai_i32 n_batch = AI_NET_RUN(network, ai_input, ai_output);

            timer_infer_end = DWT->CYCCNT;
            printf("DEBUG: Inference Done\r\n");

            if (n_batch != 1) {
                printf("ERROR: inference failed (n_batch=%ld)\r\n", (long)n_batch);
                res = -1;
            }
        }

        /* 3 — post-process */
        if (res == 0)
            res = post_process(ai_output);
    } while (res == 0);

    if (res) {
        printf("Process has FAILED\r\n");
    }
}

#endif /* USE_BACKEND_FP32 || USE_BACKEND_INT8_XCUBE */

/* ════════════════════════════════════════════════════════════════════════
 *  CMSIS-NN INT8 backend
 * ════════════════════════════════════════════════════════════════════════ */
#if defined(USE_BACKEND_INT8_CMSIS)

/* Activation buffers (ping-pong) */
static int8_t buf_a[262144] __attribute__((aligned(4)));
static int8_t buf_b[131072] __attribute__((aligned(4)));
static int8_t scratch_buf[12288] __attribute__((aligned(4)));

/* Model I/O buffers */
static int8_t model_input_buf[MODEL_IMG_H * MODEL_IMG_W * MODEL_IMG_CH] __attribute__((aligned(4)));
static int8_t model_output_buf[MODEL_NUM_CLASSES];

/* --------------------------------------------------------------------------
 * CMSIS-NN inference (layer-by-layer)
 * -------------------------------------------------------------------------- */
static arm_cmsis_nn_status cmsis_nn_run(const int8_t *input, int8_t *output)
{
    arm_cmsis_nn_status status;
    cmsis_nn_context ctx;
    ctx.buf = scratch_buf;
    ctx.size = sizeof(scratch_buf);

    /* Layer 0: Conv2D 3x3, 3->16, pad=1, fused ReLU */
    {
        cmsis_nn_conv_params params = {
            .input_offset  = -MODEL_INPUT_ZP,
            .output_offset = CONV0_OUT_ZP,
            .stride  = {.w = 1, .h = 1},
            .padding = {.w = 1, .h = 1},
            .dilation = {.w = 1, .h = 1},
            .activation = {.min = -128, .max = 127}
        };
        cmsis_nn_dims input_dims  = {.n = 1, .h = 128, .w = 128, .c = 3};
        cmsis_nn_dims filter_dims = {.n = 16, .h = 3, .w = 3, .c = 3};
        cmsis_nn_dims bias_dims   = {.n = 1, .h = 1, .w = 1, .c = 16};
        cmsis_nn_dims output_dims = {.n = 1, .h = 128, .w = 128, .c = 16};
        cmsis_nn_per_channel_quant_params quant = {
            .multiplier = (int32_t *)conv0_output_mult,
            .shift      = (int32_t *)conv0_output_shift
        };
        ctx.size = arm_convolve_wrapper_s8_get_buffer_size(
            &params, &input_dims, &filter_dims, &output_dims);
        status = arm_convolve_wrapper_s8(
            &ctx, &params, &quant,
            &input_dims, input,
            &filter_dims, conv0_weights,
            &bias_dims, conv0_bias,
            &output_dims, buf_a);
        if (status != ARM_CMSIS_NN_SUCCESS) return status;
    }

    /* Layer 1: MaxPool 2x2 — 128x128x16 -> 64x64x16 */
    {
        cmsis_nn_pool_params params = {
            .stride    = {.w = 2, .h = 2},
            .padding   = {.w = 0, .h = 0},
            .activation = {.min = -128, .max = 127}
        };
        cmsis_nn_dims input_dims  = {.n = 1, .h = 128, .w = 128, .c = 16};
        cmsis_nn_dims filter_dims = {.n = 1, .h = 2, .w = 2, .c = 1};
        cmsis_nn_dims output_dims = {.n = 1, .h = 64, .w = 64, .c = 16};
        ctx.size = 0;
        status = arm_max_pool_s8(&ctx, &params,
            &input_dims, buf_a, &filter_dims, &output_dims, buf_b);
        if (status != ARM_CMSIS_NN_SUCCESS) return status;
    }

    /* Layer 2: Conv2D 3x3, 16->32, pad=1, fused ReLU */
    {
        cmsis_nn_conv_params params = {
            .input_offset  = -CONV0_OUT_ZP,
            .output_offset = CONV1_OUT_ZP,
            .stride  = {.w = 1, .h = 1},
            .padding = {.w = 1, .h = 1},
            .dilation = {.w = 1, .h = 1},
            .activation = {.min = -128, .max = 127}
        };
        cmsis_nn_dims input_dims  = {.n = 1, .h = 64, .w = 64, .c = 16};
        cmsis_nn_dims filter_dims = {.n = 32, .h = 3, .w = 3, .c = 16};
        cmsis_nn_dims bias_dims   = {.n = 1, .h = 1, .w = 1, .c = 32};
        cmsis_nn_dims output_dims = {.n = 1, .h = 64, .w = 64, .c = 32};
        cmsis_nn_per_channel_quant_params quant = {
            .multiplier = (int32_t *)conv1_output_mult,
            .shift      = (int32_t *)conv1_output_shift
        };
        ctx.size = arm_convolve_wrapper_s8_get_buffer_size(
            &params, &input_dims, &filter_dims, &output_dims);
        status = arm_convolve_wrapper_s8(
            &ctx, &params, &quant,
            &input_dims, buf_b,
            &filter_dims, conv1_weights,
            &bias_dims, conv1_bias,
            &output_dims, buf_a);
        if (status != ARM_CMSIS_NN_SUCCESS) return status;
    }

    /* Layer 3: MaxPool 2x2 — 64x64x32 -> 32x32x32 */
    {
        cmsis_nn_pool_params params = {
            .stride    = {.w = 2, .h = 2},
            .padding   = {.w = 0, .h = 0},
            .activation = {.min = -128, .max = 127}
        };
        cmsis_nn_dims input_dims  = {.n = 1, .h = 64, .w = 64, .c = 32};
        cmsis_nn_dims filter_dims = {.n = 1, .h = 2, .w = 2, .c = 1};
        cmsis_nn_dims output_dims = {.n = 1, .h = 32, .w = 32, .c = 32};
        ctx.size = 0;
        status = arm_max_pool_s8(&ctx, &params,
            &input_dims, buf_a, &filter_dims, &output_dims, buf_b);
        if (status != ARM_CMSIS_NN_SUCCESS) return status;
    }

    /* Layer 4: Conv2D 3x3, 32->64, pad=1, fused ReLU */
    {
        cmsis_nn_conv_params params = {
            .input_offset  = -CONV1_OUT_ZP,
            .output_offset = CONV2_OUT_ZP,
            .stride  = {.w = 1, .h = 1},
            .padding = {.w = 1, .h = 1},
            .dilation = {.w = 1, .h = 1},
            .activation = {.min = -128, .max = 127}
        };
        cmsis_nn_dims input_dims  = {.n = 1, .h = 32, .w = 32, .c = 32};
        cmsis_nn_dims filter_dims = {.n = 64, .h = 3, .w = 3, .c = 32};
        cmsis_nn_dims bias_dims   = {.n = 1, .h = 1, .w = 1, .c = 64};
        cmsis_nn_dims output_dims = {.n = 1, .h = 32, .w = 32, .c = 64};
        cmsis_nn_per_channel_quant_params quant = {
            .multiplier = (int32_t *)conv2_output_mult,
            .shift      = (int32_t *)conv2_output_shift
        };
        ctx.size = arm_convolve_wrapper_s8_get_buffer_size(
            &params, &input_dims, &filter_dims, &output_dims);
        status = arm_convolve_wrapper_s8(
            &ctx, &params, &quant,
            &input_dims, buf_b,
            &filter_dims, conv2_weights,
            &bias_dims, conv2_bias,
            &output_dims, buf_a);
        if (status != ARM_CMSIS_NN_SUCCESS) return status;
    }

    /* Layer 5: MaxPool 2x2 — 32x32x64 -> 16x16x64 */
    {
        cmsis_nn_pool_params params = {
            .stride    = {.w = 2, .h = 2},
            .padding   = {.w = 0, .h = 0},
            .activation = {.min = -128, .max = 127}
        };
        cmsis_nn_dims input_dims  = {.n = 1, .h = 32, .w = 32, .c = 64};
        cmsis_nn_dims filter_dims = {.n = 1, .h = 2, .w = 2, .c = 1};
        cmsis_nn_dims output_dims = {.n = 1, .h = 16, .w = 16, .c = 64};
        ctx.size = 0;
        status = arm_max_pool_s8(&ctx, &params,
            &input_dims, buf_a, &filter_dims, &output_dims, buf_b);
        if (status != ARM_CMSIS_NN_SUCCESS) return status;
    }

    /* Layer 6: Conv2D 3x3, 64->128, pad=1, fused ReLU */
    {
        cmsis_nn_conv_params params = {
            .input_offset  = -CONV2_OUT_ZP,
            .output_offset = CONV3_OUT_ZP,
            .stride  = {.w = 1, .h = 1},
            .padding = {.w = 1, .h = 1},
            .dilation = {.w = 1, .h = 1},
            .activation = {.min = -128, .max = 127}
        };
        cmsis_nn_dims input_dims  = {.n = 1, .h = 16, .w = 16, .c = 64};
        cmsis_nn_dims filter_dims = {.n = 128, .h = 3, .w = 3, .c = 64};
        cmsis_nn_dims bias_dims   = {.n = 1, .h = 1, .w = 1, .c = 128};
        cmsis_nn_dims output_dims = {.n = 1, .h = 16, .w = 16, .c = 128};
        cmsis_nn_per_channel_quant_params quant = {
            .multiplier = (int32_t *)conv3_output_mult,
            .shift      = (int32_t *)conv3_output_shift
        };
        ctx.size = arm_convolve_wrapper_s8_get_buffer_size(
            &params, &input_dims, &filter_dims, &output_dims);
        status = arm_convolve_wrapper_s8(
            &ctx, &params, &quant,
            &input_dims, buf_b,
            &filter_dims, conv3_weights,
            &bias_dims, conv3_bias,
            &output_dims, buf_a);
        if (status != ARM_CMSIS_NN_SUCCESS) return status;
    }

    /* Layer 7: MaxPool 2x2 — 16x16x128 -> 8x8x128 */
    {
        cmsis_nn_pool_params params = {
            .stride    = {.w = 2, .h = 2},
            .padding   = {.w = 0, .h = 0},
            .activation = {.min = -128, .max = 127}
        };
        cmsis_nn_dims input_dims  = {.n = 1, .h = 16, .w = 16, .c = 128};
        cmsis_nn_dims filter_dims = {.n = 1, .h = 2, .w = 2, .c = 1};
        cmsis_nn_dims output_dims = {.n = 1, .h = 8, .w = 8, .c = 128};
        ctx.size = 0;
        status = arm_max_pool_s8(&ctx, &params,
            &input_dims, buf_a, &filter_dims, &output_dims, buf_b);
        if (status != ARM_CMSIS_NN_SUCCESS) return status;
    }

    /* Layer 8: Global Average Pooling 8x8 with requantization
     *   Conv3 output: scale=0.1947, zp=-128
     *   ReduceMean output: scale=0.00667, zp=-128
     *   We compute avgpool + requant in one step. */
    {
        static const float rescale = CONV3_OUT_SCALE / (64.0f * AVGPOOL_OUT_SCALE);
        for (int c = 0; c < 128; c++) {
            int32_t sum = 0;
            for (int pos = 0; pos < 64; pos++) {
                sum += (int32_t)buf_b[pos * 128 + c];
            }
            int32_t q = (int32_t)roundf((float)(sum + 8192) * rescale) - 128;
            if (q < -128) q = -128;
            if (q > 127)  q = 127;
            buf_a[c] = (int8_t)q;
        }
    }

    /* Layer 9: FC 128 -> 6 */
    {
        cmsis_nn_fc_params params = {
            .input_offset  = -AVGPOOL_OUT_ZP,
            .filter_offset = 0,
            .output_offset = MODEL_OUTPUT_ZP,
            .activation    = {.min = -128, .max = 127}
        };
        cmsis_nn_dims input_dims  = {.n = 1, .h = 1, .w = 1, .c = 128};
        cmsis_nn_dims filter_dims = {.n = 6, .h = 1, .w = 1, .c = 128};
        cmsis_nn_dims bias_dims   = {.n = 1, .h = 1, .w = 1, .c = 6};
        cmsis_nn_dims output_dims = {.n = 1, .h = 1, .w = 1, .c = 6};
        cmsis_nn_per_channel_quant_params quant = {
            .multiplier = (int32_t *)fc_output_mult,
            .shift      = (int32_t *)fc_output_shift
        };
        ctx.buf = scratch_buf;
        ctx.size = sizeof(scratch_buf);
        status = arm_fully_connected_per_channel_s8(
            &ctx, &params, &quant,
            &input_dims, buf_a,
            &filter_dims, fc_weights,
            &bias_dims, fc_bias,
            &output_dims, output);
    }

    return status;
}

/* --------------------------------------------------------------------------
 * CMSIS-NN preprocessing
 * -------------------------------------------------------------------------- */
static int acquire_and_process_data_cmsis(void)
{
    static uint8_t raw_input[MODEL_INPUT_SIZE];

    __HAL_UART_CLEAR_OREFLAG(&huart1);
    __HAL_UART_CLEAR_FEFLAG(&huart1);
    __HAL_UART_CLEAR_NEFLAG(&huart1);

    uint8_t ready_in_msg[] = "READY_IN\r\n";
    HAL_UART_Transmit(&huart1, ready_in_msg, sizeof(ready_in_msg) - 1, HAL_MAX_DELAY);

    HAL_StatusTypeDef status = HAL_UART_Receive(&huart1, raw_input,
                                                (uint16_t)MODEL_INPUT_SIZE, HAL_MAX_DELAY);
    if (status != HAL_OK) return -1;

    timer_pre_start = DWT->CYCCNT;

    static const float combined_scale[3] = {
        1.0f / (255.0f * 0.3270f * 0.0131675974f),
        1.0f / (255.0f * 0.3131f * 0.0131675974f),
        1.0f / (255.0f * 0.3042f * 0.0131675974f),
    };
    static const float combined_offset[3] = {
        (-0.3912f / (0.3270f * 0.0131675974f)) + (-37.0f),
        (-0.3612f / (0.3131f * 0.0131675974f)) + (-37.0f),
        (-0.3425f / (0.3042f * 0.0131675974f)) + (-37.0f),
    };

    const int total_pixels = OUT_SIZE * OUT_SIZE;
    for (int i = 0; i < total_pixels; i++) {
        for (int c = 0; c < 3; c++) {
            float val = raw_input[i * 3 + c] * combined_scale[c] + combined_offset[c];
            int32_t q = (int32_t)roundf(val);
            if (q < -128) q = -128;
            if (q > 127)  q = 127;
            model_input_buf[i * 3 + c] = (int8_t)q;
        }
    }

    timer_pre_end = DWT->CYCCNT;
    return 0;
}

/* --------------------------------------------------------------------------
 * CMSIS-NN postprocessing
 * -------------------------------------------------------------------------- */
static int post_process_cmsis(void)
{
    timer_post_start = DWT->CYCCNT;

    int pred_class = 0;
    int8_t max_val = model_output_buf[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (model_output_buf[i] > max_val) {
            max_val = model_output_buf[i];
            pred_class = i;
        }
    }

    float dequant[NUM_CLASSES];
    for (int i = 0; i < NUM_CLASSES; i++) {
        dequant[i] = ((float)model_output_buf[i] - (float)MODEL_OUTPUT_ZP) * MODEL_OUTPUT_SCALE;
    }
    float max_f = dequant[pred_class];
    float sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        sum += expf(dequant[i] - max_f);
    }
    float confidence = 1.0f / sum;

    timer_post_end = DWT->CYCCNT;

    uint32_t timer_pre   = timer_pre_end   - timer_pre_start;
    uint32_t timer_infer = timer_infer_end - timer_infer_start;
    uint32_t timer_post  = timer_post_end  - timer_post_start;
    uint32_t timer_all   = timer_post_end  - timer_pre_start;

    printf("DEBUG: Post-Process done. Sending output to HOST.\r\n");

    uint8_t ready_out_msg[] = "READY_OUT\r\n";
    HAL_UART_Transmit(&huart1, ready_out_msg, 11, HAL_MAX_DELAY);

    HAL_UART_Transmit(&huart1, (uint8_t *)&pred_class,  sizeof(int),      HAL_MAX_DELAY);
    HAL_UART_Transmit(&huart1, (uint8_t *)&confidence,  sizeof(float),    HAL_MAX_DELAY);
    HAL_UART_Transmit(&huart1, (uint8_t *)&timer_pre,   sizeof(uint32_t), HAL_MAX_DELAY);
    HAL_UART_Transmit(&huart1, (uint8_t *)&timer_infer, sizeof(uint32_t), HAL_MAX_DELAY);
    HAL_UART_Transmit(&huart1, (uint8_t *)&timer_post,  sizeof(uint32_t), HAL_MAX_DELAY);
    HAL_UART_Transmit(&huart1, (uint8_t *)&timer_all,   sizeof(uint32_t), HAL_MAX_DELAY);

    return 0;
}

/* --------------------------------------------------------------------------
 * Entry points — CMSIS-NN backend
 * -------------------------------------------------------------------------- */
void MX_X_CUBE_AI_Init(void)
{
    printf("\r\nCMSIS-NN INT8 Model - initialization\r\n");
    /* Nothing to initialise — weights are in Flash, buffers are static */
}

void MX_X_CUBE_AI_Process(void)
{
    int res = -1;

    printf("CMSIS-NN - run - main loop\r\n");

    do {
        res = acquire_and_process_data_cmsis();

        if (res == 0) {
            printf("DEBUG: Starting CMSIS-NN Inference...\r\n");
            timer_infer_start = DWT->CYCCNT;

            arm_cmsis_nn_status nn_status = cmsis_nn_run(
                model_input_buf, model_output_buf);

            timer_infer_end = DWT->CYCCNT;
            printf("DEBUG: CMSIS-NN Inference Done\r\n");

            if (nn_status != ARM_CMSIS_NN_SUCCESS) {
                printf("ERROR: CMSIS-NN inference failed (%d)\r\n", nn_status);
                res = -1;
            }
        }

        if (res == 0)
            res = post_process_cmsis();
    } while (res == 0);

    if (res) {
        printf("Process has FAILED\r\n");
    }
}

#endif /* USE_BACKEND_INT8_CMSIS */

#ifdef __cplusplus
}
#endif
