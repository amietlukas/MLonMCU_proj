/* classifier_app.c
 *
 * Custom UART-driven inference loop for host.py.
 *
 * Protocol (per inference, repeats forever):
 *   MCU -> "READY_IN\r\n"
 *   host -> 19200 raw bytes (uint8 grayscale, row-major HW, H=120, W=160)
 *   MCU  -> "READY_OUT\r\n"
 *   MCU  -> 24 bytes binary "<ifIIII":
 *               i32  pred_class
 *               f32  confidence (softmax probability of pred_class)
 *               u32  t_pre_cycles
 *               u32  t_infer_cycles
 *               u32  t_post_cycles
 *               u32  t_all_cycles
 *
 * NOTE: The bignet_pruned ONNX has mean/std normalization fused in and was
 * quantized with INPUT_SCALE=1.0 / INPUT_ZP=-128. Preprocessing on the MCU is
 * therefore just a uint8 -> int8 shift by 128.
 *
 * Network setup uses the multi-network registry (ai_mnetwork_create + init),
 * and inference reads/writes through s_report.inputs[0].data /
 * s_report.outputs[0].data. The cached return value of ai_..._inputs_get /
 * outputs_get does NOT alias the buffer the runtime actually uses for I/O on
 * X-CUBE-AI 10.2.0 with --allocate-inputs/--allocate-outputs.
 */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "main.h"
#include "usart.h"
#include "constants.h"
#include "classifier_app.h"

#include "ai_platform.h"
#include "small_net_int8.h"
#include "small_net_int8_data.h"
#include "app_x-cube-ai.h"  /* ai_mnetwork_* + AI_SMALL_NET_INT8_MODEL_NAME */

/* ------------------------------------------------------------------ */
/* AI runtime state                                                   */
/* ------------------------------------------------------------------ */
static ai_handle        s_net = AI_HANDLE_NULL;
static ai_network_report s_report;

AI_ALIGNED(4)
static uint8_t s_rx_buffer[MODEL_INPUT_BYTES];

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */
static inline uint32_t dwt_now(void)   { return DWT->CYCCNT; }

static void uart_send_str(const char *s)
{
    HAL_UART_Transmit(&huart1, (uint8_t *)s, (uint16_t)strlen(s), HAL_MAX_DELAY);
}

static void uart_send_bytes(const void *buf, uint32_t len)
{
    HAL_UART_Transmit(&huart1, (uint8_t *)buf, (uint16_t)len, HAL_MAX_DELAY);
}

static void uart_recv_bytes(void *buf, uint32_t len)
{
    HAL_UART_Receive(&huart1, (uint8_t *)buf, (uint16_t)len, HAL_MAX_DELAY);
}

/* ------------------------------------------------------------------ */
/* Pre / post-processing (1-channel int8)                             */
/* ------------------------------------------------------------------ */

static void preprocess(const uint8_t *rx, int8_t *dst)
{
    for (int i = 0; i < MODEL_INPUT_BYTES; i++) {
        dst[i] = (int8_t)((int)rx[i] - 128);
    }
}

static int postprocess(const int8_t *out_q, float *confidence)
{
    float logits[NUM_CLASSES];
    int   pred = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        logits[i] = ((float)out_q[i] - (float)OUTPUT_ZP) * OUTPUT_SCALE;
        if (logits[i] > logits[pred]) pred = i;
    }
    const float max_l = logits[pred];
    float sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        sum += expf(logits[i] - max_l);
    }
    *confidence = 1.0f / sum;
    return pred;
}

/* ------------------------------------------------------------------ */
/* Init — same path aiValidation uses                                 */
/* ------------------------------------------------------------------ */
void classifier_init(void)
{
    /* 1. Create the int8 network instance via the multi-network registry. */
    ai_error err = ai_mnetwork_create(AI_SMALL_NET_INT8_MODEL_NAME, &s_net, NULL);
    if (err.type != AI_ERROR_NONE) { uart_send_str("CREATE_FAIL\r\n"); Error_Handler(); }

    /* 2. Pull the report (we need report.inputs / report.outputs as the
     *    canonical I/O buffer descriptors for ai_mnetwork_run). */
    if (!ai_mnetwork_get_report(s_net, &s_report)) { uart_send_str("REPORT_FAIL\r\n"); Error_Handler(); }

    /* 3. Initialize. ai_mnetwork_init walks the network entry's activations
     *    list and binds the shared activation pool internally. */
    if (!ai_mnetwork_init(s_net)) { uart_send_str("INIT_FAIL\r\n"); Error_Handler(); }

    /* 4. Refresh report so report.inputs[i].data / report.outputs[i].data
     *    reflect the post-init pointers (--allocate-inputs/-outputs land them
     *    inside the activations pool). */
    if (!ai_mnetwork_get_report(s_net, &s_report)) { uart_send_str("REPORT2_FAIL\r\n"); Error_Handler(); }
}

/* ------------------------------------------------------------------ */
/* Main inference loop                                                */
/* ------------------------------------------------------------------ */
void classifier_process(void)
{
    /* Boot handshake: wait for a single byte from the host before the first
     * inference. This avoids losing the first READY_IN to the host's
     * reset_input_buffer() drain after it opens the serial port. */
    uart_send_str("BOOT\r\n");
    uint8_t sync = 0;
    uart_recv_bytes(&sync, 1);

    for (;;) {
        uart_send_str("READY_IN\r\n");
        uart_recv_bytes(s_rx_buffer, MODEL_INPUT_BYTES);

        const uint32_t t0 = dwt_now();

        int8_t *in_data  = (int8_t *)s_report.inputs[0].data;
        int8_t *out_data = (int8_t *)s_report.outputs[0].data;

        preprocess(s_rx_buffer, in_data);
        const uint32_t t_after_pre = dwt_now();

        ai_i32 nb = ai_mnetwork_run(s_net, s_report.inputs, s_report.outputs);
        if (nb != 1) { uart_send_str("RUN_FAIL\r\n"); Error_Handler(); }
        const uint32_t t_after_inf = dwt_now();

        float conf = 0.0f;
        int pred = postprocess(out_data, &conf);

        const uint32_t t1 = dwt_now();

        uart_send_str("READY_OUT\r\n");

        struct __attribute__((packed)) {
            int32_t  pred_class;
            float    confidence;
            uint32_t t_pre;
            uint32_t t_infer;
            uint32_t t_post;
            uint32_t t_all;
        } resp = {
            .pred_class = (int32_t)pred,
            .confidence = conf,
            .t_pre      = t_after_pre - t0,
            .t_infer    = t_after_inf - t_after_pre,
            .t_post     = t1          - t_after_inf,
            .t_all      = t1          - t0,
        };
        uart_send_bytes(&resp, sizeof(resp));
    }
}
