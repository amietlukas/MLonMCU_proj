/* classifier_cam_app.c
 *
 * On-board capture + inference loop for B-CAMS-OMV (OV5640) on
 * B-U585I-IOT02A. The MCU runs continuously: capture frame, downsample
 * to 160x120 grayscale, run inference, stream image + result over UART.
 *
 * Wire protocol (per frame, repeats forever):
 *   MCU -> "FRAME\r\n"
 *   MCU -> 19200 bytes  (uint8 grayscale, row-major HW, H=120 W=160)
 *   MCU -> 24 bytes "<ifIIII":
 *               i32  pred_class
 *               f32  confidence (softmax probability of pred_class)
 *               u32  t_pre_cycles    (RGB->gray + uint8->int8)
 *               u32  t_infer_cycles  (ai_mnetwork_run)
 *               u32  t_post_cycles   (argmax + softmax)
 *               u32  t_all_cycles    (sum of the three)
 *
 * No host handshake — on reset the MCU prints "BOOT\r\n" then starts the
 * FRAME loop unconditionally. Connect a passive viewer (picocom /
 * host_cam.py) any time.
 *
 * Notes:
 *  - The OV5640 is configured at QQVGA (160x120) YUV422 — the same
 *    resolution and luma representation the model was trained on. The
 *    even bytes of YUV422 are the Y channel, i.e. the grayscale image,
 *    so building the model input is a strided memcpy.
 *  - With X-CUBE-AI 10.2.0 + --allocate-inputs/--allocate-outputs, the
 *    actual I/O buffers live inside the activations pool, so we read
 *    them via s_report.inputs[0].data / .outputs[0].data — NOT the
 *    cached return of inputs_get/outputs_get.
 */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "main.h"
#include "usart.h"
#include "constants.h"
#include "classifier_cam_app.h"

#include "ai_platform.h"
#include "big_net_pruned_int8.h"
#include "big_net_pruned_int8_data.h"
#include "app_x-cube-ai.h"

/* ST BSP for B-U585I-IOT02A. Provides BSP_CAMERA_Init/Start/Stop and
 * the BSP_CAMERA_FrameEventCallback weak hook. */
#include "b_u585i_iot02a_camera.h"

/* ------------------------------------------------------------------ */
/* AI runtime state                                                   */
/* ------------------------------------------------------------------ */
static ai_handle         s_net = AI_HANDLE_NULL;
static ai_network_report s_report;

/* ------------------------------------------------------------------ */
/* Camera state                                                       */
/* ------------------------------------------------------------------ */

/* DMA-target frame buffer. Must be 32-bit aligned. DMA writes into this
 * continuously in circular mode, so to avoid tearing we snapshot it to
 * s_cam_snapshot at the start of each frame's processing. */
AI_ALIGNED(32)
static uint8_t s_cam_buffer[CAM_FRAME_BYTES];

/* Stable copy of one frame, owned by the CPU. Captured during the
 * vertical-blanking gap between camera frames so it doesn't tear. */
AI_ALIGNED(32)
static uint8_t s_cam_snapshot[CAM_FRAME_BYTES];

/* Downsampled grayscale image fed to the model and shipped to the host. */
AI_ALIGNED(4)
static uint8_t s_gray_buffer[MODEL_INPUT_BYTES];

volatile uint8_t g_frame_ready = 0;

/* ------------------------------------------------------------------ */
/* UART helpers                                                       */
/* ------------------------------------------------------------------ */
static inline uint32_t dwt_now(void) { return DWT->CYCCNT; }

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
/* YUV422 (160x120) -> Grayscale (160x120)                            */
/*                                                                    */
/* OV5640 YUV422 byte order is Y0 U Y1 V Y2 U Y3 V ... so the Y       */
/* (luminance) samples are exactly the even bytes of the frame.       */
/* ------------------------------------------------------------------ */
static void yuv422_extract_y(const uint8_t *cam, uint8_t *gray)
{
    for (int i = 0; i < MODEL_INPUT_BYTES; i++) {
        gray[i] = cam[2 * i];
    }
}

/* ------------------------------------------------------------------ */
/* Pre / post-processing (1-channel int8)                             */
/* ------------------------------------------------------------------ */
static void preprocess(const uint8_t *gray, int8_t *dst)
{
    for (int i = 0; i < MODEL_INPUT_BYTES; i++) {
        dst[i] = (int8_t)((int)gray[i] - 128);
    }
}

/* ------------------------------------------------------------------ */
/* Bluetooth command mapping — drives the paul_car Arduino sketch.    */
/*                                                                    */
/* paul_car.ino reads single ASCII chars over its hardware UART       */
/* (bridged via HC-05/06 at 9600 baud) and interprets them as:        */
/*   '0' STOP  '1' FORWARD  '2' FWD-RIGHT  '3' FWD-LEFT               */
/*   '4' BACKWARD  '5' OTHER (currently == STOP)                      */
/*                                                                    */
/* Gesture -> command mapping used here:                              */
/*   palm   '0' STOP                                                  */
/*   rock   '1' FORWARD                                               */
/*   pinkie '3' FWD-LEFT                                              */
/*   one    '2' FWD-RIGHT                                             */
/*   fist   '4' BACKWARD                                              */
/*   others '0' STOP                                                  */
/*                                                                    */
/* The U5 sends one byte over USART3 (PA7 TX) per inference. Wire     */
/* PA7 to the HC-05/06 RXD; the U5-side module is the master, paired  */
/* and bound to the slave on the Arduino side.                        */
/*                                                                    */
/* Edit the gesture->command map here to taste. Anything below the    */
/* confidence floor is sent as '5' so a flickery low-conf prediction  */
/* doesn't slam the car between directions. */
#define CMD_MIN_CONF  0.5f

static char prediction_to_cmd(int pred_class, float confidence)
{
    if (confidence < CMD_MIN_CONF) return '0';   /* low-conf -> STOP */

    switch (pred_class) {
        case 0: return '0';   /* palm   -> STOP        */
        case 1: return '1';   /* rock   -> FORWARD     */
        case 2: return '3';   /* pinkie -> FWD-LEFT    */
        case 3: return '2';   /* one    -> FWD-RIGHT   */
        case 4: return '4';   /* fist   -> BACKWARD    */
        case 5: return '0';   /* others -> STOP        */
        default: return '0';
    }
}

static void bt_send_cmd(char cmd)
{
    HAL_UART_Transmit(&huart3, (uint8_t *)&cmd, 1, HAL_MAX_DELAY);
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
/* BSP frame-ready hook (weak symbol overridden here)                 */
/* ------------------------------------------------------------------ */
volatile uint32_t g_frame_count = 0;
volatile uint32_t g_vsync_count = 0;
volatile uint32_t g_line_count  = 0;
volatile uint32_t g_error_count = 0;

void BSP_CAMERA_FrameEventCallback(uint32_t Instance)
{
    (void)Instance;
    g_frame_ready = 1;
    g_frame_count++;
}

void BSP_CAMERA_VsyncEventCallback(uint32_t Instance)
{
    (void)Instance;
    g_vsync_count++;
}

void BSP_CAMERA_LineEventCallback(uint32_t Instance)
{
    (void)Instance;
    g_line_count++;
}

void BSP_CAMERA_ErrorCallback(uint32_t Instance)
{
    (void)Instance;
    g_error_count++;
}

/* ------------------------------------------------------------------ */
/* Init                                                               */
/* ------------------------------------------------------------------ */
void classifier_cam_init(void)
{
    /* --- AI network ------------------------------------------------ */
    ai_error err = ai_mnetwork_create(AI_BIG_NET_PRUNED_INT8_MODEL_NAME, &s_net, NULL);
    if (err.type != AI_ERROR_NONE) { uart_send_str("CREATE_FAIL\r\n"); Error_Handler(); }

    if (!ai_mnetwork_get_report(s_net, &s_report)) { uart_send_str("REPORT_FAIL\r\n"); Error_Handler(); }
    if (!ai_mnetwork_init(s_net))                  { uart_send_str("INIT_FAIL\r\n");   Error_Handler(); }
    if (!ai_mnetwork_get_report(s_net, &s_report)) { uart_send_str("REPORT2_FAIL\r\n"); Error_Handler(); }

    /* --- Camera ---------------------------------------------------- */
    if (BSP_CAMERA_Init(0, CAMERA_R160x120, CAMERA_PF_YUV422) != BSP_ERROR_NONE) {
        uart_send_str("CAM_INIT_FAIL\r\n");
        Error_Handler();
    }
}

/* ------------------------------------------------------------------ */
/* Capture + infer loop                                               */
/* ------------------------------------------------------------------ */
void classifier_cam_process(void)
{
    /* No sync handshake — the MCU drives the BT command on its own and the
     * host (host_cam.py / picocom) is just a passive viewer. */
    uart_send_str("BOOT\r\n");

    /* Continuous-mode capture: BSP retriggers DMA on each VSYNC, and
     * BSP_CAMERA_FrameEventCallback flips g_frame_ready when a full
     * frame has landed in s_cam_buffer. */
    int32_t cam_rc = BSP_CAMERA_Start(0, s_cam_buffer, CAMERA_MODE_CONTINUOUS);
    {
        char buf[48];
        int n = snprintf(buf, sizeof(buf), "CAM_START_RC=%ld\r\n", (long)cam_rc);
        uart_send_bytes(buf, (uint32_t)n);
    }
    if (cam_rc != BSP_ERROR_NONE) {
        uart_send_str("CAM_START_FAIL\r\n");
        Error_Handler();
    }

    /* Heartbeat. Every ~0.5s of spinning at 160MHz with no frame, dump the
     * IRQ counters so we can see whether DCMI is producing anything at all.
     * If frame_count stays 0 but vsync_count/line_count > 0 → DMA isn't
     * fanning the data through. If everything is 0 → DCMI isn't seeing
     * a clocked signal (camera power/reset/clock issue). */
    uint32_t hb_last = dwt_now();
    const uint32_t hb_period = 80000000u;  /* ~0.5s @ 160 MHz */

    for (;;) {
        while (!g_frame_ready) {
            if ((uint32_t)(dwt_now() - hb_last) > hb_period) {
                hb_last = dwt_now();
                char buf[80];
                int n = snprintf(buf, sizeof(buf),
                    "HB frame=%lu vsync=%lu line=%lu err=%lu\r\n",
                    (unsigned long)g_frame_count,
                    (unsigned long)g_vsync_count,
                    (unsigned long)g_line_count,
                    (unsigned long)g_error_count);
                uart_send_bytes(buf, (uint32_t)n);
            }
        }
        g_frame_ready = 0;

        const uint32_t t0 = dwt_now();

        /* Snapshot the camera buffer first — DMA keeps writing into
         * s_cam_buffer in circular mode, and a 158 ms inference will
         * miss several frames worth of writes. memcpy of 38 KB at
         * 160 MHz is ~150 us, well within the vertical blanking gap. */
        memcpy(s_cam_snapshot, s_cam_buffer, CAM_FRAME_BYTES);

        yuv422_extract_y(s_cam_snapshot, s_gray_buffer);

        int8_t *in_data  = (int8_t *)s_report.inputs[0].data;
        int8_t *out_data = (int8_t *)s_report.outputs[0].data;

        preprocess(s_gray_buffer, in_data);
        const uint32_t t_after_pre = dwt_now();

        ai_i32 nb = ai_mnetwork_run(s_net, s_report.inputs, s_report.outputs);
        if (nb != 1) { uart_send_str("RUN_FAIL\r\n"); Error_Handler(); }
        const uint32_t t_after_inf = dwt_now();

        float conf = 0.0f;
        int pred = postprocess(out_data, &conf);

        const uint32_t t1 = dwt_now();

        /* Drive the car *before* the slow host stream — at 9600 baud
         * this single byte takes ~1 ms, vs the ~209 ms image upload. */
        char cmd = prediction_to_cmd(pred, conf);
        bt_send_cmd(cmd);

        /* Human-readable status line on USART1 (ST-LINK VCP).
         * Always emitted so you can see in picocom what gesture was
         * detected and which BT byte got pushed to the car. */
        static const char *GESTURE_NAME[NUM_CLASSES] = {
            "palm", "rock", "pinkie", "one", "fist", "others"
        };
        static const char *CMD_NAME[6] = {
            "STOP", "FORWARD", "FWD-RIGHT", "FWD-LEFT", "BACKWARD", "OTHER"
        };
        char line[96];
        int cmd_idx = (cmd >= '0' && cmd <= '5') ? (cmd - '0') : 0;
        int nch = snprintf(line, sizeof(line),
                           "PRED %s conf=%.2f -> BT '%c' (%s) | infer=%lums\r\n",
                           GESTURE_NAME[pred],
                           (double)conf,
                           cmd,
                           CMD_NAME[cmd_idx],
                           (unsigned long)((t_after_inf - t_after_pre) / (CPU_FREQ_HZ_GUESS / 1000)));
        uart_send_bytes(line, (uint32_t)nch);

#if STREAM_FRAMES_TO_HOST
        /* Stream frame to host: header, image, then result struct. */
        uart_send_str("FRAME\r\n");
        uart_send_bytes(s_gray_buffer, MODEL_INPUT_BYTES);

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
#endif
    }
}
