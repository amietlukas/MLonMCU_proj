 /**
 ******************************************************************************
 * @file    main.c
 * @author  GPM Application Team
 *
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2023 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
#include <string.h>
#include <unistd.h>

#include "cmw_camera.h"
#include "app_config.h"
#if APP_UVC
#include "scrl.h"
#include "stm32_lcd.h"
#include "stm32_lcd_ex.h"
#include "stlogo.h"
#endif
#include "stm32n6xx_nucleo_bus.h"
#include "stm32n6xx_nucleo_xspi.h"
#include "stm32n6xx_nucleo.h"
#include "app_fuseprogramming.h"
#include "app_postprocess.h"
#include "yolo_postproc.h"   /* proven balldet 3-head decoder (from BallDetector_N6) */
#include "stai.h"
#include "stai_network.h"
#include "app_camerapipeline.h"
#include "main.h"
#include <stdio.h>
#include "crop_img.h"
#include "motor_control.h"
#include "servo_control.h"
#include "ball_tracker.h"

/* Hardware-isolation test: 1 = hold servo dead-static at boot angle (90deg),
 * never touch CCR1 again. If it still jitters, the cause is 100% hardware. */
#ifndef SERVO_HOLD_TEST
#define SERVO_HOLD_TEST   0
#endif

/* Decode threshold before NMS. Keep this lower than the final tracking gate
 * while debugging live camera/preprocess issues, otherwise yolo_postprocess()
 * can return nb_detect=0 before the servo logic sees anything. */
#ifndef YOLO_DECODE_CONF
#define YOLO_DECODE_CONF  0.50f
#endif

/* Simple (non-tracker) servo control: a PI controller in the velocity-command
 * form. s_us accumulates the command, so GAIN_US*err is INTEGRAL action (zeroes
 * steady-state error for any ball position) and KD_US*(error rate) integrates to
 * a PROPORTIONAL position term that damps the loop. Pure-integral (KD=0) + the
 * ~80ms loop dead-time overshoots and limit-cycles ("swinging"); the KD term and
 * a moderate GAIN_US fix that while STEP_MAX still allows a fast slew for a
 * fast-moving ball. The loop runs at the inference rate (~12 Hz), so top pan
 * speed ~ STEP_MAX[us] * loop_hz / (~11 us/deg) deg/s. */
#define SIMPLE_SIGN      (-1.0f)   /* pan direction toward the ball (verified)   */
#define SIMPLE_SMOOTH     0.50f    /* EMA low-pass on detected x (higher=snappier) */
#define SIMPLE_GAIN_US    220.0f   /* integral gain: us/update per unit error     */
#define SIMPLE_KD_US      350.0f   /* derivative (damping) gain on error rate     */
#define SIMPLE_STEP_MAX   90.0f    /* max us change per update (~95 deg/s @12Hz) */
#define SIMPLE_DEADZONE   0.04f    /* |x-0.5| below this = centered, no move     */

CLASSES_TABLE;

#ifndef APP_GIT_SHA1_STRING
#define APP_GIT_SHA1_STRING "dev"
#endif
#ifndef APP_VERSION_STRING
#define APP_VERSION_STRING "unversioned"
#endif

#if APP_UVC
#define LCD_BG_WIDTH  SCREEN_WIDTH
#define LCD_BG_HEIGHT SCREEN_HEIGHT
#define LCD_FG_WIDTH  SCREEN_WIDTH
#define LCD_FG_HEIGHT SCREEN_HEIGHT

#define LCD_FG_FRAMEBUFFER_SIZE  (LCD_FG_WIDTH * LCD_FG_HEIGHT * 2)

#define UTIL_LCD_COLOR_TRANSPARENT 0

typedef struct
{
  uint32_t X0;
  uint32_t Y0;
  uint32_t XSize;
  uint32_t YSize;
} Rectangle_TypeDef;

/* Lcd Background area */
Rectangle_TypeDef lcd_bg_area = {
#if ASPECT_RATIO_MODE == ASPECT_RATIO_CROP || ASPECT_RATIO_MODE == ASPECT_RATIO_FIT
  .X0 = (LCD_BG_WIDTH - LCD_BG_HEIGHT) / 2,
#else
  .X0 = 0,
#endif
  .Y0 = 0,
  .XSize = 0,
  .YSize = 0,
};

/* Lcd Foreground area */
Rectangle_TypeDef lcd_fg_area = {
#if ASPECT_RATIO_MODE == ASPECT_RATIO_CROP || ASPECT_RATIO_MODE == ASPECT_RATIO_FIT
  .X0 = (LCD_FG_WIDTH - LCD_FG_HEIGHT) / 2,
#else
  .X0 = 0,
#endif
  .Y0 = 0,
  .XSize = 0,
  .YSize = 0,
};

#define NUMBER_COLORS 10
const uint32_t colors[NUMBER_COLORS] = {
    UTIL_LCD_COLOR_GREEN,
    UTIL_LCD_COLOR_RED,
    UTIL_LCD_COLOR_CYAN,
    UTIL_LCD_COLOR_MAGENTA,
    UTIL_LCD_COLOR_YELLOW,
    UTIL_LCD_COLOR_GRAY,
    UTIL_LCD_COLOR_BLACK,
    UTIL_LCD_COLOR_BROWN,
    UTIL_LCD_COLOR_BLUE,
    UTIL_LCD_COLOR_ORANGE
};
#endif /* APP_UVC */

#if APP_UVC
#if POSTPROCESS_TYPE == POSTPROCESS_OD_YOLO_V2_UI
  od_yolov2_pp_static_param_t pp_params;
#elif POSTPROCESS_TYPE == POSTPROCESS_OD_YOLO_V5_UU
  od_yolov5_pp_static_param_t pp_params;
#elif POSTPROCESS_TYPE == POSTPROCESS_OD_YOLO_V8_UI
  od_yolov8_pp_static_param_t pp_params;
#elif POSTPROCESS_TYPE == POSTPROCESS_OD_ST_YOLOX_UI
  od_st_yolox_pp_static_param_t pp_params;
#elif POSTPROCESS_TYPE == POSTPROCESS_OD_SSD_UI
  od_ssd_pp_static_param_t pp_params;
#elif POSTPROCESS_TYPE == POSTPROCESS_OD_ST_YOLOD_UI
  od_yolo_d_pp_static_param_t pp_params;
#elif POSTPROCESS_TYPE == POSTPROCESS_OD_BLAZEFACE_UI
  od_blazeface_pp_static_param_t pp_params;
#else
  #error "PostProcessing type not supported"
#endif
#endif /* APP_UVC */

UART_HandleTypeDef huart1;
UART_HandleTypeDef huart3;   /* HC-06 Bluetooth on Arduino D0/D1 (PD9 RX / PD8 TX) */
volatile int32_t cameraFrameReceived;
static int g_capture_inflight = 0;   /* a pipelined NN-pipe snapshot is running */
stai_ptr nn_in;
void* pp_input;
od_pp_out_t pp_output;
/* balldet decode scratch: pixel-space boxes from yolo_postprocess(), and the
 * normalized boxes Display_NetworkOutput() expects. */
static yolo_box_t        g_yolo_boxes[YOLO_MAX_DET];
static od_pp_outBuffer_t g_od_boxes[YOLO_MAX_DET];
/* Camera delivers HWC RGB888; balldet input is channel-first (CHW). Capture
 * into this scratch, then transpose into nn_in. */
__attribute__ ((aligned (32)))
static uint8_t nn_hwc[STAI_NETWORK_IN_1_SIZE_BYTES];

/* The camera module is mounted rotated 90deg CW, so we capture the frame in the
 * sensor's native (un-stretched) 3:4 orientation -- CAP_W x CAP_H = 288 x 384 --
 * and rotate it 90deg CCW into the model's 384 x 288 (4:3) input. Because the
 * sensor is 4:3 and the model is 4:3, this is a pure rotation: no stretch, balls
 * stay round. Same per-pixel cost as a plain transpose. Also converts HWC->CHW.
 *
 * Dest (model) pixel (mx,my) <- source (capture) pixel (xs,ys):
 *   xs = (CAP_W-1) - my,  ys = mx     (90deg CCW, matches host rot90_ccw). */
#define CAP_W (STAI_NETWORK_IN_1_HEIGHT)   /* 288: capture width  */
#define CAP_H (STAI_NETWORK_IN_1_WIDTH)    /* 384: capture height */

static void rotate_ccw_hwc_to_chw(const uint8_t *hwc, uint8_t *chw)
{
  const uint32_t MW = (uint32_t)STAI_NETWORK_IN_1_WIDTH;    /* 384 */
  const uint32_t MH = (uint32_t)STAI_NETWORK_IN_1_HEIGHT;   /* 288 */
  const uint32_t HW = MW * MH;
  for (uint32_t my = 0; my < MH; ++my) {
    for (uint32_t mx = 0; mx < MW; ++mx) {
      const uint32_t xs  = (CAP_W - 1u) - my;
      const uint32_t ys  = mx;
      const uint8_t *src = hwc + (ys * CAP_W + xs) * 3u;
      const uint32_t p   = my * MW + mx;
      chw[p]          = src[0];   /* R plane */
      chw[HW + p]     = src[1];   /* G plane */
      chw[2 * HW + p] = src[2];   /* B plane */
    }
  }
}

/* (legacy straight transpose, unused now -- kept for quick revert) */
static void transpose_hwc_to_chw(const uint8_t *hwc, uint8_t *chw)
{
  const uint32_t HW = (uint32_t)STAI_NETWORK_IN_1_WIDTH * (uint32_t)STAI_NETWORK_IN_1_HEIGHT;
  for (uint32_t p = 0; p < HW; ++p) {
    chw[p]          = hwc[p * 3 + 0];   /* R plane */
    chw[HW + p]     = hwc[p * 3 + 1];   /* G plane */
    chw[2 * HW + p] = hwc[p * 3 + 2];   /* B plane */
  }
}

/* ---- HC-06 Bluetooth RX: interrupt-driven ring buffer -------------------- *
 * Polling USART3 once per (slow, NN-bound) loop iteration dropped bytes and
 * latched on overrun -> commands flaky / receiver stuck. Instead receive in the
 * USART3 ISR into a ring buffer and let the loop drain it, decoupled from the
 * loop period. ORE is cleared so a single overrun never wedges reception. */
#define BT_RING_SZ 64u
/* Each event carries the received byte plus how many CPU cycles the in-ISR
 * motor/servo activation took (receive -> CCR/GPIO written). The loop prints it. */
typedef struct { uint8_t b; uint32_t act_cyc; } bt_evt_t;
static volatile bt_evt_t bt_ring[BT_RING_SZ];
static volatile uint16_t bt_head = 0, bt_tail = 0;

/* Pop one event; returns 1 and fills *evt, or 0 if the ring is empty. */
static int bt_ring_pop(bt_evt_t *evt)
{
  if (bt_tail == bt_head) return 0;
  *evt = bt_ring[bt_tail];
  bt_tail = (uint16_t)((bt_tail + 1u) % BT_RING_SZ);
  return 1;
}

/* Execute a Bluetooth command IMMEDIATELY in ISR context. Motor_Command and
 * Servo_Command are pure register writes (CCR/GPIO, no blocking, no HAL_Delay),
 * so they are ISR-safe and run the instant the byte arrives -- even during the
 * ~70 ms NPU inference, because __WFE wakes the CPU for this IRQ. That removes
 * the up-to-one-loop (~80 ms) latency of servicing commands from the main loop;
 * the loop now only echoes the bytes to the console. */
static void bt_dispatch(uint8_t b)
{
  if (b >= '0' && b <= '5') Motor_Command((char)b);
#if !SERVO_HOLD_TEST
  else if (b=='L'||b=='l'||b=='R'||b=='r'||b=='M'||b=='m'||b=='<'||b=='>') Servo_Command((char)b);
#endif
}

void USART3_IRQHandler(void)
{
  /* Drain the ENTIRE RX FIFO each entry: after the ~70ms inference window the
   * FIFO may hold several queued bytes; reading only one would leave the rest
   * (and eventually overrun). Loop while the FIFO is non-empty. */
  while (__HAL_UART_GET_FLAG(&huart3, UART_FLAG_RXNE)) {
    uint8_t b = (uint8_t)(huart3.Instance->RDR & 0xFFu);   /* pops one FIFO entry */
    uint32_t t0 = DWT->CYCCNT;
    bt_dispatch(b);                                        /* act now */
    uint32_t act_cyc = DWT->CYCCNT - t0;                   /* receive -> motor/servo set */
    uint16_t nh = (uint16_t)((bt_head + 1u) % BT_RING_SZ);
    if (nh != bt_tail) {                                   /* queue for console print */
      bt_ring[bt_head].b = b; bt_ring[bt_head].act_cyc = act_cyc; bt_head = nh;
    }
  }
  if (__HAL_UART_GET_FLAG(&huart3, UART_FLAG_ORE)) {
    __HAL_UART_CLEAR_OREFLAG(&huart3);                    /* recover from overrun */
  }
}

/* Drain pending commands and echo every byte on UART1 (the console VCP) so the
 * actual received traffic is visible while debugging. */
static void process_commands(void)
{
  bt_evt_t e;
  while (bt_ring_pop(&e)) {
    /* Already executed in USART3_IRQHandler. Print byte + how long activation took. */
    char c = (e.b >= 32 && e.b < 127) ? (char)e.b : '.';
    int is_cmd = (e.b >= '0' && e.b <= '5') ||
                 e.b=='L'||e.b=='l'||e.b=='R'||e.b=='r'||e.b=='M'||e.b=='m'||e.b=='<'||e.b=='>';
    uint32_t cpus = SystemCoreClock / 1000000u; if (!cpus) cpus = 1u;
    uint32_t ns = (uint32_t)(((uint64_t)e.act_cyc * 1000u) / cpus);
    if (is_cmd)
      printf("BT rx '%c' (0x%02X) -> motor/servo set in %lu cyc (~%lu ns)\n",
             c, (unsigned)e.b, (unsigned long)e.act_cyc, (unsigned long)ns);
    else
      printf("BT rx '%c' (0x%02X) [no command]\n", c, (unsigned)e.b);
  }
  /* USB VCP: '0'-'5' = wired fallback drive command. */
  uint8_t t = 0;
  if (HAL_UART_Receive(&huart1, &t, 1, 0) == HAL_OK) {
    if (t >= '0' && t <= '5') {   printf("VCP cmd '%c'\n", t); Motor_Command((char)t); }
  }
}

#define ALIGN_TO_16(value) (((value) + 15) & ~15)

/* When NN input dimensions are not a multiple of 16, the DCMIPP output needs cropping */
#if (STAI_NETWORK_IN_1_WIDTH * STAI_NETWORK_IN_1_CHANNEL) != ALIGN_TO_16(STAI_NETWORK_IN_1_WIDTH * STAI_NETWORK_IN_1_CHANNEL)
#define DCMIPP_NN_NEEDS_CROP 1
#define DCMIPP_OUT_NN_LEN (ALIGN_TO_16(STAI_NETWORK_IN_1_WIDTH * STAI_NETWORK_IN_1_CHANNEL) * STAI_NETWORK_IN_1_HEIGHT)
#define DCMIPP_OUT_NN_BUFF_LEN (DCMIPP_OUT_NN_LEN + 32 - DCMIPP_OUT_NN_LEN%32)

__attribute__ ((aligned (32)))
static uint8_t dcmipp_out_nn[DCMIPP_OUT_NN_BUFF_LEN];
#else
#define DCMIPP_NN_NEEDS_CROP 0
#endif

/* model */
STAI_NETWORK_CONTEXT_DECLARE(network_context, STAI_NETWORK_CONTEXT_SIZE)
#if APP_UVC
/* Lcd Background Buffer */
__attribute__ ((aligned (32)))
static uint8_t lcd_bg_buffer[LCD_BG_WIDTH * LCD_BG_HEIGHT * 2];
/* Lcd Foreground Buffer */
__attribute__ ((aligned (32)))
static uint8_t lcd_fg_buffer[2][LCD_FG_WIDTH * LCD_FG_HEIGHT * 2];
static int lcd_fg_buffer_rd_idx;
/* screen buffer */
__attribute__ ((aligned (32)))
static uint8_t screen_buffer[LCD_FG_WIDTH * LCD_FG_HEIGHT * 2];
#endif /* APP_UVC */

static void SystemClock_Config(void);
static void CONSOLE_Config(void);
static void BT_Config(void);
static void NPURam_enable(void);
static void NPUCache_config(void);
#if APP_UVC
static void Display_NetworkOutput(od_pp_out_t *p_postprocess, uint32_t inference_ms);
static void Display_init(void);
static void Display_WelcomeScreen(void);
#endif
static void Security_Config(void);
static void set_clk_sleep_mode(void);
static void IAC_Config(void);
static void Hardware_init(void);
static void NeuralNetwork_init(uint32_t *nn_in_length, stai_ptr *nn_out, stai_size *number_output, int32_t nn_out_len[]);

/* Per-head decode config (count, scale, zero-point, grid, stride) populated from
 * the generated STAI_NETWORK_OUT_* macros so the YOLO decode is fully model-
 * agnostic: 2- or 3-head, any stride set, updated automatically on regenerate. */
static int   g_n_heads;
static float g_head_scale[YOLO_NUM_HEADS];
static int   g_head_zp[YOLO_NUM_HEADS];
static int   g_head_stride[YOLO_NUM_HEADS];
static int   g_head_gw[YOLO_NUM_HEADS];
static int   g_head_gh[YOLO_NUM_HEADS];
static void head_cfg_init(void)
{
  g_n_heads = STAI_NETWORK_OUT_NUM;
  { static const float s[] = STAI_NETWORK_OUT_1_SCALES; static const int o[] = STAI_NETWORK_OUT_1_OFFSETS;
    g_head_scale[0]=s[0]; g_head_zp[0]=o[0]; g_head_gw[0]=STAI_NETWORK_OUT_1_WIDTH; g_head_gh[0]=STAI_NETWORK_OUT_1_HEIGHT; }
#if STAI_NETWORK_OUT_NUM >= 2
  { static const float s[] = STAI_NETWORK_OUT_2_SCALES; static const int o[] = STAI_NETWORK_OUT_2_OFFSETS;
    g_head_scale[1]=s[0]; g_head_zp[1]=o[0]; g_head_gw[1]=STAI_NETWORK_OUT_2_WIDTH; g_head_gh[1]=STAI_NETWORK_OUT_2_HEIGHT; }
#endif
#if STAI_NETWORK_OUT_NUM >= 3
  { static const float s[] = STAI_NETWORK_OUT_3_SCALES; static const int o[] = STAI_NETWORK_OUT_3_OFFSETS;
    g_head_scale[2]=s[0]; g_head_zp[2]=o[0]; g_head_gw[2]=STAI_NETWORK_OUT_3_WIDTH; g_head_gh[2]=STAI_NETWORK_OUT_3_HEIGHT; }
#endif
  for (int i = 0; i < g_n_heads; i++)
    g_head_stride[i] = STAI_NETWORK_IN_1_WIDTH / g_head_gw[i];   /* 8/16/32 */
}


/**
  * @brief  Main program
  * @param  None
  * @retval None
  */
int main(void)
{
  Hardware_init();

  /*** NN Init ****************************************************************/
  uint32_t nn_in_len = 0;
  stai_size number_output = 0;
  stai_ptr nn_out[STAI_NETWORK_OUT_NUM] = {0};
  int32_t nn_out_len[STAI_NETWORK_OUT_NUM] = {0};

  NeuralNetwork_init(&nn_in_len, nn_out, &number_output, nn_out_len);
  head_cfg_init();   /* per-head count/scale/zp/stride/grid from generated macros */

  int ret;

#if APP_UVC
  /*** Post Processing Init ***************************************************/
  stai_network_info info;

  ret = stai_network_get_info(network_context, &info);
  assert(ret == STAI_SUCCESS);
  app_postprocess_init(&pp_params, &info);
#endif

  /*** Camera Init ************************************************************/
  uint32_t pitch_nn = 0;
#if APP_UVC
  CameraPipeline_Init((uint32_t *[2]) {&lcd_bg_area.XSize, &lcd_fg_area.XSize}, (uint32_t *[2]) {&lcd_bg_area.YSize, &lcd_fg_area.YSize}, &pitch_nn);

  Display_init();

  /* Start LCD Display camera pipe stream */
  CameraPipeline_DisplayPipe_Start(lcd_bg_buffer, CMW_MODE_CONTINUOUS);
#else
  CameraPipeline_Init(NULL, NULL, &pitch_nn);
  /* No display pipe in headless. Start the NN pipe CONTINUOUS so the sensor +
   * ISP keep streaming: in SNAPSHOT-only the ISP auto-exposure never converged
   * and every pixel came out 0 (black frames -> no detections). Single-buffered
   * into nn_hwc; minor tearing is fine for detection. */
  CameraPipeline_NNPipe_Start(nn_hwc, CMW_MODE_CONTINUOUS);
  g_capture_inflight = 1;
#endif

  /*** App header *************************************************************/
  printf("========================================\n");
  printf("STM32N6-GettingStarted-ObjectDetection %s (%s)\n", APP_VERSION_STRING, APP_GIT_SHA1_STRING);
  printf("Build date & time: %s %s\n", __DATE__, __TIME__);
#if defined(__GNUC__)
  printf("Compiler: GCC %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(__ICCARM__)
  printf("Compiler: IAR EWARM %d.%d.%d\n", __VER__ / 1000000, (__VER__ / 1000) % 1000 ,__VER__ % 1000);
#else
  printf("Compiler: Unknown\n");
#endif
  printf("HAL: %lu.%lu.%lu\n", __STM32N6xx_HAL_VERSION_MAIN, __STM32N6xx_HAL_VERSION_SUB1, __STM32N6xx_HAL_VERSION_SUB2);
  printf("STEdgeAI Tools: %d.%d.%d\n", STAI_TOOLS_VERSION_MAJOR, STAI_TOOLS_VERSION_MINOR, STAI_TOOLS_VERSION_MICRO);
  printf("NN model: %s\n", STAI_NETWORK_ORIGIN_MODEL_NAME);
  printf("========================================\n");

  /*** Motor control (paul_car RC-car port) **********************************/
  Motor_Init();   /* car stopped; drive with single-char commands '0'..'5' */

  Servo_Init();   /* camera-pan servo -> centers to nominal home on boot */

#if SERVO_HOLD_TEST
  Servo_SetMicros(SERVO_CENTER_US);
#else
  BallTracker_Init();   /* alpha-beta tracker + 50 Hz control tick (TIM7) */
#endif

  /*** App Loop ***************************************************************/
  while (1)
  {
    static unsigned g_lc = 0; int dbg = (g_lc++ < 5);

    /* Bluetooth + VCP commands first, decoupled from the slow NN path. */
    process_commands();

#if BT_MOTOR_ISOLATE
    continue;   /* debug: motor + Bluetooth only; skip camera / NN / UVC */
#endif

    /* Real end-to-end loop rate (camera + rotate + NPU + postproc + UVC). */
    {
      static uint32_t fps_t = 0, fps_c = 0;
      fps_c++;
      static uint32_t tk_last = 0;
      uint32_t nt = HAL_GetTick();
      if (nt - fps_t >= 2000) {
        uint32_t tk = BallTracker_TickCount();
        printf("FPS~%lu (%lums/loop) ctrl~%luHz\n",
               (unsigned long)((fps_c * 1000u) / (nt - fps_t)),
               (unsigned long)((nt - fps_t) / fps_c),
               (unsigned long)(((tk - tk_last) * 1000u) / (nt - fps_t)));
        fps_t = nt; fps_c = 0; tk_last = tk;
      }
    }

    if (dbg) printf("L0 loop iter %u\n", g_lc);
    /* Pipelined capture: the snapshot for THIS frame was kicked off at the end of
     * the previous iteration so it overlapped the NPU inference. Ensure one is in
     * flight (first iteration / after a frame dump), then wait for it.
     * Headless runs the NN pipe CONTINUOUS (started once at init), so no per-loop
     * snapshot start/re-arm there -- the stream is self-sustaining. */
#if APP_UVC
    if (!g_capture_inflight) {
      CameraPipeline_NNPipe_Start(nn_hwc, CMW_MODE_SNAPSHOT);
      g_capture_inflight = 1;
    }
#endif
    CameraPipeline_IspUpdate();

    uint32_t wt0 = HAL_GetTick();
    while (cameraFrameReceived == 0) {};
    cameraFrameReceived = 0;
    g_capture_inflight = 0;
    if (dbg) printf("L2 frame wait %lums\n", (unsigned long)(HAL_GetTick() - wt0));

    uint32_t ts[2] = { 0 };

    /* HWC capture -> CHW NN input (rotated 90deg CCW, or straight transpose). */
    SCB_InvalidateDCache_by_Addr(nn_hwc, sizeof(nn_hwc));
    uint32_t rot_t0 = HAL_GetTick();
#if NN_ROTATE_90
    rotate_ccw_hwc_to_chw(nn_hwc, (uint8_t *)nn_in);   /* 288x384 capture -> 384x288 upright */
#else
    transpose_hwc_to_chw(nn_hwc, (uint8_t *)nn_in);    /* 384x288 capture, no rotation */
#endif
    if (dbg) printf("L2b cvt+CHW: %lums\n", (unsigned long)(HAL_GetTick() - rot_t0));
    SCB_CleanInvalidateDCache_by_Addr(nn_in, nn_in_len);

    /* Kick off the NEXT capture now so it overlaps the inference below (hides the
     * camera latency). Headless uses a CONTINUOUS stream (no re-arm needed). */
#if APP_UVC
    CameraPipeline_NNPipe_Start(nn_hwc, CMW_MODE_SNAPSHOT);
    g_capture_inflight = 1;
#endif

    ts[0] = HAL_GetTick();
    /* run ATON inference */
    ret = stai_network_run(network_context, STAI_MODE_SYNC);
    if (dbg) printf("L3 nn_run ret=%d\n", ret);
    assert(ret == 0);
    ts[1] = HAL_GetTick();

    /* Service Bluetooth/VCP commands again right after the ~70ms inference: bytes
     * that arrived during it are already in the ring (the USART3 ISR runs through
     * inference), so this halves worst-case command latency vs once-per-loop. */
    process_commands();

    /* Decode balldet's 3 heads (p8/p16/p32 = nn_out[0..2], CHW int8
     * [tx,ty,tw,th,tobj]) with the proven decoder, then convert pixel xyxy ->
     * normalized center/wh that Display_NetworkOutput expects. */
#if APP_UVC
    (void)pp_params;
#endif
    for (int i = 0; i < number_output; i++) {
      SCB_InvalidateDCache_by_Addr(nn_out[i], nn_out_len[i]);   /* read fresh NPU output */
    }
    const uint8_t *heads[YOLO_NUM_HEADS] = {0};
    for (int i = 0; i < g_n_heads; i++) heads[i] = (const uint8_t *)nn_out[i];
    int nb = yolo_postprocess(heads, g_n_heads, g_head_scale, g_head_zp,
                              g_head_stride, g_head_gw, g_head_gh,
                              YOLO_DECODE_CONF, 0.25f, g_yolo_boxes, YOLO_MAX_DET);
    for (int i = 0; i < nb; i++) {
      float cx = 0.5f * (g_yolo_boxes[i].x1 + g_yolo_boxes[i].x2);
      float cy = 0.5f * (g_yolo_boxes[i].y1 + g_yolo_boxes[i].y2);
      g_od_boxes[i].x_center    = cx / (float)STAI_NETWORK_IN_1_WIDTH;
      g_od_boxes[i].y_center    = cy / (float)STAI_NETWORK_IN_1_HEIGHT;
      g_od_boxes[i].width       = (g_yolo_boxes[i].x2 - g_yolo_boxes[i].x1) / (float)STAI_NETWORK_IN_1_WIDTH;
      g_od_boxes[i].height      = (g_yolo_boxes[i].y2 - g_yolo_boxes[i].y1) / (float)STAI_NETWORK_IN_1_HEIGHT;
      g_od_boxes[i].conf        = g_yolo_boxes[i].score;
      g_od_boxes[i].class_index = 0;   /* single class: ball */
    }
    pp_output.pOutBuff  = g_od_boxes;
    pp_output.nb_detect = nb;
    if (dbg) printf("L4 postproc nb_detect=%d\n", nb);

#if SERVO_HOLD_TEST
    /* Servo held at the center pulse. Reassert the same CCR value so any stray
     * command/noise cannot leave it away from center while this test is active. */
    {
      static uint32_t hold_log = 0; uint32_t now = HAL_GetTick();
      Servo_SetMicros(SERVO_CENTER_US);
      if (now - hold_log >= 1000) {
        printf("HOLD servo us=%u ang=%d (forced center)\n",
               (unsigned)Servo_GetMicros(), Servo_GetAngle());
        Servo_DumpState();
        hold_log = now;
      }
    }
#elif TRACKER_MODE == TRACKER_ALPHABETA
    /* --- ALPHA-BETA tracker path ----------------------------------------------
     * Feed the best confident ball into the alpha-beta filter + track lifecycle
     * (ball_tracker.c), then run one control tick. The tracker owns the servo
     * angle (Servo_SetAngleF), including velocity prediction and coast through
     * brief detection dropouts. Tune its gains in ball_tracker.c. */
    {
      int present = 0; float ball_x = 0.5f, best = 0.0f, maxscore = 0.0f;
      for (int i = 0; i < nb; i++) {
        if (g_yolo_boxes[i].score > maxscore) maxscore = g_yolo_boxes[i].score;
        if (g_yolo_boxes[i].score < TRACK_CONF_MIN) continue;          /* confidence gate */
        if (!present || g_yolo_boxes[i].score > best) {
          best = g_yolo_boxes[i].score; ball_x = g_od_boxes[i].x_center; present = 1;
        }
      }
      BallTracker_Measure(present, ball_x);   /* deposit detection for the filter */
      BallTracker_ControlTick();              /* alpha-beta + lifecycle + servo   */

      static uint32_t last_log = 0;
      uint32_t now = HAL_GetTick();
      if (dbg || ((nb > 0) && (now - last_log) > 250)) {
        printf("AB   nb=%d top=%d%% present=%d x=%d%% est=%d%% st=%d ang=%d\n",
               nb, (int)(maxscore * 100.0f), present, (int)(ball_x * 100.0f),
               (int)(BallTracker_GetX() * 100.0f), BallTracker_State(),
               Servo_GetAngle());
        last_log = now;
      }
    }
#else
    /* --- SIMPLE control (PI position controller, velocity-command form) -------
     * Take the most-confident ball, low-pass its x, then drive the servo pulse
     * with GAIN_US*err (integral) + KD_US*(error rate) (derivative -> proportional
     * damping). Writes CCR1 directly via Servo_SetMicros. This is the default;
     * toggle to the alpha-beta path with TRACKER_MODE in app_config.h. */
    {
      static float s_xfilt = 0.5f;
      static float s_us    = (float)SERVO_CENTER_US;
      static float s_eprev = 0.0f;   /* previous error, for the damping term */
      static int   s_init  = 0;
      if (!s_init) { s_us = (float)Servo_GetMicros(); s_init = 1; }

      int   present = 0; float ball_x = 0.5f, best = 0.0f, maxscore = 0.0f;
      for (int i = 0; i < nb; i++) {
        if (g_yolo_boxes[i].score > maxscore) maxscore = g_yolo_boxes[i].score;
        if (g_yolo_boxes[i].score < TRACK_CONF_MIN) continue;          /* confidence gate */
        if (!present || g_yolo_boxes[i].score > best) {
          best = g_yolo_boxes[i].score; ball_x = g_od_boxes[i].x_center; present = 1;
        }
      }

      int moved = 0;
      if (present) {
        s_xfilt += SIMPLE_SMOOTH * (ball_x - s_xfilt);                 /* low-pass detection */
        float err  = s_xfilt - 0.5f;
        float derr = err - s_eprev;                                    /* per-update error rate */
        s_eprev = err;
        if (err > SIMPLE_DEADZONE || err < -SIMPLE_DEADZONE) {
          /* PI: integral (GAIN*err) + damping (KD*derr). KD opposes the swing as
           * the ball approaches center, so the integrator stops overshooting. */
          float dus = SIMPLE_SIGN * (SIMPLE_GAIN_US * err + SIMPLE_KD_US * derr);
          if (dus >  SIMPLE_STEP_MAX) dus =  SIMPLE_STEP_MAX;
          if (dus < -SIMPLE_STEP_MAX) dus = -SIMPLE_STEP_MAX;
          s_us += dus;
          if (s_us < (float)SERVO_MIN_US) s_us = (float)SERVO_MIN_US;
          if (s_us > (float)SERVO_MAX_US) s_us = (float)SERVO_MAX_US;
          Servo_SetMicros((uint16_t)(s_us + 0.5f));                    /* -> CCR1 */
          moved = 1;
        }
      } else {
        s_xfilt = 0.5f;                                                /* no ball -> hold servo */
        s_eprev = 0.0f;                                                /* no derivative kick on re-acquire */
      }

      static uint32_t last_log = 0;
      uint32_t now = HAL_GetTick();
      if (dbg || ((nb > 0 || moved) && (now - last_log) > 250)) {
        printf("SMPL nb=%d top=%d%% present=%d x=%d%% us=%d ang=%d%s\n",
               nb, (int)(maxscore * 100.0f), present, (int)(ball_x * 100.0f),
               (int)s_us, Servo_GetAngle(), moved ? " *MOVED*" : "");
        last_log = now;
      }
    }
#endif /* SERVO_HOLD_TEST */

#if APP_UVC
    uint32_t ut0 = HAL_GetTick();
    Display_NetworkOutput(&pp_output, ts[1] - ts[0]);
    if (dbg) printf("L6 uvc %lums (infer %lums)\n",
                    (unsigned long)(HAL_GetTick() - ut0), (unsigned long)(ts[1] - ts[0]));
#else
    if (dbg) printf("L6 uvc skipped (infer %lums)\n", (unsigned long)(ts[1] - ts[0]));
#endif
    /* Discard nn_out region (used by pp_input and pp_outputs variables) to avoid Dcache evictions during nn inference */
    for (int i = 0; i < number_output; i++)
    {
      void *tmp = nn_out[i];
      SCB_InvalidateDCache_by_Addr(tmp, nn_out_len[i]);
    }
  }
}


static void Hardware_init(void)
{
  /* Power on ICACHE */
  MEMSYSCTL->MSCR |= MEMSYSCTL_MSCR_ICACTIVE_Msk;

  /* Set back system and CPU clock source to HSI */
  __HAL_RCC_CPUCLK_CONFIG(RCC_CPUCLKSOURCE_HSI);
  __HAL_RCC_SYSCLK_CONFIG(RCC_SYSCLKSOURCE_HSI);

  HAL_Init();

  SCB_EnableICache();

#if defined(USE_DCACHE)
  /* Power on DCACHE */
  MEMSYSCTL->MSCR |= MEMSYSCTL_MSCR_DCACTIVE_Msk;
  SCB_EnableDCache();
#endif

  SystemClock_Config();

  /* Enable the DWT cycle counter so we can time the in-ISR BT command activation
   * (used by process_commands to print receive -> motor/servo latency). */
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

  CONSOLE_Config();

  BT_Config();   /* HC-06 Bluetooth (USART3 @ 9600 on Arduino D0/D1) */

  NPURam_enable();

  Fuse_Programming();

  NPUCache_config();

  /*** External NOR Flash *********************************************/
  BSP_XSPI_NOR_Init_t NOR_Init;
  NOR_Init.InterfaceMode = BSP_XSPI_NOR_OPI_MODE;
  NOR_Init.TransferRate = BSP_XSPI_NOR_DTR_TRANSFER;
  BSP_XSPI_NOR_Init(0, &NOR_Init);
  BSP_XSPI_NOR_EnableMemoryMappedMode(0);

  /* Set all required IPs as secure privileged */
  Security_Config();

  IAC_Config();
  set_clk_sleep_mode();

}

static void NeuralNetwork_init(uint32_t *nn_in_length, stai_ptr *nn_out, stai_size *number_output, int32_t nn_out_len[])
{
  stai_network_info info;
  int ret;

  /* initialize runtime */
  ret = stai_runtime_init();
  assert(ret == STAI_SUCCESS);
  /* init model instance */
  ret = stai_network_init(network_context);
  assert(ret == STAI_SUCCESS);

  ret = stai_network_get_info(network_context, &info);
  assert(ret == STAI_SUCCESS);
  assert(info.n_inputs == 1);
  *number_output = STAI_NETWORK_OUT_NUM;

  /* Get the input buffer size & address */
  *nn_in_length = info.inputs[0].size_bytes;
  ret = stai_network_get_inputs(network_context, &nn_in, (stai_size *)&info.n_inputs);
  assert(ret == STAI_SUCCESS);

  /* Get the output buffers size & address */
  ret = stai_network_get_outputs(network_context, nn_out, number_output);
  assert(ret == STAI_SUCCESS);
  for (int i = 0; i < *number_output; i++)
  {
    nn_out_len[i] = info.outputs[i].size_bytes;
  }
}

static void NPURam_enable(void)
{
  __HAL_RCC_NPU_CLK_ENABLE();
  __HAL_RCC_NPU_FORCE_RESET();
  __HAL_RCC_NPU_RELEASE_RESET();

  /* Enable NPU RAMs (4x448KB) */
  __HAL_RCC_AXISRAM3_MEM_CLK_ENABLE();
  __HAL_RCC_AXISRAM4_MEM_CLK_ENABLE();
  __HAL_RCC_AXISRAM5_MEM_CLK_ENABLE();
  __HAL_RCC_AXISRAM6_MEM_CLK_ENABLE();
  __HAL_RCC_RAMCFG_CLK_ENABLE();
  RAMCFG_HandleTypeDef hramcfg = {0};
  hramcfg.Instance =  RAMCFG_SRAM3_AXI;
  HAL_RAMCFG_EnableAXISRAM(&hramcfg);
  hramcfg.Instance =  RAMCFG_SRAM4_AXI;
  HAL_RAMCFG_EnableAXISRAM(&hramcfg);
  hramcfg.Instance =  RAMCFG_SRAM5_AXI;
  HAL_RAMCFG_EnableAXISRAM(&hramcfg);
  hramcfg.Instance =  RAMCFG_SRAM6_AXI;
  HAL_RAMCFG_EnableAXISRAM(&hramcfg);
}

static void set_clk_sleep_mode(void)
{
  /*** Enable sleep mode support during NPU inference *************************/
  /* Configure peripheral clocks to remain active during sleep mode */
  /* Keep all IP's enabled during WFE so they can wake up CPU. Fine tune
   * this if you want to save maximum power
   */
#if APP_UVC
  __HAL_RCC_XSPI1_CLK_SLEEP_ENABLE();    /* For display frame buffer */
#endif
  __HAL_RCC_XSPI2_CLK_SLEEP_ENABLE();    /* For NN weights */
  __HAL_RCC_NPU_CLK_SLEEP_ENABLE();      /* For NN inference */
  __HAL_RCC_CACHEAXI_CLK_SLEEP_ENABLE(); /* For NN inference */
#if APP_UVC
  __HAL_RCC_DMA2D_CLK_SLEEP_ENABLE();    /* For display */
#endif
  __HAL_RCC_DCMIPP_CLK_SLEEP_ENABLE();   /* For camera configuration retention */
  __HAL_RCC_CSI_CLK_SLEEP_ENABLE();      /* For camera configuration retention */
  __HAL_RCC_TIM16_CLK_SLEEP_ENABLE();    /* Keep servo PWM alive during WFE/NPU */
  __HAL_RCC_GPIOA_CLK_SLEEP_ENABLE();    /* Keep PA3 alternate output active */

  __HAL_RCC_FLEXRAM_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM1_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM2_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM3_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM4_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM5_MEM_CLK_SLEEP_ENABLE();
  __HAL_RCC_AXISRAM6_MEM_CLK_SLEEP_ENABLE();
}

static void NPUCache_config(void)
{
  npu_cache_enable();
}

static void Security_Config(void)
{
  __HAL_RCC_RIFSC_CLK_ENABLE();
  RIMC_MasterConfig_t RIMC_master = {0};
  RIMC_master.MasterCID = RIF_CID_1;
  RIMC_master.SecPriv = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV;
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_NPU, &RIMC_master);
#if APP_UVC
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_DMA2D, &RIMC_master);
#endif
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_DCMIPP, &RIMC_master);
#if APP_UVC
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_LTDC1 , &RIMC_master);
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_LTDC2 , &RIMC_master);
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_OTG1 , &RIMC_master);
#endif
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_NPU , RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);
#if APP_UVC
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_DMA2D , RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);
#endif
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_CSI    , RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_DCMIPP , RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);
#if APP_UVC
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_LTDC   , RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_LTDCL1 , RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_LTDCL2 , RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_OTG1HS , RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);
#endif
  HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_SPI5 , RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV);
}

static void IAC_Config(void)
{
/* Configure IAC to trap illegal access events */
  __HAL_RCC_IAC_CLK_ENABLE();
  __HAL_RCC_IAC_FORCE_RESET();
  __HAL_RCC_IAC_RELEASE_RESET();
}

void IAC_IRQHandler(void)
{
  while (1)
  {
  }
}

#if APP_UVC
static void Display_Text90CW(uint32_t x, uint32_t y, const char *text)
{
  sFONT *font = UTIL_LCD_GetFont();
  const uint32_t fw = font->Width;
  const uint32_t fh = font->Height;
  const uint32_t bytes_per_row = (fw + 7u) / 8u;
  const uint32_t row_offset = (8u * bytes_per_row) - fw;
  const uint32_t color = UTIL_LCD_GetTextColor();

  for (uint32_t ci = 0; text[ci] != '\0'; ++ci) {
    uint8_t ch = (uint8_t)text[ci];
    if (ch < ' ' || ch > '~') ch = '?';
    const uint8_t *glyph = font->table + ((uint32_t)(ch - ' ') * fh * bytes_per_row);

    for (uint32_t row = 0; row < fh; ++row) {
      const uint8_t *p = glyph + (bytes_per_row * row);
      uint32_t bits;
      if (bytes_per_row == 1u) {
        bits = p[0];
      } else if (bytes_per_row == 2u) {
        bits = ((uint32_t)p[0] << 8) | p[1];
      } else {
        bits = ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2];
      }

      for (uint32_t col = 0; col < fw; ++col) {
        if ((bits & (1u << (fw - col + row_offset - 1u))) == 0u) continue;
        uint32_t dx = x + (fh - 1u - row);
        uint32_t dy = y + (ci * fw) + col;
        if (dx < lcd_fg_area.XSize && dy < lcd_fg_area.YSize) {
          UTIL_LCD_SetPixel((uint16_t)dx, (uint16_t)dy, color);
        }
      }
    }
  }
}

/**
* @brief Display Neural Network output classification results as well as other performances informations
*
* @param p_postprocess pointer to postprocessing output
* @param inference_ms inference time in ms
*/
static void Display_NetworkOutput(od_pp_out_t *p_postprocess, uint32_t inference_ms)
{

  od_pp_outBuffer_t *rois = p_postprocess->pOutBuff;
  uint32_t nb_rois = p_postprocess->nb_detect;
  int ret;
  (void)inference_ms;

  __disable_irq();
  ret = SCRL_SetAddress_NoReload(lcd_fg_buffer[lcd_fg_buffer_rd_idx], SCRL_LAYER_1);
  assert(ret == HAL_OK);
  __enable_irq();

  /* Draw bounding boxes */
  UTIL_LCD_FillRect(0, 0, lcd_fg_area.XSize, lcd_fg_area.YSize, UTIL_LCD_COLOR_TRANSPARENT); /* Clear previous boxes */
  for (int32_t i = 0; i < nb_rois; i++)
  {
#if NN_ROTATE_90
    /* The NN runs on a 90deg-CCW-rotated, 3:4-CROPPED frame; the display shows
     * the raw full 4:3 (sideways) camera. Map the box back into display space:
     * undo the rotation (90deg CW) AND undo the crop, else the box is stretched.
     * crop fraction cf = (MH/MW)^2 of the display width; offset centers it.
     *   xc = off + (1 - yu)*cf,  yc = xu,  wc = hu*cf,  hc = wu              */
    const float32_t cf  = ((float32_t)STAI_NETWORK_IN_1_HEIGHT * STAI_NETWORK_IN_1_HEIGHT) /
                          ((float32_t)STAI_NETWORK_IN_1_WIDTH  * STAI_NETWORK_IN_1_WIDTH);
    const float32_t off = (1.0f - cf) * 0.5f;
    float32_t xc = off + (1.0f - rois[i].y_center) * cf;
    float32_t yc = rois[i].x_center;
    float32_t wc = rois[i].height * cf;
    float32_t hc = rois[i].width;
#else
    float32_t xc = rois[i].x_center;
    float32_t yc = rois[i].y_center;
    float32_t wc = rois[i].width;
    float32_t hc = rois[i].height;
#endif
    uint32_t x0 = (uint32_t) ((xc - wc / 2) * ((float32_t) lcd_bg_area.XSize));
    uint32_t y0 = (uint32_t) ((yc - hc / 2) * ((float32_t) lcd_bg_area.YSize));
    uint32_t width = (uint32_t) (wc * ((float32_t) lcd_bg_area.XSize));
    uint32_t height = (uint32_t) (hc * ((float32_t) lcd_bg_area.YSize));
    /* Draw boxes without going outside of the image */
    x0 = x0 < lcd_bg_area.XSize ? x0 : lcd_bg_area.XSize - 1;
    y0 = y0 < lcd_bg_area.YSize ? y0 : lcd_bg_area.YSize - 1;
    width = ((x0 + width) < lcd_bg_area.XSize) ? width : (lcd_bg_area.XSize - x0 - 1);
    height = ((y0 + height) < lcd_bg_area.YSize) ? height : (lcd_bg_area.YSize - y0 - 1);
    UTIL_LCD_DrawRect(x0, y0, width, height, UTIL_LCD_COLOR_RED);
    /* class label ("person") removed; single-class ball detector. Keep conf%.
     * Draw it rotated 90deg CW (like the "Objects" label) so it reads upright on
     * the sideways display, placed just left of the box's top corner. */
    {
      char conf_label[8];
      sFONT *cfont = UTIL_LCD_GetFont();
      snprintf(conf_label, sizeof(conf_label), "%d%%", (int)(rois[i].conf * 100.0f + 0.5f));
      uint32_t lx = (x0 >= cfont->Height) ? (x0 - cfont->Height) : 0u;
      UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_RED);
      Display_Text90CW(lx, y0, conf_label);
    }
  }

  {
    char objects_label[24];
    sFONT *font = UTIL_LCD_GetFont();
    snprintf(objects_label, sizeof(objects_label), "Objects %lu", (unsigned long)nb_rois);
    UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_WHITE);
    Display_Text90CW(lcd_fg_area.XSize - font->Height - 2u, 4u, objects_label);
  }
  UTIL_LCD_SetBackColor(0);

  Display_WelcomeScreen();

  SCB_CleanDCache_by_Addr(lcd_fg_buffer[lcd_fg_buffer_rd_idx], LCD_FG_FRAMEBUFFER_SIZE);
  __disable_irq();
  ret = SCRL_ReloadLayer(SCRL_LAYER_1);
  assert(ret == HAL_OK);
  __enable_irq();
  lcd_fg_buffer_rd_idx = 1 - lcd_fg_buffer_rd_idx;
}

static void Display_init(void)
{
  SCRL_LayerConfig layers_config[2] = {
    {
      .origin = {lcd_bg_area.X0, lcd_bg_area.Y0},
      .size = {lcd_bg_area.XSize, lcd_bg_area.YSize},
      .format = SCRL_RGB565,
      .address = lcd_bg_buffer,
    },
    {
      .origin = {lcd_fg_area.X0, lcd_fg_area.Y0},
      .size = {lcd_fg_area.XSize, lcd_fg_area.YSize},
      .format = SCRL_ARGB4444,
      .address = lcd_fg_buffer[lcd_fg_buffer_rd_idx],
    },
  };
  SCRL_ScreenConfig screen_config = {
    .size = {LCD_FG_WIDTH, LCD_FG_HEIGHT},
#ifdef SCR_LIB_USE_SPI
    .format = SCRL_RGB565,
#else
    .format = SCRL_YUV422, /* Use SCRL_RGB565 if host support this format to reduce cpu load */
#endif
    .address = screen_buffer,
    .fps = CAMERA_FPS,
  };
  int ret;

  /* Initialize the LCD to black */
#ifdef SCR_LIB_USE_SPI
  memset(screen_buffer, 0, sizeof(screen_buffer));
  SCB_CleanDCache_by_Addr(screen_buffer, sizeof(screen_buffer));
#else
  uint32_t *p_screen_buffer = (uint32_t *) screen_buffer;
  for (int i = 0; i < sizeof(screen_buffer)/4; i++)
  {
    p_screen_buffer[i] = 0x80108010;
  }
  SCB_CleanDCache_by_Addr(screen_buffer, sizeof(screen_buffer));
#endif

  ret = SCRL_Init((SCRL_LayerConfig *[2]){&layers_config[0], &layers_config[1]}, &screen_config);
  assert(ret == 0);

  UTIL_LCD_SetLayer(SCRL_LAYER_1);
  UTIL_LCD_Clear(UTIL_LCD_COLOR_TRANSPARENT);
  UTIL_LCD_SetFont(&Font12);
  UTIL_LCD_SetTextColor(UTIL_LCD_COLOR_WHITE);
}

/**
 * @brief Displays a Welcome screen
 */
static void Display_WelcomeScreen(void)
{
  static uint32_t t0 = 0;
  if (t0 == 0)
    t0 = HAL_GetTick();

  if (HAL_GetTick() - t0 < 4000)
  {
    /* Draw logo */
    UTIL_LCD_FillRGBRect((lcd_bg_area.XSize-200)/2, 54, (uint8_t *) stlogo, 200, 107);

    /* Display welcome message */
    UTIL_LCD_SetBackColor(0x40000000);
    UTIL_LCDEx_PrintfAt(0, LINE(15), CENTER_MODE, "Object Detection");
    UTIL_LCDEx_PrintfAt(0, LINE(16), CENTER_MODE, WELCOME_MSG_1);
    UTIL_LCDEx_PrintfAt(0, LINE(17), CENTER_MODE, WELCOME_MSG_2[0]);
    UTIL_LCDEx_PrintfAt(0, LINE(18), CENTER_MODE, WELCOME_MSG_2[1]);
    UTIL_LCD_SetBackColor(0);
  }
}
#endif /* APP_UVC */

/**
  * @brief  DCMIPP Clock Config for DCMIPP.
  * @param  hdcmipp  DCMIPP Handle
  *         Being __weak it can be overwritten by the application
  * @retval HAL_status
  */
HAL_StatusTypeDef MX_DCMIPP_ClockConfig(DCMIPP_HandleTypeDef *hdcmipp)
{
  RCC_PeriphCLKInitTypeDef RCC_PeriphCLKInitStruct = {0};
  HAL_StatusTypeDef ret = HAL_OK;

  RCC_PeriphCLKInitStruct.PeriphClockSelection = RCC_PERIPHCLK_DCMIPP;
  RCC_PeriphCLKInitStruct.DcmippClockSelection = RCC_DCMIPPCLKSOURCE_IC17;
  RCC_PeriphCLKInitStruct.ICSelection[RCC_IC17].ClockSelection = RCC_ICCLKSOURCE_PLL2;
  RCC_PeriphCLKInitStruct.ICSelection[RCC_IC17].ClockDivider = 3;
  ret = HAL_RCCEx_PeriphCLKConfig(&RCC_PeriphCLKInitStruct);
  if (ret)
  {
    return ret;
  }

  RCC_PeriphCLKInitStruct.PeriphClockSelection = RCC_PERIPHCLK_CSI;
  RCC_PeriphCLKInitStruct.ICSelection[RCC_IC18].ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_PeriphCLKInitStruct.ICSelection[RCC_IC18].ClockDivider = 40;
  ret = HAL_RCCEx_PeriphCLKConfig(&RCC_PeriphCLKInitStruct);
  if (ret)
  {
    return ret;
  }

  return ret;
}

static void SystemClock_Config(void)
{
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_PeriphCLKInitTypeDef RCC_PeriphCLKInitStruct = {0};

  /* Ensure VDDCORE=0.9V before increasing the system frequency */
  BSP_SMPS_Init(SMPS_VOLTAGE_OVERDRIVE);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_NONE;

  /* PLL1 = 64 x 25 / 2 = 800MHz */
  RCC_OscInitStruct.PLL1.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL1.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL1.PLLM = 2;
  RCC_OscInitStruct.PLL1.PLLN = 25;
  RCC_OscInitStruct.PLL1.PLLFractional = 0;
  RCC_OscInitStruct.PLL1.PLLP1 = 1;
  RCC_OscInitStruct.PLL1.PLLP2 = 1;

  /* PLL2 = 64 x 125 / 8 = 1000MHz */
  RCC_OscInitStruct.PLL2.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL2.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL2.PLLM = 8;
  RCC_OscInitStruct.PLL2.PLLFractional = 0;
  RCC_OscInitStruct.PLL2.PLLN = 125;
  RCC_OscInitStruct.PLL2.PLLP1 = 1;
  RCC_OscInitStruct.PLL2.PLLP2 = 1;

  /* PLL3 = (64 x 225 / 8) / (1 * 2) = 900MHz */
  RCC_OscInitStruct.PLL3.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL3.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL3.PLLM = 8;
  RCC_OscInitStruct.PLL3.PLLN = 225;
  RCC_OscInitStruct.PLL3.PLLFractional = 0;
  RCC_OscInitStruct.PLL3.PLLP1 = 1;
  RCC_OscInitStruct.PLL3.PLLP2 = 2;

  /* PLL4 = (64 x 225 / 8) / (6 * 6) = 50 MHz */
  RCC_OscInitStruct.PLL4.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL4.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL4.PLLM = 8;
  RCC_OscInitStruct.PLL4.PLLFractional = 0;
  RCC_OscInitStruct.PLL4.PLLN = 225;
  RCC_OscInitStruct.PLL4.PLLP1 = 6;
  RCC_OscInitStruct.PLL4.PLLP2 = 6;

  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    while(1);
  }

  RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_CPUCLK | RCC_CLOCKTYPE_SYSCLK |
                                 RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_PCLK1 |
                                 RCC_CLOCKTYPE_PCLK2 | RCC_CLOCKTYPE_PCLK4 |
                                 RCC_CLOCKTYPE_PCLK5);

  /* CPU CLock (sysa_ck) = ic1_ck = PLL1 output/ic1_divider = 800 MHz */
  RCC_ClkInitStruct.CPUCLKSource = RCC_CPUCLKSOURCE_IC1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_IC2_IC6_IC11;
  RCC_ClkInitStruct.IC1Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC1Selection.ClockDivider = 1;

  /* AXI Clock (sysb_ck) = ic2_ck = PLL1 output/ic2_divider = 400 MHz */
  RCC_ClkInitStruct.IC2Selection.ClockSelection = RCC_ICCLKSOURCE_PLL1;
  RCC_ClkInitStruct.IC2Selection.ClockDivider = 2;

  /* NPU Clock (sysc_ck) = ic6_ck = PLL2 output/ic6_divider = 1000 MHz */
  RCC_ClkInitStruct.IC6Selection.ClockSelection = RCC_ICCLKSOURCE_PLL2;
  RCC_ClkInitStruct.IC6Selection.ClockDivider = 1;

  /* AXISRAM3/4/5/6 Clock (sysd_ck) = ic11_ck = PLL3 output/ic11_divider = 900 MHz */
  RCC_ClkInitStruct.IC11Selection.ClockSelection = RCC_ICCLKSOURCE_PLL3;
  RCC_ClkInitStruct.IC11Selection.ClockDivider = 1;

  /* HCLK = sysb_ck / HCLK divider = 200 MHz */
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;

  /* PCLKx = HCLK / PCLKx divider = 200 MHz */
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV1;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;
  RCC_ClkInitStruct.APB5CLKDivider = RCC_APB5_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct) != HAL_OK)
  {
    while(1);
  }

  RCC_PeriphCLKInitStruct.PeriphClockSelection = 0;

  /* XSPI1 kernel clock (ck_ker_xspi1) = HCLK = 200MHz */
  RCC_PeriphCLKInitStruct.PeriphClockSelection |= RCC_PERIPHCLK_XSPI1;
  RCC_PeriphCLKInitStruct.Xspi1ClockSelection = RCC_XSPI1CLKSOURCE_HCLK;

  /* XSPI2 kernel clock (ck_ker_xspi1) = HCLK =  200MHz */
  RCC_PeriphCLKInitStruct.PeriphClockSelection |= RCC_PERIPHCLK_XSPI2;
  RCC_PeriphCLKInitStruct.Xspi2ClockSelection = RCC_XSPI2CLKSOURCE_HCLK;

  if (HAL_RCCEx_PeriphCLKConfig(&RCC_PeriphCLKInitStruct) != HAL_OK)
  {
    while (1);
  }
}

/* HC-06 Bluetooth module: transparent serial bridge (same role as Serial in
 * paul_car.ino). It sits on the Arduino D0/D1 header pins = USART3 (PD9 RX,
 * PD8 TX). HC-06 default baud is 9600. Received bytes are drive commands. */
static void BT_Config(void)
{
  GPIO_InitTypeDef gpio_init = {0};

  __HAL_RCC_USART3_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  gpio_init.Mode      = GPIO_MODE_AF_PP;
  gpio_init.Pull      = GPIO_PULLUP;
  gpio_init.Speed     = GPIO_SPEED_FREQ_HIGH;
  gpio_init.Pin       = GPIO_PIN_8 | GPIO_PIN_9;   /* PD8 USART3_TX, PD9 USART3_RX */
  gpio_init.Alternate = GPIO_AF7_USART3;
  HAL_GPIO_Init(GPIOD, &gpio_init);

  huart3.Instance          = USART3;
  huart3.Init.BaudRate     = 9600;                 /* HC-06 default */
  huart3.Init.Mode         = UART_MODE_TX_RX;
  huart3.Init.Parity       = UART_PARITY_NONE;
  huart3.Init.WordLength   = UART_WORDLENGTH_8B;
  huart3.Init.StopBits     = UART_STOPBITS_1;
  huart3.Init.HwFlowCtl    = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    while (1);
  }

  /* Enable the 8-byte RX FIFO. The CPU is busy in NPU inference ~70ms of every
   * ~80ms loop and can't service this ISR meanwhile; with only the 1-byte RDR a
   * command burst overruns and all but the last byte is lost (the "Arduino was
   * better" symptom -- the Arduino is never blocked). The FIFO buffers a whole
   * short burst in hardware through that window; the ISR drains it all at once
   * once the CPU is free (see the while-loop in USART3_IRQHandler). */
  HAL_UARTEx_SetRxFifoThreshold(&huart3, UART_RXFIFO_THRESHOLD_1_8);
  HAL_UARTEx_EnableFifoMode(&huart3);

  /* Interrupt-driven RX (see USART3_IRQHandler / ring buffer). */
  HAL_NVIC_SetPriority(USART3_IRQn, 6, 0);
  HAL_NVIC_EnableIRQ(USART3_IRQn);
  __HAL_UART_ENABLE_IT(&huart3, UART_IT_RXNE);
}

static void CONSOLE_Config()
{
  GPIO_InitTypeDef gpio_init;

  __HAL_RCC_USART1_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

 /* DISCO & NUCLEO USART1 (PE5/PE6) */
  gpio_init.Mode      = GPIO_MODE_AF_PP;
  gpio_init.Pull      = GPIO_PULLUP;
  gpio_init.Speed     = GPIO_SPEED_FREQ_HIGH;
  gpio_init.Pin       = GPIO_PIN_5 | GPIO_PIN_6;
  gpio_init.Alternate = GPIO_AF7_USART1;
  HAL_GPIO_Init(GPIOE, &gpio_init);

  huart1.Instance          = USART1;
  huart1.Init.BaudRate     = 4000000;   /* fast benchmark image upload (exact 200MHz/50; was 921600) */
  huart1.Init.Mode         = UART_MODE_TX_RX;
  huart1.Init.Parity       = UART_PARITY_NONE;
  huart1.Init.WordLength   = UART_WORDLENGTH_8B;
  huart1.Init.StopBits     = UART_STOPBITS_1;
  huart1.Init.HwFlowCtl    = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_8;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    while (1);
  }
}

int _write(int file, char *ptr, int len)
{
  HAL_StatusTypeDef status;

  if ((file != STDOUT_FILENO) && (file != STDERR_FILENO)) {
      errno = EBADF;
      return -1;
  }

  status = HAL_UART_Transmit(&huart1, (uint8_t*)ptr, len, ~0);

  return (status == HAL_OK ? len : 0);
}


void npu_cache_enable_clocks_and_reset(void)
{
  __HAL_RCC_CACHEAXIRAM_MEM_CLK_ENABLE();
  __HAL_RCC_CACHEAXI_CLK_ENABLE();
  __HAL_RCC_CACHEAXI_FORCE_RESET();
  __HAL_RCC_CACHEAXI_RELEASE_RESET();
}

void npu_cache_disable_clocks_and_reset(void)
{
  __HAL_RCC_CACHEAXIRAM_MEM_CLK_DISABLE();
  __HAL_RCC_CACHEAXI_CLK_DISABLE();
  __HAL_RCC_CACHEAXI_FORCE_RESET();
}

#ifdef  USE_FULL_ASSERT

/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t* file, uint32_t line)
{
  UNUSED(file);
  UNUSED(line);
  __BKPT(0);
  while (1)
  {
  }
}

#endif
