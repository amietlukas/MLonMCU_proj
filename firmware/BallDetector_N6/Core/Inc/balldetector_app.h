#ifndef BALLDETECTOR_APP_H
#define BALLDETECTOR_APP_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bring up the network runtime + UART framer, send the INFO frame.
 * Call from MX_X_CUBE_AI_Init() (USER CODE block). */
void balldetector_init(void);

/* Main loop — never returns. Call from MX_X_CUBE_AI_Process().
 *   IMG mode : assemble a host-streamed RGB frame -> NPU -> decode -> INFER_DEC
 *   CAM mode : DCMIPP frame -> NPU -> decode -> CAM_FRAME + INFER_DEC
 * Motor/servo control loop plugs in here later (PWM pinmux deferred). */
void balldetector_run(void);

/* Set by the DCMIPP frame-event ISR, cleared by the capture loop.
 * Declared here so a HAL_DCMIPP_PIPE_FrameEventCallback override (in main.c
 * or a BSP wrapper) can flip it. */
extern volatile uint8_t g_frame_ready;

#ifdef __cplusplus
}
#endif

#endif /* BALLDETECTOR_APP_H */
