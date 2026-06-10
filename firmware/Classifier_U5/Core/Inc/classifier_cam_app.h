#ifndef CLASSIFIER_CAM_APP_H
#define CLASSIFIER_CAM_APP_H

#ifdef __cplusplus
extern "C" {
#endif

void classifier_cam_init(void);
void classifier_cam_process(void);  /* never returns */

/* Set by the BSP frame-event ISR; cleared by the capture loop.
 * Declared here so a BSP_CAMERA_FrameEventCallback override (placed
 * in main.c or a BSP wrapper) can flip the flag. */
extern volatile uint8_t g_frame_ready;

#ifdef __cplusplus
}
#endif

#endif
