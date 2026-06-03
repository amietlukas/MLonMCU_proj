/* Single-class STYOLO-nano post-processing for ball_n6.
 *
 * Decodes the three uint8 NPU output heads (p8/p16/p32, channel-first
 * [5,H,W] with channels = tx,ty,tw,th,tobj) into xyxy float boxes in
 * resized-image space (640x480), with NMS.
 *
 * Mirrors software/ball_detector_n6/yolo_decode.py exactly.
 *
 * Self-contained — no firmware or X-CUBE-AI dependencies. Pass the raw
 * uint8 output buffers from `ai_run` / `stai_run`.
 */
#ifndef YOLO_POSTPROC_H
#define YOLO_POSTPROC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YOLO_INPUT_W   384
#define YOLO_INPUT_H   288
#define YOLO_NUM_HEADS   3
#define YOLO_MAX_DET     8  /* caller can cap below this */

typedef struct {
    float x1, y1, x2, y2;
    float score;
} yolo_box_t;

/* Run decode + NMS.
 *
 *   heads[0] = p8  buffer, 5*60*80 = 24000 bytes, channel-first
 *   heads[1] = p16 buffer, 5*30*40 =  6000 bytes
 *   heads[2] = p32 buffer, 5*15*20 =  1500 bytes
 *
 *   conf_thresh : objectness threshold (post-sigmoid), typ. 0.50
 *   iou_thresh  : NMS IoU threshold, typ. 0.25
 *   out         : caller-provided buffer of yolo_box_t
 *   out_cap     : capacity of `out` (<= YOLO_MAX_DET in practice)
 *
 * Returns number of boxes written.
 */
/* N-head, fully model-agnostic: pass the actual head count and per-head
 * scale / zero-point / stride / grid (all from the generated STAI_NETWORK_OUT_*
 * macros). Works for 2- or 3-head models, any stride set. heads[] has n_heads
 * pointers (heads[h] = nn_out[h]). */
int yolo_postprocess(
    const uint8_t * const *heads,
    int          n_heads,
    const float *scales,    /* [n_heads] output scale     */
    const int   *zps,       /* [n_heads] zero-point        */
    const int   *strides,   /* [n_heads] = input_w/grid_w  */
    const int   *grid_w,    /* [n_heads]                   */
    const int   *grid_h,    /* [n_heads]                   */
    float conf_thresh,
    float iou_thresh,
    yolo_box_t *out,
    int out_cap
);

#ifdef __cplusplus
}
#endif

#endif /* YOLO_POSTPROC_H */
