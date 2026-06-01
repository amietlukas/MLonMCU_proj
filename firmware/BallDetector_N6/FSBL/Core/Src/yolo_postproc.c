/* Implementation — see yolo_postproc.h.
 *
 * Hot path:
 *   1. For each head and cell, reject early using a *quantized* threshold
 *      on tobj (no math, no dequant). Survivors are typically a few hundred
 *      out of 6300 cells.
 *   2. For survivors, dequant + sigmoid/exp decode → xyxy.
 *   3. Bubble-sort top-K by score (K small, ~32), then greedy NMS.
 *
 * Quant params come from stedgeai's analyze report and are fixed for this
 * model. If the model is re-trained or re-quantized, re-run
 * `stedgeai analyze` and update these constants.
 */
#include "yolo_postproc.h"

#include <math.h>
#include <string.h>

/* --- per-head config (from ball_n6 analyze report) ------------------------ */
typedef struct {
    float    scale;
    uint8_t  zero_point;
    int16_t  stride;
    int16_t  grid_h;
    int16_t  grid_w;
} yolo_head_cfg_t;

static const yolo_head_cfg_t HEADS[YOLO_NUM_HEADS] = {
    { 0.256579876f,  41,  8, 36, 48 },  /* p8  */
    { 0.204285681f,   0, 16, 18, 24 },  /* p16 */
    { 0.146335885f,   0, 32,  9, 12 },  /* p32 */
};

/* Match the training-time decode clamp range. */
#define TWTH_CLAMP_MIN  (-4.0f)
#define TWTH_CLAMP_MAX  ( 4.0f)

/* Pre-NMS candidate cap. 6300 total cells; in practice <100 survive a 0.5
 * objectness threshold even on busy scenes. 128 leaves headroom. */
#define PRENMS_CAP  128

/* --- helpers -------------------------------------------------------------- */

static inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* logit(p) = ln(p/(1-p)) — inverse of sigmoid. Used to convert the
 * post-sigmoid confidence threshold into a pre-sigmoid (and then
 * quantized) threshold so the inner loop can reject cells with a single
 * uint8 compare. */
static inline float logitf(float p) {
    return logf(p / (1.0f - p));
}

/* --- main ----------------------------------------------------------------- */

int yolo_postprocess(
    const uint8_t * const heads[YOLO_NUM_HEADS],
    float conf_thresh,
    float iou_thresh,
    yolo_box_t *out,
    int out_cap)
{
    if (out_cap <= 0 || !out) return 0;
    conf_thresh = clampf(conf_thresh, 1e-4f, 0.9999f);

    yolo_box_t cand[PRENMS_CAP];
    int n_cand = 0;

    const float logit_thr = logitf(conf_thresh);

    for (int h = 0; h < YOLO_NUM_HEADS; ++h) {
        const yolo_head_cfg_t *cfg = &HEADS[h];
        const uint8_t *buf = heads[h];
        if (!buf) continue;

        const int H = cfg->grid_h;
        const int W = cfg->grid_w;
        const int HW = H * W;
        const float scale = cfg->scale;
        const int   zp    = cfg->zero_point;
        const float stride = (float)cfg->stride;

        /* Quantized objectness threshold: u8 >= u8_thr iff tobj_float >= logit_thr.
         * tobj_float = (u8 - zp) * scale  =>  u8 = zp + logit_thr / scale  */
        const float u8_thr_f = (float)zp + logit_thr / scale;
        int u8_thr;
        if      (u8_thr_f <= 0.0f)    u8_thr = 0;
        else if (u8_thr_f >= 255.0f)  u8_thr = 256;   /* never matches */
        else                          u8_thr = (int)ceilf(u8_thr_f);

        const uint8_t *p_tobj = buf + 4 * HW;
        const uint8_t *p_tx   = buf + 0 * HW;
        const uint8_t *p_ty   = buf + 1 * HW;
        const uint8_t *p_tw   = buf + 2 * HW;
        const uint8_t *p_th   = buf + 3 * HW;

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int idx = y * W + x;
                const int u8 = p_tobj[idx];
                if (u8 < u8_thr) continue;

                const float tx = ((float)p_tx[idx] - zp) * scale;
                const float ty = ((float)p_ty[idx] - zp) * scale;
                const float tw = ((float)p_tw[idx] - zp) * scale;
                const float th = ((float)p_th[idx] - zp) * scale;
                const float to = ((float)u8       - zp) * scale;

                const float cx = (sigmoidf(tx) + (float)x) * stride;
                const float cy = (sigmoidf(ty) + (float)y) * stride;
                const float pw = expf(clampf(tw, TWTH_CLAMP_MIN, TWTH_CLAMP_MAX)) * stride;
                const float ph = expf(clampf(th, TWTH_CLAMP_MIN, TWTH_CLAMP_MAX)) * stride;

                yolo_box_t b;
                b.x1 = clampf(cx - 0.5f * pw, 0.0f, (float)YOLO_INPUT_W);
                b.y1 = clampf(cy - 0.5f * ph, 0.0f, (float)YOLO_INPUT_H);
                b.x2 = clampf(cx + 0.5f * pw, 0.0f, (float)YOLO_INPUT_W);
                b.y2 = clampf(cy + 0.5f * ph, 0.0f, (float)YOLO_INPUT_H);
                b.score = sigmoidf(to);

                if (n_cand < PRENMS_CAP) {
                    cand[n_cand++] = b;
                } else {
                    /* Buffer full — evict the lowest-scoring entry if this one is better.
                     * Linear scan is fine; PRENMS_CAP is small. */
                    int worst = 0;
                    for (int k = 1; k < PRENMS_CAP; ++k) {
                        if (cand[k].score < cand[worst].score) worst = k;
                    }
                    if (b.score > cand[worst].score) cand[worst] = b;
                }
            }
        }
    }

    if (n_cand == 0) return 0;

    /* Sort candidates by score desc (insertion sort, small n). */
    for (int i = 1; i < n_cand; ++i) {
        yolo_box_t key = cand[i];
        int j = i - 1;
        while (j >= 0 && cand[j].score < key.score) {
            cand[j + 1] = cand[j];
            --j;
        }
        cand[j + 1] = key;
    }

    /* Greedy NMS. */
    uint8_t suppressed[PRENMS_CAP] = {0};
    int n_out = 0;
    for (int i = 0; i < n_cand && n_out < out_cap; ++i) {
        if (suppressed[i]) continue;
        const yolo_box_t *bi = &cand[i];
        out[n_out++] = *bi;
        const float ai = (bi->x2 - bi->x1) * (bi->y2 - bi->y1);
        for (int j = i + 1; j < n_cand; ++j) {
            if (suppressed[j]) continue;
            const yolo_box_t *bj = &cand[j];
            const float xx1 = bi->x1 > bj->x1 ? bi->x1 : bj->x1;
            const float yy1 = bi->y1 > bj->y1 ? bi->y1 : bj->y1;
            const float xx2 = bi->x2 < bj->x2 ? bi->x2 : bj->x2;
            const float yy2 = bi->y2 < bj->y2 ? bi->y2 : bj->y2;
            const float iw = xx2 - xx1; if (iw <= 0.0f) continue;
            const float ih = yy2 - yy1; if (ih <= 0.0f) continue;
            const float inter = iw * ih;
            const float aj = (bj->x2 - bj->x1) * (bj->y2 - bj->y1);
            const float uni = ai + aj - inter;
            if (uni > 0.0f && (inter / uni) >= iou_thresh) suppressed[j] = 1;
        }
    }
    return n_out;
}
