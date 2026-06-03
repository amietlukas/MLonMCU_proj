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

/* Per-head config (count/scale/zp/stride/grid) is now passed in at call time,
 * sourced from the generated STAI_NETWORK_OUT_* macros -- see head_cfg_init()
 * in main.c. Fully model-agnostic: 2- or 3-head, any stride set. */

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
    const uint8_t * const *heads,
    int          n_heads,
    const float *scales,
    const int   *zps,
    const int   *strides,
    const int   *grid_w,
    const int   *grid_h,
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

    for (int h = 0; h < n_heads; ++h) {
        const uint8_t *buf = heads[h];
        if (!buf) continue;

        const int H = grid_h[h];
        const int W = grid_w[h];
        const int HW = H * W;
        const float scale = scales[h];          /* per-head, from generated macros */
        const int   zp    = zps[h];
        const float stride = (float)strides[h];

        /* The NPU outputs are SIGNED int8 (QLinear int8). They MUST be read as
         * int8_t: reading as uint8 turns a negative logit (e.g. -2 = 0xFE) into
         * +254, which dequantizes to a huge value and saturates sigmoid to 1.0.
         * That was the "8 boxes @ score=1.000 on every frame" bug — worst on the
         * zero_point=0 heads (p16/p32), where most objectness logits are negative. */
        /* Objectness threshold (signed): q >= q_thr iff tobj_float >= logit_thr,
         * tobj_float = (q - zp) * scale  =>  q = zp + logit_thr / scale. */
        const float q_thr_f = (float)zp + logit_thr / scale;
        int q_thr;
        if      (q_thr_f <= -128.0f)  q_thr = -128;
        else if (q_thr_f >=  127.0f)  q_thr =  128;   /* never matches */
        else                          q_thr = (int)ceilf(q_thr_f);

        const int8_t *p_tobj = (const int8_t *)buf + 4 * HW;
        const int8_t *p_tx   = (const int8_t *)buf + 0 * HW;
        const int8_t *p_ty   = (const int8_t *)buf + 1 * HW;
        const int8_t *p_tw   = (const int8_t *)buf + 2 * HW;
        const int8_t *p_th   = (const int8_t *)buf + 3 * HW;

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int idx = y * W + x;
                const int q = p_tobj[idx];          /* signed int8 objectness */
                if (q < q_thr) continue;

                const float tx = ((float)p_tx[idx] - zp) * scale;
                const float ty = ((float)p_ty[idx] - zp) * scale;
                const float tw = ((float)p_tw[idx] - zp) * scale;
                const float th = ((float)p_th[idx] - zp) * scale;
                const float to = ((float)q        - zp) * scale;

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
