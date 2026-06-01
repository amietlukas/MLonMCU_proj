/* UART framing for BallDetector_N6 <-> host. See uart_protocol.h and
 * firmware/Host/protocol_balldetector_n6.md for the wire format.
 *
 * Blocking HAL transfers over the ST-LINK VCP USART. Fine for IMG mode and
 * the low box-count INFER_DEC frames; CAM-mode JPEG streaming will want DMA. */

#include "uart_protocol.h"
#include "constants.h"
#include "main.h"          /* HAL + UART_HandleTypeDef */

#include <string.h>

static const uint8_t PROTO_MAGIC[4] = {
    PROTO_MAGIC_0, PROTO_MAGIC_1, PROTO_MAGIC_2, PROTO_MAGIC_3
};

/* Bound UART instance (set by proto_init). */
static UART_HandleTypeDef *s_huart;

uint16_t proto_crc16_ccitt(const uint8_t *data, size_t len, uint16_t crc)
{
    /* CRC-16/CCITT-FALSE, identical to the host's crc16_ccitt(). */
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; ++b) {
            crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x1021u)
                                  : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

void proto_init(void *huart)
{
    s_huart = (UART_HandleTypeDef *)huart;
}

/* One blocking write; small helper so the frame pieces share the timeout. */
static void tx(const uint8_t *p, uint16_t n)
{
    HAL_UART_Transmit(s_huart, (uint8_t *)p, n, UART_TX_TIMEOUT_MS);
}

void proto_send_frame(uint8_t type, const uint8_t *payload, uint32_t len)
{
    uint8_t hdr[5];
    hdr[0] = type;
    hdr[1] = (uint8_t)(len);
    hdr[2] = (uint8_t)(len >> 8);
    hdr[3] = (uint8_t)(len >> 16);
    hdr[4] = (uint8_t)(len >> 24);

    /* crc16 covers <type><len><payload>, matching the host. */
    uint16_t crc = proto_crc16_ccitt(hdr, sizeof(hdr), 0xFFFFu);
    if (len) {
        crc = proto_crc16_ccitt(payload, len, crc);
    }
    uint8_t crc_le[2] = { (uint8_t)crc, (uint8_t)(crc >> 8) };

    tx(PROTO_MAGIC, sizeof(PROTO_MAGIC));
    tx(hdr, sizeof(hdr));
    if (len) {
        tx(payload, (uint16_t)len);
    }
    tx(crc_le, sizeof(crc_le));
}

void proto_send_info(void)
{
    /* fw_ver:u16, model_w:u16, model_h:u16, model_c:u8, n_outputs:u8
     * (host unpacks "<HHHBB"). */
    uint8_t p[8];
    p[0] = (uint8_t)(FW_VERSION);
    p[1] = (uint8_t)(FW_VERSION >> 8);
    p[2] = (uint8_t)(MODEL_W);
    p[3] = (uint8_t)(MODEL_W >> 8);
    p[4] = (uint8_t)(MODEL_H);
    p[5] = (uint8_t)(MODEL_H >> 8);
    p[6] = (uint8_t)MODEL_C;
    p[7] = (uint8_t)YOLO_NUM_HEADS;
    proto_send_frame(PROTO_T_INFO, p, sizeof(p));
}

void proto_send_log(const char *msg)
{
    proto_send_frame(PROTO_T_LOG, (const uint8_t *)msg, (uint32_t)strlen(msg));
}

void proto_send_infer_dec(uint32_t inference_us, const yolo_box_t *boxes, uint16_t n)
{
    /* inference_us:u32, n_boxes:u16, then n * {x1,y1,x2,y2,score} f32.
     * Built in a stack buffer sized for the box cap; n is already <= cap. */
    uint8_t p[6 + DET_MAX_BOXES * 20];
    uint32_t off = 0;

    memcpy(&p[off], &inference_us, 4); off += 4;
    memcpy(&p[off], &n, 2);            off += 2;
    for (uint16_t i = 0; i < n; ++i) {
        memcpy(&p[off], &boxes[i].x1, 4);    off += 4;
        memcpy(&p[off], &boxes[i].y1, 4);    off += 4;
        memcpy(&p[off], &boxes[i].x2, 4);    off += 4;
        memcpy(&p[off], &boxes[i].y2, 4);    off += 4;
        memcpy(&p[off], &boxes[i].score, 4); off += 4;
    }
    proto_send_frame(PROTO_T_INFER_DEC, p, off);
}

/* Blocking read of exactly n bytes; returns 1 on success, 0 on timeout. */
static int rx_exact(uint8_t *buf, uint32_t n, uint32_t timeout_ms)
{
    return HAL_UART_Receive(s_huart, buf, (uint16_t)n, timeout_ms) == HAL_OK;
}

int proto_recv_frame(uint8_t *type, uint8_t *payload, uint32_t cap, uint32_t *len)
{
    /* Hunt for the 4-byte magic with a sliding window. */
    uint8_t win[4] = {0};
    uint8_t b;
    int matched = 0;
    for (int tries = 0; tries < 4096 && !matched; ++tries) {
        if (!rx_exact(&b, 1, UART_RX_TIMEOUT_MS)) {
            return 0;  /* timeout */
        }
        win[0] = win[1]; win[1] = win[2]; win[2] = win[3]; win[3] = b;
        matched = (win[0] == PROTO_MAGIC_0 && win[1] == PROTO_MAGIC_1 &&
                   win[2] == PROTO_MAGIC_2 && win[3] == PROTO_MAGIC_3);
    }
    if (!matched) {
        return 0;
    }

    uint8_t hdr[5];
    if (!rx_exact(hdr, sizeof(hdr), UART_RX_TIMEOUT_MS)) {
        return 0;
    }
    uint32_t plen = (uint32_t)hdr[1] | ((uint32_t)hdr[2] << 8) |
                    ((uint32_t)hdr[3] << 16) | ((uint32_t)hdr[4] << 24);
    if (plen > cap) {
        return -1;  /* payload would overflow the caller buffer */
    }
    if (plen && !rx_exact(payload, plen, UART_RX_TIMEOUT_MS)) {
        return 0;
    }

    uint8_t crc_bytes[2];
    if (!rx_exact(crc_bytes, sizeof(crc_bytes), UART_RX_TIMEOUT_MS)) {
        return 0;
    }
    uint16_t crc_recv = (uint16_t)crc_bytes[0] | ((uint16_t)crc_bytes[1] << 8);
    uint16_t crc_calc = proto_crc16_ccitt(hdr, sizeof(hdr), 0xFFFFu);
    if (plen) {
        crc_calc = proto_crc16_ccitt(payload, plen, crc_calc);
    }
    if (crc_recv != crc_calc) {
        return -1;  /* CRC mismatch */
    }

    *type = hdr[0];
    *len = plen;
    return 1;
}
