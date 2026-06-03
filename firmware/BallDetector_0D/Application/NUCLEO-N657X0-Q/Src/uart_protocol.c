/* UART framing for BallDetector_0D benchmark mode. See uart_protocol.h and
 * firmware/Host/protocol_balldetector_n6.md. Blocking HAL transfers on the
 * ST-LINK VCP USART (USART1 @ 921600). */

#include "uart_protocol.h"
#include "main.h"          /* HAL + UART_HandleTypeDef */
#include <string.h>

#define UART_TX_TIMEOUT_MS  1000u
#define UART_RX_TIMEOUT_MS  2000u
#define PROTO_MAX_BOXES     YOLO_MAX_DET

static const uint8_t PROTO_MAGIC[4] = {
    PROTO_MAGIC_0, PROTO_MAGIC_1, PROTO_MAGIC_2, PROTO_MAGIC_3
};
static UART_HandleTypeDef *s_huart;

uint16_t proto_crc16_ccitt(const uint8_t *data, size_t len, uint16_t crc)
{
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; ++b) {
            crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x1021u)
                                  : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

void proto_init(void *huart) { s_huart = (UART_HandleTypeDef *)huart; }

static void tx(const uint8_t *p, uint16_t n)
{
    HAL_UART_Transmit(s_huart, (uint8_t *)p, n, UART_TX_TIMEOUT_MS);
}

void proto_send_frame(uint8_t type, const uint8_t *payload, uint32_t len)
{
    uint8_t hdr[5] = { type, (uint8_t)len, (uint8_t)(len >> 8),
                       (uint8_t)(len >> 16), (uint8_t)(len >> 24) };
    uint16_t crc = proto_crc16_ccitt(hdr, sizeof(hdr), 0xFFFFu);
    if (len) crc = proto_crc16_ccitt(payload, len, crc);
    uint8_t crc_le[2] = { (uint8_t)crc, (uint8_t)(crc >> 8) };

    tx(PROTO_MAGIC, sizeof(PROTO_MAGIC));
    tx(hdr, sizeof(hdr));
    if (len) tx(payload, (uint16_t)len);
    tx(crc_le, sizeof(crc_le));
}

void proto_send_info(uint16_t fw_ver, uint16_t w, uint16_t h, uint8_t c, uint8_t n_out)
{
    uint8_t p[8] = { (uint8_t)fw_ver, (uint8_t)(fw_ver >> 8),
                     (uint8_t)w, (uint8_t)(w >> 8),
                     (uint8_t)h, (uint8_t)(h >> 8), c, n_out };
    proto_send_frame(PROTO_T_INFO, p, sizeof(p));
}

void proto_send_log(const char *msg)
{
    proto_send_frame(PROTO_T_LOG, (const uint8_t *)msg, (uint32_t)strlen(msg));
}

void proto_send_infer_dec(uint32_t pre_us, uint32_t infer_us, uint32_t post_us,
                          const yolo_box_t *boxes, uint16_t n)
{
    uint8_t p[14 + PROTO_MAX_BOXES * 20];
    uint32_t off = 0;
    memcpy(&p[off], &pre_us,   4); off += 4;
    memcpy(&p[off], &infer_us, 4); off += 4;
    memcpy(&p[off], &post_us,  4); off += 4;
    memcpy(&p[off], &n,        2); off += 2;
    for (uint16_t i = 0; i < n; ++i) {
        memcpy(&p[off], &boxes[i].x1, 4);    off += 4;
        memcpy(&p[off], &boxes[i].y1, 4);    off += 4;
        memcpy(&p[off], &boxes[i].x2, 4);    off += 4;
        memcpy(&p[off], &boxes[i].y2, 4);    off += 4;
        memcpy(&p[off], &boxes[i].score, 4); off += 4;
    }
    proto_send_frame(PROTO_T_INFER_DEC, p, off);
}

static int rx_exact(uint8_t *buf, uint32_t n, uint32_t timeout_ms)
{
    if (HAL_UART_Receive(s_huart, buf, (uint16_t)n, timeout_ms) == HAL_OK) {
        return 1;
    }
    /* Clear overrun/framing/noise so one glitch on the unpaced image upload
     * doesn't permanently wedge RX (the frame is abandoned + resynced via the
     * magic hunt; the host's chunk-delay prevents most of these). */
    __HAL_UART_CLEAR_OREFLAG(s_huart);
    __HAL_UART_CLEAR_FEFLAG(s_huart);
    __HAL_UART_CLEAR_NEFLAG(s_huart);
    return 0;
}

int proto_recv_frame(uint8_t *type, uint8_t *payload, uint32_t cap, uint32_t *len)
{
    uint8_t win[4] = {0}, b;
    int matched = 0;
    for (int tries = 0; tries < 65536 && !matched; ++tries) {
        if (!rx_exact(&b, 1, UART_RX_TIMEOUT_MS)) return 0;
        win[0] = win[1]; win[1] = win[2]; win[2] = win[3]; win[3] = b;
        matched = (win[0] == PROTO_MAGIC_0 && win[1] == PROTO_MAGIC_1 &&
                   win[2] == PROTO_MAGIC_2 && win[3] == PROTO_MAGIC_3);
    }
    if (!matched) return 0;

    uint8_t hdr[5];
    if (!rx_exact(hdr, sizeof(hdr), UART_RX_TIMEOUT_MS)) return 0;
    uint32_t plen = (uint32_t)hdr[1] | ((uint32_t)hdr[2] << 8) |
                    ((uint32_t)hdr[3] << 16) | ((uint32_t)hdr[4] << 24);
    if (plen > cap) return -1;
    if (plen && !rx_exact(payload, plen, UART_RX_TIMEOUT_MS)) return 0;

    uint8_t crc_bytes[2];
    if (!rx_exact(crc_bytes, sizeof(crc_bytes), UART_RX_TIMEOUT_MS)) return 0;
    uint16_t crc_recv = (uint16_t)crc_bytes[0] | ((uint16_t)crc_bytes[1] << 8);
    uint16_t crc_calc = proto_crc16_ccitt(hdr, sizeof(hdr), 0xFFFFu);
    if (plen) crc_calc = proto_crc16_ccitt(payload, plen, crc_calc);
    if (crc_recv != crc_calc) return -1;

    *type = hdr[0];
    *len  = plen;
    return 1;
}
