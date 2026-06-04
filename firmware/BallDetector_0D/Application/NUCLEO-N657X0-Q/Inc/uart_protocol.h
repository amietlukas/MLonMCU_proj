/* UART framing for the BallDetector_0D benchmark mode <-> host.
 *
 * Wire format (multi-byte fields little-endian); source of truth is
 * firmware/Host/protocol_balldetector_n6.md:
 *   AA 55 A5 5A  <type:u8>  <len:u32>  <payload[len]>  <crc16:u16>
 * crc16 = CRC-16/CCITT-FALSE over <type><len><payload>.
 *
 * Ported from firmware/BallDetector_N6; INFER_DEC extended to report the
 * per-stage timings (pre / infer / post) for the model comparison table.
 */
#ifndef UART_PROTOCOL_H
#define UART_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>
#include "yolo_postproc.h"   /* yolo_box_t */

#define PROTO_MAGIC_0  0xAA
#define PROTO_MAGIC_1  0x55
#define PROTO_MAGIC_2  0xA5
#define PROTO_MAGIC_3  0x5A

typedef enum {
    PROTO_T_HELLO     = 0x01,  /* H->F */
    PROTO_T_INFO      = 0x02,  /* F->H : fw:u16, w:u16, h:u16, c:u8, n_out:u8 */
    PROTO_T_IMG_BEGIN = 0x10,  /* H->F : w:u16, h:u16, c:u8, fmt:u8 */
    PROTO_T_IMG_CHUNK = 0x11,  /* H->F : idx:u16, bytes[] */
    PROTO_T_IMG_END   = 0x12,  /* H->F */
    PROTO_T_INFER_DEC = 0x21,  /* F->H : pre_us:u32, infer_us:u32, post_us:u32,
                                *        n:u16, n*{x1,y1,x2,y2,score} f32      */
    PROTO_T_LOG       = 0xFE,  /* F->H : UTF-8 */
    PROTO_T_ACK       = 0xFF,
} proto_type_t;

uint16_t proto_crc16_ccitt(const uint8_t *data, size_t len, uint16_t crc);

void proto_init(void *huart);            /* bind to a UART_HandleTypeDef* */
void proto_send_frame(uint8_t type, const uint8_t *payload, uint32_t len);
void proto_send_info(uint16_t fw_ver, uint16_t w, uint16_t h, uint8_t c, uint8_t n_out);
void proto_send_log(const char *msg);
void proto_send_infer_dec(uint32_t pre_us, uint32_t infer_us, uint32_t post_us,
                          const yolo_box_t *boxes, uint16_t n);

/* Returns 1 on a good frame, 0 on timeout, <0 on CRC/overflow. */
int proto_recv_frame(uint8_t *type, uint8_t *payload, uint32_t cap, uint32_t *len);

#endif /* UART_PROTOCOL_H */
