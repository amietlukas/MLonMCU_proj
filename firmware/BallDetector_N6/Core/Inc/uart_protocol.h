/* UART framing for BallDetector_N6 <-> host.
 *
 * Source of truth for the wire format is
 * firmware/Host/protocol_balldetector_n6.md — keep this header and the
 * host's protocol constants in lockstep with that doc.
 *
 * Frame layout (multi-byte fields little-endian):
 *   AA 55 A5 5A  <type:u8>  <len:u32>  <payload[len]>  <crc16:u16>
 * crc16 is CRC-16/CCITT-FALSE over <type><len><payload>.
 *
 * The transmit/receive primitives bind to the project's UART instance via
 * proto_init(); the handle is passed opaquely so this header stays free of
 * the HAL headers (those only exist once CubeMX has generated the project).
 */
#ifndef UART_PROTOCOL_H
#define UART_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#include "yolo_postproc.h"   /* yolo_box_t */

#ifdef __cplusplus
extern "C" {
#endif

/* 4-byte frame magic. */
#define PROTO_MAGIC_0  0xAA
#define PROTO_MAGIC_1  0x55
#define PROTO_MAGIC_2  0xA5
#define PROTO_MAGIC_3  0x5A

/* Frame types (see protocol_balldetector_n6.md). */
typedef enum {
    PROTO_T_HELLO     = 0x01,  /* H->F  */
    PROTO_T_INFO      = 0x02,  /* F->H  */
    PROTO_T_IMG_BEGIN = 0x10,  /* H->F  */
    PROTO_T_IMG_CHUNK = 0x11,  /* H->F  */
    PROTO_T_IMG_END   = 0x12,  /* H->F  */
    PROTO_T_INFER_RAW = 0x20,  /* F->H  (debug: raw int8 heads) */
    PROTO_T_INFER_DEC = 0x21,  /* F->H  (decoded boxes, default) */
    PROTO_T_CAM_START = 0x30,  /* H->F  */
    PROTO_T_CAM_STOP  = 0x31,  /* H->F  */
    PROTO_T_CAM_FRAME = 0x32,  /* F->H  */
    PROTO_T_LOG       = 0xFE,  /* F->H  UTF-8 string */
    PROTO_T_ACK       = 0xFF,  /* both  status:u8 */
} proto_type_t;

/* CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF). Matches host crc16_ccitt(). */
uint16_t proto_crc16_ccitt(const uint8_t *data, size_t len, uint16_t crc);

/* Bind the framer to the project UART. `huart` is a UART_HandleTypeDef*
 * (opaque here to avoid pulling in the HAL from this header). */
void proto_init(void *huart);

/* Blocking transmit of one framed packet. */
void proto_send_frame(uint8_t type, const uint8_t *payload, uint32_t len);

/* Convenience emitters. */
void proto_send_info(void);
void proto_send_log(const char *msg);
void proto_send_infer_dec(uint32_t inference_us, const yolo_box_t *boxes, uint16_t n);

/* Blocking receive of one framed packet into the caller's buffer.
 * Returns 1 and sets *type/*len on success, 0 on timeout, <0 on CRC/overflow.
 * Payloads longer than `cap` are rejected (returns <0). */
int proto_recv_frame(uint8_t *type, uint8_t *payload, uint32_t cap, uint32_t *len);

#ifdef __cplusplus
}
#endif

#endif /* UART_PROTOCOL_H */
