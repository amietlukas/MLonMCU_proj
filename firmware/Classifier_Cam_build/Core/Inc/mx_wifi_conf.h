/**
  * mx_wifi_conf.h — bare-OS, SPI transport variant.
  * Copied from Drivers/BSP/Components/mx_wifi/mx_wifi_conf_template.h with:
  *   MX_WIFI_USE_SPI       = 1   (SPI2 transport on B-U585I-IOT02A)
  *   MX_WIFI_USE_CMSIS_OS  = 0   (no RTOS)
  *   DMA_ON_USE            = 1   (default; HAL DMA via SPI)
  */
#ifndef MX_WIFI_CONF_H
#define MX_WIFI_CONF_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <inttypes.h>
#include "main.h"

#define MX_WIFI_USE_SPI                 (1)
#define DMA_ON_USE                      (1)
#define MX_WIFI_USE_CMSIS_OS            (0)

/* Bypass the X-CUBE-AI __wrap_malloc/__wrap_free pointer-offset wrapper for
 * mx_wifi allocations — the BSP must always pair malloc with the same free,
 * but mx_wifi's free might land in libraries that link against the
 * __real_free, mismatching the +4 offset and corrupting the heap. */
extern void *__real_malloc(size_t bytes);
extern void  __real_free(void *ptr);
#define MX_WIFI_MALLOC __real_malloc
#define MX_WIFI_FREE   __real_free
#define MX_WIFI_NETWORK_BYPASS_MODE     (0)
#define MX_WIFI_TX_BUFFER_NO_COPY       (1)

#if (MX_WIFI_USE_CMSIS_OS == 1)
#include "mx_wifi_cmsis_os.h"
#else
#include "mx_wifi_bare_os.h"
#endif

#define MX_WIFI_PRODUCT_NAME            ("MXCHIP-WIFI")
#define MX_WIFI_PRODUCT_ID              ("EMW3080B")

#define MX_WIFI_UART_BAUDRATE           (115200 * 2)
#define MX_WIFI_MTU_SIZE                (1500)
#define MX_WIFI_BYPASS_HEADER_SIZE      (28)
#define MX_WIFI_PBUF_LINK_HLEN          (14)

#if (MX_WIFI_NETWORK_BYPASS_MODE == 1)
#define MX_WIFI_BUFFER_SIZE \
  (MX_WIFI_MTU_SIZE + MX_WIFI_BYPASS_HEADER_SIZE + MX_WIFI_PBUF_LINK_HLEN)
#else
#define MX_WIFI_BUFFER_SIZE             (2500)
#endif

#define MX_WIFI_IPC_PAYLOAD_SIZE        (MX_WIFI_BUFFER_SIZE - 6)
#define MX_WIFI_SOCKET_DATA_SIZE        (MX_WIFI_IPC_PAYLOAD_SIZE - 12)

#define MX_WIFI_CMD_TIMEOUT             (10000)
#define MX_WIFI_MAX_SOCKET_NBR          (8)
#define MX_WIFI_MAX_DETECTED_AP         (10)
#define MX_WIFI_MAX_SSID_NAME_SIZE      (32)
#define MX_WIFI_MAX_PSWD_NAME_SIZE      (64)
#define MX_WIFI_PRODUCT_NAME_SIZE       (32)
#define MX_WIFI_PRODUCT_ID_SIZE         (32)
#define MX_WIFI_FW_REV_SIZE             (24)

#ifndef MX_WIFI_MAX_RX_BUFFER_COUNT
#define MX_WIFI_MAX_RX_BUFFER_COUNT     (8)
#endif

#ifndef MX_CIRCULAR_UART_RX_BUFFER_SIZE
#define MX_CIRCULAR_UART_RX_BUFFER_SIZE (400)
#endif

/* Stats infrastructure — referenced by mx_wifi_spi.c via MX_STAT(alloc).
 * Disabled (compiles to no-ops) since we don't print stats. */
#define MX_STAT_ON                      0

#if (MX_STAT_ON == 1)
typedef struct {
  uint32_t alloc;
  uint32_t free;
  uint32_t cmd_get_answer;
  uint32_t callback;
  uint32_t in_fifo;
  uint32_t out_fifo;
} mx_stat_t;
extern mx_stat_t mx_stat;
#define MX_STAT_INIT()    (void) memset((void*)&mx_stat, 0, sizeof(mx_stat))
#define MX_STAT(A)        mx_stat.A++
#define MX_STAT_LOG()
#define MX_STAT_DECLARE() mx_stat_t mx_stat
#else
#define MX_STAT_INIT()
#define MX_STAT(A)
#define MX_STAT_LOG()
#define MX_STAT_DECLARE()
#endif

#ifdef __cplusplus
}
#endif

#endif /* MX_WIFI_CONF_H */
