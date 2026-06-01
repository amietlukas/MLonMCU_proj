/* BallDetector_N6 application layer.
 *
 * Called from the X-CUBE-AI lifecycle hooks in app_x-cube-ai.c:
 *   MX_X_CUBE_AI_Init()    -> balldetector_init()   (after NPU clk/reset/cache)
 *   MX_X_CUBE_AI_Process() -> balldetector_run()    (never returns)
 *
 * Two paths share the NPU + decoder:
 *   IMG mode (implemented): host streams an RGB888 frame over UART, we infer
 *     and reply INFER_DEC. No camera needed — used for accuracy validation.
 *   CAM mode (TODO):        DCMIPP/IMX335 capture -> infer -> CAM_FRAME. Needs
 *     the IMX335 BSP bring-up + HAL_DCMIPP_CSI_PIPE_Start that CubeMX does not
 *     generate; see balldetector_run().
 */

#include "balldetector_app.h"
#include "constants.h"
#include "uart_protocol.h"
#include "yolo_postproc.h"

#include "main.h"             /* HAL, huart */
#include "app_x-cube-ai.h"    /* ll_aton_runtime.h, npu_cache.h */
#include "network_prunedint8.h"
#include "stm32n6xx_nucleo_xspi.h"  /* BSP OctoFlash */
#include "npu_cache.h"              /* CACHEAXI clean/invalidate for NPU I/O */
#include "stm32n6xx_hal_rif.h"      /* RISAF/RIMC: grant the NPU master memory access */

#include <string.h>
#include <stdio.h>            /* snprintf for debug logs */

/* This TU owns its own (static) network instance + interface, both pointing at
 * the generated network's global functions. app_x-cube-ai.c declares its own
 * static one too — harmless, that copy is unused once balldetector_run() takes
 * over MX_X_CUBE_AI_Process. */
LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(network_prunedint8)

/* VCP UART. Enable USART1 in CubeMX (Async, 921600 8N1, routed to ST-LINK);
 * that generates huart1 + MX_USART1_UART_Init() called from main(). */
extern UART_HandleTypeDef huart1;

volatile uint8_t g_frame_ready = 0;

/* NPU I/O buffer pointers, resolved once at init from the LL_ATON descriptors
 * (do NOT cache the result of a getter call across networks — read the
 * descriptors; see feedback_xcubeai_inputs_get_gotcha). */
static uint8_t       *s_in;        /* 331776 B, uint8 RGB                    */
static const uint8_t *s_out[YOLO_NUM_HEADS];  /* p8 / p16 / p32 int8 heads   */
static uint8_t        s_weights_mapped;
static int32_t        s_xspi_init_rc;
static int32_t        s_xspi_mmap_rc;

/* Output buffer sizes (bytes) from the generated network header — used for the
 * post-inference cache invalidation so the CPU reads the NPU's fresh outputs. */
static const uint32_t OUT_SZ[YOLO_NUM_HEADS] = {
    LL_ATON_NETWORK_PRUNEDINT8_OUT_1_SIZE_BYTES,   /* 8640 */
    LL_ATON_NETWORK_PRUNEDINT8_OUT_2_SIZE_BYTES,   /* 2160 */
    LL_ATON_NETWORK_PRUNEDINT8_OUT_3_SIZE_BYTES,   /*  540 */
};

static inline uint32_t round_up32(uint32_t n) { return (n + 31u) & ~31u; }

/* Fault reporter: dump the stacked frame + Cortex-M fault status registers over
 * the VCP so we can see exactly what/where faulted (called from HardFault_Handler
 * in stm32n6xx_it.c via a naked asm trampoline that passes the exception frame).
 * frame = [r0 r1 r2 r3 r12 lr pc xpsr]. Halts after one report. */
void balld_fault_report(uint32_t *frame)
{
    uint32_t cfsr = SCB->CFSR, hfsr = SCB->HFSR;
    uint32_t bfar = SCB->BFAR, mmfar = SCB->MMFAR;
    char d[128];
    snprintf(d, sizeof d, "FAULT pc=%08lX lr=%08lX cfsr=%08lX hfsr=%08lX bfar=%08lX mmfar=%08lX",
             (unsigned long)frame[6], (unsigned long)frame[5],
             (unsigned long)cfsr, (unsigned long)hfsr,
             (unsigned long)bfar, (unsigned long)mmfar);
    proto_send_log(d);
    for (;;) { __NOP(); }
}

/* ---- N6 Resource Isolation Framework: grant the NPU master memory access ---
 * THE root-cause fix. After reset the RISAF allows only secure/privileged/CID=1
 * transactions; the CPU matches that, but the Neural-ART NPU is a separate bus
 * master whose transactions were being filtered out — so the NPU read zeros and
 * its writes vanished, making every inference input-independent.
 *
 * Full bring-up mirroring ST's X-CUBE-AI 10.2.0 reference
 * (Projects/.../hello_world/Src/misc_toolbox.c + main.c), in its exact order:
 *   NPU_Config()  : enable CACHEAXI (clk+reset+init+enable), set the NPU RIMC to
 *                   CID1/secure/priv.
 *   RISAF_Config(): open the RISAF base regions for the memories the NPU uses.
 * Order matters: the CACHEAXI must be fully ENABLED before its RISAF regions
 * (8/15) are configured, otherwise that config faults (which it did when we
 * opened them with the cache only clocked). */
static uint32_t risaf_max_addr(RISAF_TypeDef *risaf)
{
    if      ((risaf == RISAF2_S)  || (risaf == RISAF2_NS))  return RISAF2_LIMIT_ADDRESS_SPACE_SIZE;
    else if ((risaf == RISAF3_S)  || (risaf == RISAF3_NS))  return RISAF3_LIMIT_ADDRESS_SPACE_SIZE;
    else if ((risaf == RISAF4_S)  || (risaf == RISAF4_NS))  return RISAF4_LIMIT_ADDRESS_SPACE_SIZE;
    else if ((risaf == RISAF5_S)  || (risaf == RISAF5_NS))  return RISAF5_LIMIT_ADDRESS_SPACE_SIZE;
    else if ((risaf == RISAF6_S)  || (risaf == RISAF6_NS))  return RISAF6_LIMIT_ADDRESS_SPACE_SIZE;
    else if ((risaf == RISAF7_S)  || (risaf == RISAF7_NS))  return RISAF7_LIMIT_ADDRESS_SPACE_SIZE;
    else if ((risaf == RISAF8_S)  || (risaf == RISAF8_NS))  return RISAF8_LIMIT_ADDRESS_SPACE_SIZE;
    else if ((risaf == RISAF15_S) || (risaf == RISAF15_NS)) return RISAF15_LIMIT_ADDRESS_SPACE_SIZE;
    else if ((risaf == RISAF12_S) || (risaf == RISAF12_NS)) return RISAF12_LIMIT_ADDRESS_SPACE_SIZE;
    return 0U;
}

static void set_risaf_default(RISAF_TypeDef *risaf)
{
    RISAF_BaseRegionConfig_t c;
    c.StartAddress  = 0x0;
    c.EndAddress    = risaf_max_addr(risaf);
    c.Filtering     = RISAF_FILTER_ENABLE;
    c.PrivWhitelist = RIF_CID_NONE;   /* all compartments, priv + unpriv */
    c.ReadWhitelist = RIF_CID_MASK;   /* all compartments may read       */
    c.WriteWhitelist= RIF_CID_MASK;   /* all compartments may write      */
    c.Secure = RIF_ATTRIBUTE_SEC;
    HAL_RIF_RISAF_ConfigBaseRegion(risaf, 0, &c);
    c.Secure = RIF_ATTRIBUTE_NSEC;
    HAL_RIF_RISAF_ConfigBaseRegion(risaf, 1, &c);
}

static void balld_security_config(void)
{
    /* --- NPU_Config: CACHEAXI on (weights pool is cacheable=ON) + NPU RIMC ---
     * Enable BOTH the cache controller clock AND the cache RAM clock
     * (CACHEAXIRAM). Missing the RAM clock is what hung the RISAF8 (NPU_CACHE)
     * configuration below — that region protects the cache RAM, and touching it
     * with the RAM unclocked stalls forever. Mirrors ST's
     * npu_cache_enable_clocks_and_reset(). */
    proto_send_log("sec: begin");
    __HAL_RCC_CACHEAXIRAM_MEM_CLK_ENABLE();
    __HAL_RCC_CACHEAXI_CLK_ENABLE();
    __HAL_RCC_CACHEAXI_FORCE_RESET();
    __HAL_RCC_CACHEAXI_RELEASE_RESET();
    npu_cache_init();
    npu_cache_enable();
    proto_send_log("sec: cache on");

    __HAL_RCC_RIFSC_CLK_ENABLE();
    RIMC_MasterConfig_t m;
    m.MasterCID = RIF_CID_1;
    m.SecPriv   = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_PRIV;
    HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_NPU, &m);
    /* Do NOT secure the NPU peripheral (RISC): the LL_ATON runtime accesses the
     * NPU registers via the NON-SECURE alias (ATON_BASE=NPU_BASE_NS, since
     * CPU_IN_SECURE_STATE is undefined). Securing the peripheral makes every
     * NPU register access from the runtime fault (NS-alias access to a secure
     * slave). The open RISAF base regions already grant the NPU master's memory
     * accesses to all CIDs, so the peripheral can stay non-secure. */
    /* HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_NPU,
                                          RIF_ATTRIBUTE_PRIV | RIF_ATTRIBUTE_SEC); */
    proto_send_log("sec: rimc ok");

    /* --- RISAF_Config: open the base regions (cache now enabled) ------------- */
    __HAL_RCC_RISAF_CLK_ENABLE();
    set_risaf_default(RISAF2_S);   /* SRAM1_AXI            */
    proto_send_log("sec: risaf2 ok");
    set_risaf_default(RISAF3_S);   /* SRAM2_AXI (our code) */
    proto_send_log("sec: risaf3 ok");
    set_risaf_default(RISAF4_S);   /* NPU MST0             */
    set_risaf_default(RISAF5_S);   /* NPU MST1             */
    proto_send_log("sec: risaf4-5 ok");
    set_risaf_default(RISAF6_S);   /* SRAM3,4,5,6_AXI <- activation pool */
    set_risaf_default(RISAF7_S);   /* FLEXMEM              */
    proto_send_log("sec: risaf6-7 ok");
    set_risaf_default(RISAF8_S);   /* NPU_CACHE            */
    proto_send_log("sec: risaf8 ok");
    set_risaf_default(RISAF15_S);  /* NPU_CACHE config     */
    proto_send_log("sec: risaf15 ok");
    set_risaf_default(RISAF12_S);  /* OCTOSPI2 0x70000000 <- weights     */
    proto_send_log("sec: risaf12 ok");
}

/* ---- xSPI2 OctoFlash: memory-mapped read via ST's Nucleo BSP -------------
 * The MX25UM51245G holds the NPU weights. The programmer leaves it in OPI
 * mode, and a hand-rolled mode switch proved unreliable, so we use ST's tested
 * BSP driver: it resets the flash from whatever mode it's in, configures
 * octal-DTR, and enables memory-mapped read. After this the NPU reads weights
 * at 0x71000000 (XSPI2 memory-mapped base 0x70000000 + 16 MB). */
static HAL_StatusTypeDef octoflash_mmap_init(int32_t *init_rc, int32_t *mmap_rc)
{
    BSP_XSPI_NOR_Init_t init = {
        .InterfaceMode = BSP_XSPI_NOR_OPI_MODE,
        .TransferRate  = BSP_XSPI_NOR_DTR_TRANSFER,
    };
    *init_rc = BSP_XSPI_NOR_Init(0, &init);
    if (*init_rc != BSP_ERROR_NONE) {
        return HAL_ERROR;
    }
    *mmap_rc = BSP_XSPI_NOR_EnableMemoryMappedMode(0);
    if (*mmap_rc != BSP_ERROR_NONE) {
        return HAL_ERROR;
    }
    return HAL_OK;
}

/* ---- microsecond timer (DWT cycle counter) ------------------------------- */
static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}
static inline uint32_t dwt_us(uint32_t cyc)
{
    return cyc / (SystemCoreClock / 1000000u);
}

/* ---- one blocking NPU inference (same sequence as the X-CUBE-AI template) - */
static int run_inference(void)
{
    if (!s_weights_mapped) {
        proto_send_log("ERROR: inference skipped: xSPI2 weights are not memory-mapped");
        return 0;
    }

    /* The CPU just wrote the image into s_in (now through the M55 D-cache).
     * Clean+invalidate that range so the writes hit SRAM before the NPU reads
     * the input. The activation pools are non-cacheable on the NPU/CACHEAXI
     * side, so flushing the M55 D-cache is sufficient for coherency (this is
     * what ST's N6 getting-started does: SCB_CleanInvalidateDCache_by_Addr). */
    SCB_CleanInvalidateDCache_by_Addr((void *)s_in, (int32_t)MODEL_INPUT_BYTES);

    /* DIAG: stamp the OUTPUT buffers with a CPU-written sentinel (0xAB) before
     * the NPU runs. If the NPU writes CPU-visible output, the post-inference
     * "out sums" will NOT match the all-0xAB pattern (p8=0x168CC0, p16=0x5A2D0,
     * p32=0x00168F4). If they DO match 0xAB, the NPU never wrote here -> the NPU
     * and CPU don't share the activation SRAM (root cause). */
    for (int i = 0; i < YOLO_NUM_HEADS; ++i)
        memset((void *)s_out[i], 0xAB, OUT_SZ[i]);

    LL_ATON_RT_RetValues_t r;
    LL_ATON_RT_Init_Network(&NN_Instance_network_prunedint8);

    /* DEBUG: after Init_Network, where does the runtime say the input is, what
     * does the getter return, and are our pixels actually there? Compares to
     * s_in (what we wrote). */
    {
        const LL_Buffer_InfoTypeDef *ii =
            NN_Interface_network_prunedint8.input_buffers_info();
        const uint8_t *ia = (const uint8_t *)LL_Buffer_addr_start(&ii[0]);
        void *ig = LL_ATON_Get_User_Input_Buffer(&NN_Instance_network_prunedint8, 0);
        char d[100];
        snprintf(d, sizeof d,
                 "NPU in: info@%08lX b0=%02X%02X%02X  get@%08lX  s_in@%08lX",
                 (unsigned long)(uintptr_t)ia, ia[0], ia[1], ia[2],
                 (unsigned long)(uintptr_t)ig, (unsigned long)(uintptr_t)s_in);
        proto_send_log(d);
    }

    /* DEBUG: the NPU end-of-epoch IRQ (NPU0_IRQn=53) faulted to PC=0. Check the
     * live vector base + the IRQ-53 slot + whether it's enabled and secure.
     * IRQ53 -> ISER[1]/ITNS[1] bit 21; vector at VTOR + (16+53)*4. */
    {
        uint32_t vtor = SCB->VTOR;
        uint32_t vec53 = *(volatile uint32_t *)(vtor + (16u + 53u) * 4u);
        uint32_t iser1 = NVIC->ISER[1];
        uint32_t itns1 = NVIC->ITNS[1];
        char d[120];
        snprintf(d, sizeof d,
                 "IRQ dbg: vtor=%08lX vec53=%08lX iser1.b21=%lu itns1.b21=%lu",
                 (unsigned long)vtor, (unsigned long)vec53,
                 (unsigned long)((iser1 >> 21) & 1u), (unsigned long)((itns1 >> 21) & 1u));
        proto_send_log(d);
    }

    do {
        r = LL_ATON_RT_RunEpochBlock(&NN_Instance_network_prunedint8);
        if (r == LL_ATON_RT_WFE) {
            LL_ATON_OSAL_WFE();
        }
    } while (r != LL_ATON_RT_DONE);
    LL_ATON_RT_DeInit_Network(&NN_Instance_network_prunedint8);

    /* Make the NPU's freshly written outputs visible to the CPU decoder. */
    for (int i = 0; i < YOLO_NUM_HEADS; ++i) {
        uint32_t sz = round_up32(OUT_SZ[i]);
        /* DIAG: CACHEAXI forced off; only the M55 D-cache (off too) maintenance. */
        /* npu_cache_clean_invalidate_range((uint32_t)s_out[i], (uint32_t)s_out[i] + sz); */
        SCB_InvalidateDCache_by_Addr((void *)s_out[i], (int32_t)sz);
    }

    /* DEBUG: checksum each raw output head + log their addresses. If these vary
     * per image, the NPU computed on our input (decode is then the suspect); if
     * constant, the NPU/output buffer is the problem. */
    {
        char d[110];
        uint32_t os[YOLO_NUM_HEADS] = {0};
        for (int i = 0; i < YOLO_NUM_HEADS; ++i)
            for (uint32_t k = 0; k < OUT_SZ[i]; ++k) os[i] += s_out[i][k];
        /* DIAG: read the INPUT pool back AFTER inference. 0x342e0000 is reused
         * as scratch by many mid-network tensors, so if the NPU shares this
         * SRAM with the CPU these bytes will now hold NPU scratch (!= our fill).
         * If they STILL equal our fill byte, the NPU never wrote here -> the NPU
         * and CPU do NOT see the same memory at 0x342e0000 (routing/RIF/power),
         * which is why our input is ignored. */
        SCB_InvalidateDCache_by_Addr((void *)s_in, 32);
        snprintf(d, sizeof d, "in pool after infer: b0=%02X%02X%02X%02X (fill was last DIAG)",
                 s_in[0], s_in[1], s_in[2], s_in[3]);
        proto_send_log(d);

        snprintf(d, sizeof d, "out sums: p8=%08lX p16=%08lX p32=%08lX @%08lX/%08lX/%08lX",
                 (unsigned long)os[0], (unsigned long)os[1], (unsigned long)os[2],
                 (unsigned long)(uintptr_t)s_out[0], (unsigned long)(uintptr_t)s_out[1],
                 (unsigned long)(uintptr_t)s_out[2]);
        proto_send_log(d);
    }
    return 1;
}

/* Decode the three int8 heads -> boxes, return count. */
static int decode(yolo_box_t *boxes)
{
    const uint8_t *heads[YOLO_NUM_HEADS] = { s_out[0], s_out[1], s_out[2] };
    return yolo_postprocess(heads, DET_CONF_THRESH, DET_NMS_IOU,
                            boxes, DET_MAX_BOXES);
}

static void send_xspi_status(void)
{
    char d[96];
    if (s_weights_mapped) {
        snprintf(d, sizeof d, "xSPI2 OctoFlash memory-map OK init=%ld mmap=%ld",
                 (long)s_xspi_init_rc, (long)s_xspi_mmap_rc);
    } else {
        snprintf(d, sizeof d, "ERROR: xSPI2 OctoFlash memory-map failed init=%ld mmap=%ld",
                 (long)s_xspi_init_rc, (long)s_xspi_mmap_rc);
    }
    proto_send_log(d);

    /* DIAG: read the weights back through the memory-mapped flash exactly where
     * the NPU expects them (network_prunedint8.c: every Conv2D_*_weights
     * .addr_base = 0x71000000). Compare to the host's blob: first16 should be
     * f90100d2eac9dd0b... and sum of first 64KB should be 0x0080FCFC. If these
     * don't match, the NPU is reading garbage weights -> input-independent
     * (constant) output, which is exactly what we observe. */
    {
        const volatile uint8_t *w = (const volatile uint8_t *)0x71000000UL;
        uint32_t s = 0;
        for (uint32_t k = 0; k < 65536u; ++k) s += w[k];
        snprintf(d, sizeof d,
                 "WEIGHTS @0x71000000: b0=%02X%02X%02X%02X%02X%02X%02X%02X sum64k=0x%08lX",
                 w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], (unsigned long)s);
        proto_send_log(d);
    }
}

void balldetector_init(void)
{
    dwt_init();
    proto_init(&huart1);

    /* Route Bus/Hard/NMI faults to SECURE state and enable the configurable
     * fault handlers, so any fault during NPU register/DMA access is delivered
     * to our (secure) reporter instead of vanishing to the non-secure vector
     * table (VTOR_NS=0 -> PC=0). Without this, faults were going to 0x0. */
    SCB->SHCSR |= SCB_SHCSR_BUSFAULTENA_Msk | SCB_SHCSR_USGFAULTENA_Msk |
                  SCB_SHCSR_MEMFAULTENA_Msk | SCB_SHCSR_SECUREFAULTENA_Msk;
    SCB->AIRCR = (0x05FAUL << SCB_AIRCR_VECTKEY_Pos) |
                 (SCB->AIRCR & (SCB_AIRCR_PRIGROUP_Msk | SCB_AIRCR_PRIS_Msk));

    /* ROOT-CAUSE FIX: grant the NPU bus master access to the activation RAM and
     * weight flash via the RISAF, and set its RIMC to CID1/secure/priv. Without
     * this the NPU's memory accesses were filtered out (proven by the
     * diagnostics: the NPU never read our input pool nor wrote our output
     * sentinel). Must run before the first inference; do it first thing. */
    /* RISAF + NPU RIMC + CACHEAXI, in ST's reference order (cache enabled before
     * its RISAF regions). The activation pools are cacheable=OFF so the NPU
     * reads/writes them straight to SRAM; with the M55 D-cache off too, CPU<->NPU
     * I/O is coherent with no maintenance. (M55 D-cache stays off until it has an
     * MPU region for the xSPI flash, which it corrupts otherwise.) */
    balld_security_config();

    /* CubeMX's MX_USART1_UART_Init() disables the RX FIFO; re-enable it so the
     * hardware buffers bytes while our byte-by-byte framer is between reads.
     * Without this, the sustained image stream at 921600 overruns the 1-byte
     * RX register and the frame parser loses sync (the handshake still works
     * because it's tiny and the host pauses after it). */
    HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8);
    HAL_UARTEx_EnableFifoMode(&huart1);

    /* Map the OctoFlash so the NPU can read its weights at 0x71000000. Must
     * happen before any inference. Log (don't hard-fault) so a failure is
     * visible on the VCP instead of a silent lockup. */
    s_xspi_init_rc = 0;
    s_xspi_mmap_rc = 0;
    if (octoflash_mmap_init(&s_xspi_init_rc, &s_xspi_mmap_rc) != HAL_OK) {
        s_weights_mapped = 0;
    } else {
        s_weights_mapped = 1;
    }

    /* Resolve the NPU I/O buffer addresses from the network interface. */
    const LL_Buffer_InfoTypeDef *in_info  =
        NN_Interface_network_prunedint8.input_buffers_info();
    const LL_Buffer_InfoTypeDef *out_info =
        NN_Interface_network_prunedint8.output_buffers_info();
    s_in = (uint8_t *)LL_Buffer_addr_start(&in_info[0]);
    for (int i = 0; i < YOLO_NUM_HEADS; ++i) {
        s_out[i] = (const uint8_t *)LL_Buffer_addr_start(&out_info[i]);
    }

    LL_ATON_RT_RuntimeInit();
    proto_send_log("BallDetector_N6 ready");
}

void balldetector_run(void)
{
    static uint8_t rx[UART_MAX_CHUNK + 16];   /* IMG_CHUNK = idx(2) + <=4096 */
    yolo_box_t boxes[DET_MAX_BOXES];
    uint32_t img_off = 0;                     /* bytes received into s_in    */

    for (;;) {
        uint8_t  type;
        uint32_t len;
        int rc = proto_recv_frame(&type, rx, sizeof(rx), &len);
        if (rc <= 0) {
            continue;  /* timeout or framing error — just keep listening */
        }

        switch (type) {
        case PROTO_T_HELLO:
            send_xspi_status();
            proto_send_info();
            break;

        case PROTO_T_IMG_BEGIN:
            /* width/height/channels/fmt in payload; we assume they match the
             * model (host sends MODEL_W x MODEL_H, RGB888 packed HWC). */
            /* The host fills s_in through the CPU, bypassing CACHEAXI. ST's NPU
             * cache range maintenance is clean+invalidate, so do it before the
             * CPU writes the new frame; doing it after can restore stale NPU
             * cache lines over the fresh image. */
            /* DIAG: CACHEAXI forced off in init, so no NPU-cache maintenance. */
            /* npu_cache_clean_invalidate_range((uint32_t)s_in,
                                             (uint32_t)s_in + MODEL_INPUT_BYTES); */
            img_off = 0;
            break;

        case PROTO_T_IMG_CHUNK: {
            /* payload: chunk_idx:u16, then the pixel bytes. Place by index so
             * we're robust to reordering. Host sends HWC-packed RGB888; the
             * NPU input buffer is fed in the same byte order. */
            if (len < 2) { break; }
            uint32_t idx = (uint32_t)rx[0] | ((uint32_t)rx[1] << 8);
            uint32_t off = idx * UART_MAX_CHUNK;
            uint32_t n   = len - 2;
            if (off + n <= MODEL_INPUT_BYTES) {
                memcpy(s_in + off, rx + 2, n);
                img_off = off + n;
            }
            break;
        }

        case PROTO_T_IMG_END: {
            char dbg[96];
            if (img_off != MODEL_INPUT_BYTES) {
                snprintf(dbg, sizeof dbg, "IMG size mismatch: %lu/%u",
                         (unsigned long)img_off, (unsigned)MODEL_INPUT_BYTES);
                proto_send_log(dbg);
                break;
            }
            /* Checksum the received image so we can tell if (a) different
             * images actually arrive distinct in s_in, and (b) which buffer. */
            uint32_t sum = 0;
            for (uint32_t k = 0; k < MODEL_INPUT_BYTES; ++k) sum += s_in[k];
            snprintf(dbg, sizeof dbg,
                     "image received: sum=0x%08lX b0=%02X%02X%02X @0x%08lX",
                     (unsigned long)sum, s_in[0], s_in[1], s_in[2],
                     (unsigned long)(uintptr_t)s_in);
            proto_send_log(dbg);

            /* ==== TEMP DIAGNOSTIC (remove after) ===========================
             * Overwrite the received image with a uniform value that changes
             * every inference (0x00 black -> 0x80 gray -> 0xFF white). If the
             * NPU truly recomputes on its input, "out sums" MUST change between
             * runs. If they stay identical across all-black vs all-white, the
             * NPU is NOT recomputing (runtime/buffer-binding bug, not cache).
             * The IMG_BEGIN already invalidated CACHEAXI over this region; these
             * CPU writes land in RAM (D-cache off) and run_inference cleans the
             * D-cache before the NPU reads. */
            {
                static const uint8_t fills[3] = { 0x00, 0x80, 0xFF };
                static uint8_t fi = 0;
                uint8_t fv = fills[fi];
                fi = (uint8_t)((fi + 1) % 3);
                memset(s_in, fv, MODEL_INPUT_BYTES);
                snprintf(dbg, sizeof dbg, "DIAG: input overwritten with 0x%02X", fv);
                proto_send_log(dbg);
            }

            uint32_t t0 = DWT->CYCCNT;
            if (!run_inference()) {
                proto_send_infer_dec(0, boxes, 0);
                break;
            }
            uint32_t us = dwt_us(DWT->CYCCNT - t0);
            snprintf(dbg, sizeof dbg, "inference done: %lu us", (unsigned long)us);
            proto_send_log(dbg);
            int n = decode(boxes);
            snprintf(dbg, sizeof dbg, "post-process done: %d boxes", n);
            proto_send_log(dbg);
            proto_send_infer_dec(us, boxes, (uint16_t)n);
            break;
        }

        case PROTO_T_CAM_START:
            /* TODO(CAM mode): bring up the IMX335 over I2C1 (ST BSP), start
             * DCMIPP Pipe1 capture into s_in (HAL_DCMIPP_CSI_PIPE_Start), set
             * g_frame_ready in HAL_DCMIPP_PIPE_FrameEventCallback, then per
             * frame run_inference() + decode() + emit CAM_FRAME/INFER_DEC.
             * The motor/servo control loop also plugs in on this path. */
            proto_send_log("CAM mode not implemented yet");
            break;

        case PROTO_T_CAM_STOP:
        default:
            break;
        }
    }
}
