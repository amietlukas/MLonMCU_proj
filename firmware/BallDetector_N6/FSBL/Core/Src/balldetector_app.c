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
#include "ll_aton.h"          /* ATON_STRENG_* accessors for the stall watchdog */
#include "stm32n6xx_nucleo_xspi.h"  /* BSP OctoFlash */
#include "npu_cache.h"              /* CACHEAXI clean/invalidate for NPU I/O */
#include "stm32n6xx_hal_rif.h"      /* RISAF/RIMC: grant the NPU master memory access */

#include <string.h>
#include <stdio.h>            /* snprintf for debug logs */

/* Keep our manual RISAF base-region opening (1) or rely on RIMC+RISC alone (0,
 * matching ST's x-cube-n6-ai reference, which never reconfigures RISAF). Gated
 * ON by default because our RAM-boot path skips the signed bootROM that would
 * otherwise set the RISAF defaults; set to 0 to test ST's minimal recipe. */
#ifndef BALLD_CONFIG_RISAF
#define BALLD_CONFIG_RISAF 1
#endif

/* Secure the NPU peripheral as a RISC slave (1) or leave it at reset default
 * (0). ST's reference secures it; set to 0 to A/B-test whether securing the NPU
 * is what makes the streaming engine stall in LL_Streng_Wait. */
#ifndef BALLD_CONFIG_SECURE_NPU
#define BALLD_CONFIG_SECURE_NPU 0   /* A/B: NPU slave left at reset default (was 1) */
#endif

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

/* ---- NPU streaming-engine stall guard (SysTick-driven) -------------------
 * LL_ATON's own watchdog (checkWatchdog) is compiled out (NDEBUG), so a stuck
 * streaming engine spins forever in LL_Streng_Wait() with no escape — the
 * observed hang. We piggy-back on the 1 kHz HAL SysTick instead: while an
 * inference is in flight (s_infer_active), count ms; if it runs well past the
 * ~105 ms norm, dump which streaming engine is still RUNNING and the address it
 * is stuck transferring, then halt. A stuck addr in XSPI2 (0x70/0x71xxxxxx) =>
 * NPU-master weight read blocked; in AXISRAM (0x34xxxxxx) => an activation/IO
 * buffer the NPU can't reach. Overriding the __weak HAL_IncTick keeps normal
 * HAL timekeeping (uwTick) intact. */
extern __IO uint32_t uwTick;            /* HAL tick counter */
static volatile uint8_t  s_infer_active;
static volatile uint32_t s_infer_ms;
#define BALLD_STALL_MS 1500u

void HAL_IncTick(void)
{
    uwTick += (uint32_t)uwTickFreq;     /* preserve HAL_Delay/HAL_GetTick */
    if (!s_infer_active) { return; }
    if (++s_infer_ms < BALLD_STALL_MS) { return; }

    s_infer_active = 0;                 /* one-shot */
    proto_send_log("STALL: NPU epoch over budget — streaming engines RUNNING:");
    char d[96];
    for (int i = 0; i < ATON_STRENG_NUM; ++i) {
        uint32_t ctrl = ATON_STRENG_CTRL_GET(i);
        if (ctrl & (1U << ATON_STRENG_CTRL_RUNNING_LSB)) {
            snprintf(d, sizeof d, "  streng[%d] ctrl=%08lX addr=%08lX",
                     i, (unsigned long)ctrl,
                     (unsigned long)ATON_STRENG_ADDR_GET(i));
            proto_send_log(d);
        }
    }
    proto_send_log("STALL: halting");
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
/* ---- AXISRAM power-up (THE streng-stall fix) ----------------------------
 * MX_X_CUBE_AI_Init() clears RAMCFG_SRAMx_AXI->CR.SRAMSD to bring AXISRAM2..6
 * out of shutdown, but it never enables the RAMCFG peripheral clock first — so
 * those APB writes hit a clock-gated peripheral and are SILENTLY DROPPED. Banks
 * that default to shutdown (AXISRAM2 @0x34100000) then stay powered down, and
 * the NPU stalls forever in LL_Streng_Wait when a streaming engine reads such a
 * bank (no AXI response). The CPU-written input sits in AXISRAM3 (@0x342E0000),
 * which is on by default — masking the dead AXISRAM2. Enable the RAMCFG clock,
 * then redo the un-shutdown so it actually takes effect (mirrors ST's
 * NPURam_enable()). */
static void balld_enable_axisram(void)
{
    __HAL_RCC_RAMCFG_CLK_ENABLE();
    RAMCFG_SRAM2_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM3_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM4_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM5_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    RAMCFG_SRAM6_AXI->CR &= ~RAMCFG_CR_SRAMSD;
    __DSB();
    __ISB();
    proto_send_log("axisram: ramcfg clk on, sram2-6 un-shutdown");
}

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
    /* Secure the NPU peripheral as a RISC slave — matching ST's validated
     * x-cube-n6-ai reference Security_Config(). This build DEFINES
     * CPU_IN_SECURE_STATE (see FSBL/.cproject), so the LL_ATON runtime accesses
     * the NPU registers through the SECURE alias (ATON_BASE=NPU_BASE_S). A secure
     * CPU touching a secure slave is legal, so securing it is consistent — and
     * stricter than leaving the slave at its reset default. (Prior code left
     * this out on the FALSE premise that CPU_IN_SECURE_STATE was undefined / the
     * NS alias was in use.) */
#if BALLD_CONFIG_SECURE_NPU
    HAL_RIF_RISC_SetSlaveSecureAttributes(RIF_RISC_PERIPH_INDEX_NPU,
                                          RIF_ATTRIBUTE_PRIV | RIF_ATTRIBUTE_SEC);
    proto_send_log("sec: rimc+risc ok");
#else
    proto_send_log("sec: rimc ok (NPU slave left at reset default)");
#endif

#if BALLD_CONFIG_RISAF
    /* --- RISAF_Config: open the base regions (cache now enabled) -------------
     * ST's reference does NOT touch RISAF at all — RIMC (master) + RISC (slave)
     * suffice in both flash- and dev/RAM-boot. We keep this gated-ON because our
     * RAM-boot path skips the signed bootROM that would set RISAF defaults; set
     * BALLD_CONFIG_RISAF=0 to test whether RIMC+RISC alone is enough. */
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
#else
    proto_send_log("sec: risaf skipped (RIMC+RISC only, ST-style)");
#endif
}

/* ---- IAC: trap RIF illegal accesses -------------------------------------
 * Mirrors ST's IAC_Config() + IAC_IRQHandler. With this armed, any transaction
 * the RIF filters out (e.g. the NPU master/slave being blocked — the suspected
 * root cause) raises an interrupt and halts in a known handler with a log,
 * instead of silently producing constant/garbage NPU output. The RIFSC
 * peripheral banks map to ISR[0..4]; the NPU sits in bank 3, bit 10. */
void IAC_IRQHandler(void)
{
    char d[96];
    for (int i = 0; i < 5; ++i) {
        uint32_t isr = IAC->ISR[i];
        if (isr) {
            snprintf(d, sizeof d,
                     "IAC ILLEGAL ACCESS: ISR[%d]=%08lX (NPU=ISR[3].bit10)",
                     i, (unsigned long)isr);
            proto_send_log(d);
            IAC->ICR[i] = isr;   /* acknowledge */
        }
    }
    for (;;) { __NOP(); }
}

static void balld_iac_config(void)
{
    __HAL_RCC_IAC_CLK_ENABLE();
    __HAL_RCC_IAC_FORCE_RESET();
    __HAL_RCC_IAC_RELEASE_RESET();
    for (int i = 0; i < 5; ++i) {
        IAC->ICR[i] = 0xFFFFFFFFu;   /* clear stale latches            */
        IAC->IER[i] = 0xFFFFFFFFu;   /* trap every peripheral's illegal access */
    }
    HAL_NVIC_SetPriority(IAC_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(IAC_IRQn);
    proto_send_log("sec: iac trap armed");
}

/* ---- NPU interrupt vectoring fix ----------------------------------------
 * The NPU has 5 IRQ lines: NPU0..NPU3 (53..56) + CACHEAXI (57). The LL_ATON
 * runtime installs/handles only line 53 (ATON_STD_IRQ_LINE=0). If the NPU
 * raises completion/cache on another line that is non-secure-targeted, it
 * dispatches through the empty non-secure vector table (VTOR_NS=0) -> PC=0, the
 * crash we see mid-epoch. Force all five lines to SECURE so they use our
 * populated secure VTOR, and give 54..57 (which the runtime leaves unhandled) a
 * reporting handler so an unexpected line announces itself instead of crashing. */
static void npu_unexpected_irq(int irqn)
{
    char d[72];
    snprintf(d, sizeof d, "UNEXPECTED NPU IRQ %d fired (runtime handles only 53)", irqn);
    proto_send_log(d);
    NVIC_DisableIRQ((IRQn_Type)irqn);
    for (;;) { __NOP(); }
}
void NPU1_IRQHandler(void)     { npu_unexpected_irq(NPU1_IRQn); }
void NPU2_IRQHandler(void)     { npu_unexpected_irq(NPU2_IRQn); }
void NPU3_IRQHandler(void)     { npu_unexpected_irq(NPU3_IRQn); }
void CACHEAXI_IRQHandler(void) { npu_unexpected_irq(CACHEAXI_IRQn); }

static void npu_irq_secure_target(void)
{
    /* IRQ 53..57 -> bits 21..25 of NVIC ITNS[1] (n - 32). Clear = secure. */
    NVIC->ITNS[1] &= ~(0x1Fu << 21);
    __DSB();
    __ISB();
    proto_send_log("sec: npu irqs forced secure (53-57)");
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

/* ---- weights: XSPI2 staging -> internal AXISRAM (noextmem-fsbl profile) ----
 * The internal-only network (model/stm32n6__noextmem_fsbl.mpool) hardcodes its
 * weights in AXISRAM3/4/5 (0x34200000/0x34270000/0x342E0000), so the NPU master
 * reads ONLY on-chip RAM — XSPI2 is out of the inference path. flash_weights.sh
 * stages the three 448 KB blobs in XSPI2 at 0x71000000/0x71070000/0x710E0000;
 * we copy them into AXISRAM here, once, after the OctoFlash is memory-mapped.
 * AXISRAM3/4/5 are already un-shutdown (balld_enable_axisram); D-cache is off so
 * these CPU writes land straight in SRAM (a DSB orders them before inference). */
#define WEIGHT_BANK_BYTES 0x70000u   /* 448 KB per AXISRAM bank == blob size */
static void copy_weights_to_axisram(void)
{
    memcpy((void *)0x34200000u, (const void *)0x71000000u, WEIGHT_BANK_BYTES);
    memcpy((void *)0x34270000u, (const void *)0x71070000u, WEIGHT_BANK_BYTES);
    memcpy((void *)0x342E0000u, (const void *)0x710E0000u, WEIGHT_BANK_BYTES);
    __DSB();
    proto_send_log("weights: copied 3x448K XSPI2 staging -> AXISRAM3/4/5");
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

    s_infer_ms = 0;
    s_infer_active = 1;   /* arm the SysTick stall guard for this inference */
    do {
        r = LL_ATON_RT_RunEpochBlock(&NN_Instance_network_prunedint8);
        if (r == LL_ATON_RT_WFE) {
            LL_ATON_OSAL_WFE();
        }
    } while (r != LL_ATON_RT_DONE);
    s_infer_active = 0;   /* completed within budget */
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
    /* Build tag — printed on every HELLO so we can confirm at a glance that the
     * binary actually running is THIS build (CubeIDE can silently launch a stale
     * .elf past its phantom-error prompt). Bump the date/suffix on each change. */
    proto_send_log("BUILD=2026-06-01T noextmem-fsbl conf045");
    /* Readback RAMCFG to confirm AXISRAM2/3 are actually out of shutdown
     * (SRAMSD bit clear). If sram2 still shows SRAMSD set, the un-shutdown did
     * not take -> the NPU will stall reading AXISRAM2 @0x34100000. */
    {
        char r[88];
        snprintf(r, sizeof r,
                 "RAMCFG.CR sram2=%08lX sram3=%08lX (SRAMSD mask=%08lX; 0=on)",
                 (unsigned long)RAMCFG_SRAM2_AXI->CR,
                 (unsigned long)RAMCFG_SRAM3_AXI->CR,
                 (unsigned long)RAMCFG_CR_SRAMSD);
        proto_send_log(r);
    }
    if (s_weights_mapped) {
        snprintf(d, sizeof d, "xSPI2 OctoFlash memory-map OK init=%ld mmap=%ld",
                 (long)s_xspi_init_rc, (long)s_xspi_mmap_rc);
    } else {
        snprintf(d, sizeof d, "ERROR: xSPI2 OctoFlash memory-map failed init=%ld mmap=%ld",
                 (long)s_xspi_init_rc, (long)s_xspi_mmap_rc);
    }
    proto_send_log(d);

    /* DIAG: confirm the boot copy worked — sum the first 64KB of the XSPI2
     * staging (0x71000000) and the AXISRAM copy (0x34200000, where the NPU
     * actually reads weights). They MUST match; a mismatch means the copy/flash
     * is wrong and the NPU will see garbage weights. */
    if (s_weights_mapped) {
        const volatile uint8_t *stg = (const volatile uint8_t *)0x71000000UL; /* XSPI2 staging  */
        const volatile uint8_t *axi = (const volatile uint8_t *)0x34200000UL; /* AXISRAM3 (NPU) */
        uint32_t ss = 0, sa = 0;
        for (uint32_t k = 0; k < 65536u; ++k) { ss += stg[k]; sa += axi[k]; }
        snprintf(d, sizeof d,
                 "WEIGHTS staging@71000000 sum64k=0x%08lX  npu@34200000 sum64k=0x%08lX %s",
                 (unsigned long)ss, (unsigned long)sa, (ss == sa) ? "(copy OK)" : "(MISMATCH!)");
        proto_send_log(d);
    } else {
        /* xSPI2 not mapped -> 0x71000000 would BusFault. Weights are NOT in AXISRAM. */
        proto_send_log("WEIGHTS NOT LOADED: xSPI2 mmap failed -> power-cycle board, re-flash weights, retry");
    }
}

void balldetector_init(void)
{
    dwt_init();
    proto_init(&huart1);

    /* Bring AXISRAM2..6 properly out of shutdown — the generated RAMCFG writes
     * in MX_X_CUBE_AI_Init ran with the RAMCFG clock gated and were dropped.
     * Must happen before the first inference; doing it first thing is safe. */
    balld_enable_axisram();

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

    /* Map the OctoFlash to read the staged weight blobs, then copy them into
     * AXISRAM (the NPU reads weights from on-chip RAM, not XSPI2). Log (don't
     * hard-fault) so a failure is visible on the VCP instead of a silent lockup. */
    s_xspi_init_rc = 0;
    s_xspi_mmap_rc = 0;
    if (octoflash_mmap_init(&s_xspi_init_rc, &s_xspi_mmap_rc) != HAL_OK) {
        s_weights_mapped = 0;
    } else {
        s_weights_mapped = 1;
        copy_weights_to_axisram();   /* XSPI2 staging -> AXISRAM3/4/5 for the NPU */
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

    /* Arm the IAC last — after the peripheral bring-up accesses (XSPI2 BSP,
     * UART) are done — so it traps only the NPU pipeline's accesses during
     * inference. If it fires immediately, the logged ISR bitmap names the
     * offending peripheral bank. */
    balld_iac_config();

    LL_ATON_RT_RuntimeInit();
    /* RuntimeInit installs the NPU line-53 handler; now force all NPU/cache IRQ
     * lines secure so a completion IRQ on any line can't dispatch to VTOR_NS=0
     * (the PC=0 crash). Must run after RuntimeInit. */
    npu_irq_secure_target();
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

            /* (removed) input-overwrite DIAG — confirmed the NPU computes on its
             * input (out sums varied with 0x00/0x80/0xFF). We now infer on the
             * real uploaded image. */

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
