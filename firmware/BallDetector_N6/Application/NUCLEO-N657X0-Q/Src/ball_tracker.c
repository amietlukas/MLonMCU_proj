/**
 ******************************************************************************
 * @file    ball_tracker.c
 * @brief   1-D ball tracker + 50 Hz servo control (see ball_tracker.h).
 ******************************************************************************
 */
#include "main.h"
#include "ball_tracker.h"
#include "servo_control.h"

/* Control runs in the TIM7 ISR. We integrate with MEASURED elapsed time (not a
 * fixed period) so the servo moves the correct amount whatever the actual ISR
 * rate turns out to be (the NPU inference can starve the timer). */
#define TICK_TARGET_HZ   50.0f

/* ---- alpha-beta filter gains (position / velocity correction) ------------ *
 * Control rate == detection rate (~12 Hz, inference-bound), so prediction buys
 * nothing and adds noise -> BETA=0 disables velocity, leaving a clean smoothed
 * proportional controller (robust, like the original) plus the lifecycle. */
#define TRK_ALPHA        0.25f
#define TRK_BETA         0.0f

/* ---- control law (pan rate from position error + velocity feedforward) --- */
#define TRK_SIGN         (-1.0f)   /* pan direction toward ball (verified)      */
#define TRK_KP           60.0f     /* deg/s per unit of error                   */
#define TRK_KV           0.0f      /* velocity feedforward off (tune later)     */
#define TRK_OMEGA_MAX    95.0f     /* max pan rate (matches the faster SIMPLE path) */
#define TRK_DEADZONE     0.05f     /* |x-0.5| below this = centered, no pan      */
#define TRK_COAST_TAU    0.5f      /* coast pan-rate decay time constant (s)     */

/* ---- lifecycle ----------------------------------------------------------- */
#define TRK_CONFIRM      3          /* consecutive hits to confirm a track       */
#define TRK_COAST_START  0.20f      /* s without a hit -> start coasting         */
#define TRK_COAST_MAX    0.50f      /* s coasting -> declare lost, hold          */
#define TRK_COAST_DECAY  0.96f      /* per-tick pan-rate decay while coasting     */

/* Tracker state -- written ONLY by BallTracker_ControlTick (main-loop context). */
static volatile float    g_x     = 0.5f;   /* filtered horizontal position */
static volatile float    g_v     = 0.0f;   /* velocity (units/s)           */
static volatile int      g_state = TRK_IDLE;
static float    g_omega = 0.0f;            /* current pan rate (deg/s)     */
static float    g_theta = 90.0f;           /* servo angle the tracker owns */
static int      g_hits  = 0;
static float    g_age   = 0.0f;            /* s since last associated hit  */
static uint32_t g_ticks = 0;

/* Mailbox: main loop -> ISR. seq incremented last so the ISR sees a coherent
 * (present, x) once it observes a new seq. */
static volatile float    g_mbx_x       = 0.5f;
static volatile uint8_t  g_mbx_present = 0;
static volatile uint32_t g_mbx_seq     = 0;
static uint32_t          g_last_seq    = 0;

float    BallTracker_GetX(void)      { return g_x; }
int      BallTracker_State(void)     { return g_state; }
uint32_t BallTracker_TickCount(void) { return g_ticks; }

void BallTracker_Measure(int present, float x_center)
{
  g_mbx_x       = x_center;
  g_mbx_present = (uint8_t)(present ? 1 : 0);
  __DMB();
  g_mbx_seq++;
}

/* Apply one detection-frame result to the filter + lifecycle (ISR context). */
static void tracker_apply_measurement(int present, float mx)
{
  if (present) {
    if (g_state == TRK_IDLE) {                 /* seed a tentative track */
      g_x = mx; g_v = 0.0f; g_hits = 1; g_age = 0.0f;
      g_state = TRK_TENTATIVE;
      g_theta = (float)Servo_GetAngle();       /* resync to actual servo angle */
      return;
    }
    /* alpha-beta correction (dt = time since last hit, clamped). */
    float dtm = g_age; if (dtm < 0.005f) dtm = 0.005f; if (dtm > 0.5f) dtm = 0.5f;
    float r = mx - g_x;
    g_x += TRK_ALPHA * r;
    g_v += (TRK_BETA / dtm) * r;
    g_age = 0.0f;
    if (g_hits < TRK_CONFIRM) g_hits++;
    if (g_state == TRK_TENTATIVE && g_hits >= TRK_CONFIRM) g_state = TRK_CONFIRMED;
    if (g_state == TRK_COAST) g_state = TRK_CONFIRMED;   /* re-acquired */
  } else {
    /* miss: a tentative track needs consecutive hits, so drop it; a confirmed
     * track is handled by the coast timeout (age keeps growing below). */
    if (g_state == TRK_TENTATIVE) { g_state = TRK_IDLE; g_hits = 0; g_v = 0.0f; }
  }
}

void BallTracker_ControlTick(void)
{
  g_ticks++;

  /* Measured elapsed time since the last tick (rate-agnostic integration). */
  static uint32_t last_ms = 0;
  uint32_t now = HAL_GetTick();
  float dt = (last_ms == 0) ? (1.0f / TICK_TARGET_HZ) : (float)(now - last_ms) * 0.001f;
  last_ms = now;
  if (dt < 0.001f) dt = 0.001f;
  if (dt > 0.10f)  dt = 0.10f;

  /* Consume a new measurement, if any. */
  if (g_mbx_seq != g_last_seq) {
    g_last_seq = g_mbx_seq;
    tracker_apply_measurement((int)g_mbx_present, g_mbx_x);
  }

  /* Predict the estimate forward and age the track. */
  g_x += g_v * dt;
  if (g_x < 0.0f) g_x = 0.0f;
  if (g_x > 1.0f) g_x = 1.0f;
  g_age += dt;

  /* Lifecycle by elapsed time without a hit. */
  if (g_state == TRK_CONFIRMED && g_age > TRK_COAST_START) g_state = TRK_COAST;
  if (g_state == TRK_COAST && g_age > TRK_COAST_MAX) {
    g_state = TRK_IDLE; g_omega = 0.0f; g_v = 0.0f; g_hits = 0;
  }

  /* Control law -> pan rate -> integrate into servo angle. */
  if (g_state == TRK_CONFIRMED) {
    float err = g_x - 0.5f;
    if (err < TRK_DEADZONE && err > -TRK_DEADZONE) err = 0.0f;
    g_omega = TRK_SIGN * (TRK_KP * err + TRK_KV * g_v);
    if (g_omega >  TRK_OMEGA_MAX) g_omega =  TRK_OMEGA_MAX;
    if (g_omega < -TRK_OMEGA_MAX) g_omega = -TRK_OMEGA_MAX;
  } else if (g_state == TRK_COAST) {
    g_omega -= g_omega * (dt / TRK_COAST_TAU);   /* time-based decay of pan rate */
  } else {
    g_omega = 0.0f;                              /* IDLE/TENTATIVE: hold (manual ok) */
  }

  if (g_state == TRK_CONFIRMED || g_state == TRK_COAST) {
    g_theta += g_omega * dt;
    if (g_theta < 0.0f)   g_theta = 0.0f;
    if (g_theta > 180.0f) g_theta = 180.0f;
    Servo_SetAngleF(g_theta);
  }
}

void BallTracker_Init(void)
{
  /* Control runs from the main loop (BallTracker_ControlTick), not a timer ISR:
   * the NPU inference masks interrupts globally for ~72 ms, so a high-rate tick
   * is impossible during inference. Main-loop pacing (~12 Hz, the real ceiling)
   * with measured-dt integration is deterministic and simpler. */
  g_state = TRK_IDLE;
  g_theta = (float)Servo_GetAngle();
  g_x = 0.5f; g_v = 0.0f; g_omega = 0.0f; g_hits = 0; g_age = 0.0f;
}
