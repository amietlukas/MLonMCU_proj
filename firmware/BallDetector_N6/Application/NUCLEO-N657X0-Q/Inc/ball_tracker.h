/**
 ******************************************************************************
 * @file    ball_tracker.h
 * @brief   Lightweight 1-D single-ball tracker (tracking-by-detection / SORT-lite)
 *          driving the camera-pan servo, decoupled from the slow inference rate.
 *
 * Two rates:
 *   - Main loop (~detection rate, ~7-13 Hz): selects the best ball detection and
 *     deposits it as a measurement via BallTracker_Measure().
 *   - TIM7 ISR (50 Hz): BallTracker_ControlTick() runs the alpha-beta filter
 *     (position x + velocity v), the track lifecycle, and the servo control law.
 *
 * Lifecycle: IDLE -> TENTATIVE (needs N consecutive confident hits) -> CONFIRMED
 *            -> COAST (predict through brief dropouts) -> IDLE (hold on timeout).
 ******************************************************************************
 */
#ifndef BALL_TRACKER_H
#define BALL_TRACKER_H

#include <stdint.h>

/* Detector confidence required to feed a box to the tracker (display may draw
 * weaker ones). Association gate: while tracking, only accept detections within
 * TRK_GATE (normalized) of the current estimate -> rejects far false positives. */
#ifndef TRACK_CONF_MIN
#define TRACK_CONF_MIN  0.60f
#endif
#define TRK_GATE        0.25f

enum { TRK_IDLE = 0, TRK_TENTATIVE = 1, TRK_CONFIRMED = 2, TRK_COAST = 3 };

/* Set up state + the 50 Hz control timer (TIM7). Call after Servo_Init(). */
void BallTracker_Init(void);

/* Deposit one detection-frame result (main loop). present!=0 means a confident,
 * gated ball at normalized horizontal x_center; present==0 means none this frame. */
void BallTracker_Measure(int present, float x_center);

/* Introspection for the main-loop association gate + debug. */
float    BallTracker_GetX(void);     /* current filtered estimate [0,1] */
int      BallTracker_State(void);    /* TRK_IDLE..TRK_COAST             */
uint32_t BallTracker_TickCount(void);/* 50 Hz ticks, for rate check     */

/* Called from TIM7_IRQHandler at 50 Hz (sole owner of tracker state). */
void BallTracker_ControlTick(void);

#endif /* BALL_TRACKER_H */
