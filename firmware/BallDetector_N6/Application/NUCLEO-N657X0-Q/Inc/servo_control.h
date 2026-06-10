/**
 ******************************************************************************
 * @file    servo_control.h
 * @brief   Hobby servo (SG90-class) control for the camera pan, on the Arduino
 *          sensor-shield GVS header D10 -> PA3 -> TIM16_CH1, 50 Hz PWM.
 *
 * Servo wiring (standard color code): brown=GND, red=+5V (middle), orange=signal.
 *   GVS header: G<-brown, V<-red, S<-orange.
 *
 * Position control is open-loop/absolute: pulse width sets the angle. There is
 * NO position readback on a 3-wire servo (internal pot is not exposed), so the
 * "nominal" position is DEFINED by commanding center at init, and the current
 * angle is whatever we last commanded (tracked in software).
 *
 * Conservative pulse band keeps the servo off its mechanical end-stops.
 ******************************************************************************
 */
#ifndef SERVO_CONTROL_H
#define SERVO_CONTROL_H

#include <stdint.h>

/* This unit is a 0.5-2.5 ms servo (verified on the bench: 500 us = a clean 0 deg,
 * 1500 us = 90 deg, and the mapping is linear ~11 us/deg). 1000-2000 us only gave
 * the middle 45-135 deg, so we use the full band for true 0-180 deg.
 * 500 us (0 deg) confirmed safe; check 2500 us (180 deg) doesn't strain on first use. */
#define SERVO_MIN_US     500u     /* 0 deg   */
#define SERVO_CENTER_US  1500u    /* 90 deg / nominal home */
#define SERVO_MAX_US     2500u    /* 180 deg */

/* Configure TIM16 PWM on PA3 and drive the servo to center (the nominal home).
 * NOTE: the servo physically moves to center on boot. */
void Servo_Init(void);

void     Servo_SetMicros(uint16_t us);  /* clamped to [SERVO_MIN_US, SERVO_MAX_US] */
void     Servo_SetAngle(int deg);       /* 0..180 mapped across the safe band      */
void     Servo_SetAngleF(float deg);    /* float angle (smooth 50 Hz tracker)      */
int      Servo_GetAngle(void);          /* last COMMANDED angle (no HW readback)   */
uint16_t Servo_GetMicros(void);
void     Servo_DumpState(void);         /* print TIM16/PA3 state for PWM debugging */

/* Manual test protocol: 'L'/'l' step left, 'R'/'r' step right, 'M'/'m' center. */
void Servo_Command(char cmd);

/* Ball tracking now lives in ball_tracker.{c,h} (alpha-beta + 50 Hz control). */

#endif /* SERVO_CONTROL_H */
