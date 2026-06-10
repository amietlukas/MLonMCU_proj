/**
 ******************************************************************************
 * @file    motor_control.h
 * @brief   Differential-drive motor control for the Arduino-shield RC car,
 *          ported from firmware/paul_car/paul_car.ino to the NUCLEO-N657X0-Q.
 *
 * Arduino-header -> STM32 pin mapping (UM3417 Table 12):
 *   D2 pinLB (left back)   -> PD0   (GPIO out)
 *   D4 pinLF (left fwd)    -> PE0   (GPIO out)
 *   D7 pinRB (right back)  -> PE11  (GPIO out)
 *   D8 pinRF (right fwd)   -> PD12  (GPIO out)
 *   D6 Lpwm  (left speed)  -> PD5   (TIM1_CH4N)
 *   D5 Rpwm  (right speed) -> PE10  (TIM1_CH2N)
 *
 * Speed range 0..255 (matches Arduino analogWrite).
 ******************************************************************************
 */
#ifndef MOTOR_CONTROL_H
#define MOTOR_CONTROL_H

/* Configure GPIOs + TIM1 PWM and leave the car stopped. */
void Motor_Init(void);

/* paul_car single-char protocol:
 *   '0' stop, '1' forward, '2' forward-right, '3' forward-left,
 *   '4' backward, '5' stop. Unknown chars are ignored (keep previous). */
void Motor_Command(char cmd);

/* Lower-level primitives (useful for the upcoming ball-tracking step). */
void Motor_SetSpeed(int leftSpeed, int rightSpeed);   /* 0..255 each */
void Motor_Stop(void);
void Motor_ForwardStraight(int speed);
void Motor_BackwardStraight(int speed);
void Motor_Curve(int leftSpeed, int rightSpeed);

#endif /* MOTOR_CONTROL_H */
