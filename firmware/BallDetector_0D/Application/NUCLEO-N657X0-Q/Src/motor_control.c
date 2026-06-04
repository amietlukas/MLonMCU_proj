/**
 ******************************************************************************
 * @file    motor_control.c
 * @brief   Differential-drive motor control ported from paul_car.ino.
 *          See motor_control.h for the Arduino-header -> STM32 pin mapping.
 ******************************************************************************
 */
#include "main.h"
#include "motor_control.h"

/* ---- Direction GPIOs (HIGH/LOW like Arduino digitalWrite) ---------------- */
#define LB_PORT  GPIOD
#define LB_PIN   GPIO_PIN_0      /* D2  pinLB */
#define LF_PORT  GPIOE
#define LF_PIN   GPIO_PIN_0      /* D4  pinLF */
#define RB_PORT  GPIOE
#define RB_PIN   GPIO_PIN_11     /* D7  pinRB */
#define RF_PORT  GPIOD
#define RF_PIN   GPIO_PIN_12     /* D8  pinRF */

/* ---- PWM speed outputs on TIM1 complementary channels -------------------- */
#define LPWM_CH  TIM_CHANNEL_4   /* D6 PD5  -> TIM1_CH4N (left  speed) */
#define RPWM_CH  TIM_CHANNEL_2   /* D5 PE10 -> TIM1_CH2N (right speed) */

/* PWM: ARR=255 so a 0..255 command maps directly to duty (like analogWrite).
 * Prescaler targets ~1-2 kHz (safe for common H-bridge driver shields); tune
 * MOTOR_PWM_PSC if the actual TIM1 kernel clock makes the motors whine/stall. */
#define MOTOR_PWM_ARR   255u
#define MOTOR_PWM_PSC   780u

static TIM_HandleTypeDef htim1;

static int clamp_speed(int v)
{
  if (v < 0)   return 0;
  if (v > 255) return 255;
  return v;
}

void Motor_SetSpeed(int leftSpeed, int rightSpeed)
{
  __HAL_TIM_SET_COMPARE(&htim1, LPWM_CH, (uint32_t)clamp_speed(leftSpeed));
  __HAL_TIM_SET_COMPARE(&htim1, RPWM_CH, (uint32_t)clamp_speed(rightSpeed));
}

void Motor_Stop(void)
{
  /* Brake: all direction pins HIGH, speed 0 (matches paul_car stopCar). */
  HAL_GPIO_WritePin(LB_PORT, LB_PIN, GPIO_PIN_SET);
  HAL_GPIO_WritePin(LF_PORT, LF_PIN, GPIO_PIN_SET);
  HAL_GPIO_WritePin(RB_PORT, RB_PIN, GPIO_PIN_SET);
  HAL_GPIO_WritePin(RF_PORT, RF_PIN, GPIO_PIN_SET);
  Motor_SetSpeed(0, 0);
}

void Motor_ForwardStraight(int speed)
{
  HAL_GPIO_WritePin(LB_PORT, LB_PIN, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(LF_PORT, LF_PIN, GPIO_PIN_SET);
  HAL_GPIO_WritePin(RB_PORT, RB_PIN, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(RF_PORT, RF_PIN, GPIO_PIN_SET);
  Motor_SetSpeed(speed, speed);
}

void Motor_BackwardStraight(int speed)
{
  HAL_GPIO_WritePin(LB_PORT, LB_PIN, GPIO_PIN_SET);
  HAL_GPIO_WritePin(LF_PORT, LF_PIN, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(RB_PORT, RB_PIN, GPIO_PIN_SET);
  HAL_GPIO_WritePin(RF_PORT, RF_PIN, GPIO_PIN_RESET);
  Motor_SetSpeed(speed, speed);
}

void Motor_Curve(int leftSpeed, int rightSpeed)
{
  /* Both sides forward, different speeds -> curve toward the slower side. */
  HAL_GPIO_WritePin(LB_PORT, LB_PIN, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(LF_PORT, LF_PIN, GPIO_PIN_SET);
  HAL_GPIO_WritePin(RB_PORT, RB_PIN, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(RF_PORT, RF_PIN, GPIO_PIN_SET);
  Motor_SetSpeed(leftSpeed, rightSpeed);
}

void Motor_Command(char cmd)
{
  switch (cmd) {
    case '0': Motor_Stop();              break;  /* stop          */
    case '1': Motor_ForwardStraight(110);break;  /* forward       */
    case '2': Motor_Curve(110, 55);      break;  /* forward-right */
    case '3': Motor_Curve(55, 110);      break;  /* forward-left  */
    case '4': Motor_BackwardStraight(100);break; /* backward      */
    case '5': Motor_Stop();              break;  /* (reserved)    */
    default:  /* keep previous command */ break;
  }
}

void Motor_Init(void)
{
  GPIO_InitTypeDef gpio = {0};

  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_TIM1_CLK_ENABLE();

  /* Direction pins: push-pull outputs, start low. */
  gpio.Mode  = GPIO_MODE_OUTPUT_PP;
  gpio.Pull  = GPIO_NOPULL;
  gpio.Speed = GPIO_SPEED_FREQ_LOW;
  gpio.Pin   = LB_PIN | RF_PIN;                 /* PD0, PD12 */
  HAL_GPIO_Init(GPIOD, &gpio);
  gpio.Pin   = LF_PIN | RB_PIN;                 /* PE0, PE11 */
  HAL_GPIO_Init(GPIOE, &gpio);

  /* PWM pins as TIM1 alternate function. */
  gpio.Mode      = GPIO_MODE_AF_PP;
  gpio.Pull      = GPIO_NOPULL;
  gpio.Speed     = GPIO_SPEED_FREQ_HIGH;
  gpio.Alternate = GPIO_AF1_TIM1;
  gpio.Pin       = GPIO_PIN_5;                   /* PD5  -> TIM1_CH4N */
  HAL_GPIO_Init(GPIOD, &gpio);
  gpio.Pin       = GPIO_PIN_10;                  /* PE10 -> TIM1_CH2N */
  HAL_GPIO_Init(GPIOE, &gpio);

  /* TIM1 base + PWM. */
  htim1.Instance               = TIM1;
  htim1.Init.Prescaler         = MOTOR_PWM_PSC;
  htim1.Init.CounterMode       = TIM_COUNTERMODE_UP;
  htim1.Init.Period            = MOTOR_PWM_ARR;
  htim1.Init.ClockDivision     = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_PWM_Init(&htim1) != HAL_OK) { while (1); }

  TIM_OC_InitTypeDef oc = {0};
  oc.OCMode       = TIM_OCMODE_PWM1;
  oc.Pulse        = 0;
  oc.OCPolarity   = TIM_OCPOLARITY_HIGH;
  oc.OCNPolarity  = TIM_OCNPOLARITY_HIGH;
  oc.OCFastMode   = TIM_OCFAST_DISABLE;
  oc.OCIdleState  = TIM_OCIDLESTATE_RESET;
  oc.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &oc, LPWM_CH) != HAL_OK) { while (1); }
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &oc, RPWM_CH) != HAL_OK) { while (1); }

  TIM_BreakDeadTimeConfigTypeDef bdt = {0};
  bdt.OffStateRunMode  = TIM_OSSR_DISABLE;
  bdt.OffStateIDLEMode = TIM_OSSI_DISABLE;
  bdt.LockLevel        = TIM_LOCKLEVEL_OFF;
  bdt.DeadTime         = 0;
  bdt.BreakState       = TIM_BREAK_DISABLE;
  bdt.BreakPolarity    = TIM_BREAKPOLARITY_HIGH;
  bdt.AutomaticOutput  = TIM_AUTOMATICOUTPUT_ENABLE;
  HAL_TIMEx_ConfigBreakDeadTime(&htim1, &bdt);

  /* Drive the complementary (N) outputs that the shield's PWM pins land on. */
  HAL_TIMEx_PWMN_Start(&htim1, LPWM_CH);
  HAL_TIMEx_PWMN_Start(&htim1, RPWM_CH);

  Motor_Stop();   /* safe state: car stopped until first command */
}
