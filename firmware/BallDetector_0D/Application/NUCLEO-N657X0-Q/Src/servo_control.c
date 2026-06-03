/**
 ******************************************************************************
 * @file    servo_control.c
 * @brief   Camera-pan servo on PA3 (Arduino D10) = TIM16_CH1, 50 Hz PWM.
 *          See servo_control.h for wiring and the open-loop position notes.
 ******************************************************************************
 */
#include "main.h"
#include "servo_control.h"
#include <stdio.h>

#define SERVO_STEP_DEG   5         /* manual L/R step size (gentle for calibration) */
#define SERVO_FINE_US    20        /* fine '<' / '>' nudge for finding center */

static TIM_HandleTypeDef htim16;
static int      g_servo_deg = 90;            /* last commanded angle (tracked)  */
static uint16_t g_servo_us  = SERVO_CENTER_US;

int      Servo_GetAngle(void)  { return g_servo_deg; }
uint16_t Servo_GetMicros(void) { return g_servo_us; }

void Servo_SetMicros(uint16_t us)
{
  if (us < SERVO_MIN_US) us = SERVO_MIN_US;   /* never command past the safe band */
  if (us > SERVO_MAX_US) us = SERVO_MAX_US;
  g_servo_us  = us;
  g_servo_deg = (int)(((int)us - (int)SERVO_MIN_US) * 180 /
                      (int)(SERVO_MAX_US - SERVO_MIN_US));   /* keep deg in sync */
  __HAL_TIM_SET_COMPARE(&htim16, TIM_CHANNEL_1, us);
}

void Servo_SetAngle(int deg)
{
  if (deg < 0)   deg = 0;
  if (deg > 180) deg = 180;
  g_servo_deg = deg;
  /* Map 0..180 deg linearly across the conservative pulse band. */
  uint16_t us = (uint16_t)(SERVO_MIN_US +
                ((uint32_t)deg * (SERVO_MAX_US - SERVO_MIN_US)) / 180u);
  Servo_SetMicros(us);
}

/* Float angle (used by the 50 Hz tracker for smooth sub-degree motion). */
void Servo_SetAngleF(float deg)
{
  if (deg < 0.0f)   deg = 0.0f;
  if (deg > 180.0f) deg = 180.0f;
  float us = (float)SERVO_MIN_US +
             deg * (float)(SERVO_MAX_US - SERVO_MIN_US) / 180.0f;
  Servo_SetMicros((uint16_t)(us + 0.5f));
}

void Servo_Command(char cmd)
{
  switch (cmd) {
    /* L/R swapped vs first build so they match the physical direction. */
    case 'L': case 'l': Servo_SetAngle(g_servo_deg + SERVO_STEP_DEG); break;
    case 'R': case 'r': Servo_SetAngle(g_servo_deg - SERVO_STEP_DEG); break;
    case 'M': case 'm': Servo_SetAngle(90); break;            /* center / home  */
    /* Fine nudges in microseconds for finding the exact center/extremes. */
    case '<':           Servo_SetMicros(g_servo_us - SERVO_FINE_US); break;
    case '>':           Servo_SetMicros(g_servo_us + SERVO_FINE_US); break;
    default: break;
  }
}

void Servo_Init(void)
{
  GPIO_InitTypeDef gpio = {0};

  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_TIM16_CLK_ENABLE();

  /* PA3 -> TIM16_CH1 alternate function. */
  gpio.Mode      = GPIO_MODE_AF_PP;
  gpio.Pull      = GPIO_NOPULL;
  gpio.Speed     = GPIO_SPEED_FREQ_LOW;     /* 50 Hz signal, no need for high speed */
  gpio.Alternate = GPIO_AF1_TIM16;
  gpio.Pin       = GPIO_PIN_3;
  HAL_GPIO_Init(GPIOA, &gpio);

  /* 50 Hz / 20 ms period with a 1 us tick so the compare value IS the pulse
   * width in microseconds. On this STM32N6 the APB2 timer kernel clock is
   * 2 x PCLK2 (HAL_RCC_GetPCLK2Freq reports the lower number); using PCLK2
   * directly gave 0.5 us ticks -> angles compressed ~2-3x. Hence the x2. */
  uint32_t pclk2  = HAL_RCC_GetPCLK2Freq();
  uint32_t timclk = 2u * pclk2;
  uint32_t psc    = (timclk / 1000000u);    /* ticks per us */
  if (psc == 0u) psc = 1u;

  htim16.Instance               = TIM16;
  htim16.Init.Prescaler         = psc - 1u;          /* 1 us per tick      */
  htim16.Init.CounterMode       = TIM_COUNTERMODE_UP;
  htim16.Init.Period            = 20000u - 1u;       /* 20 ms -> 50 Hz     */
  htim16.Init.ClockDivision     = TIM_CLOCKDIVISION_DIV1;
  htim16.Init.RepetitionCounter = 0;
  htim16.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_PWM_Init(&htim16) != HAL_OK) { while (1); }

  TIM_OC_InitTypeDef oc = {0};
  oc.OCMode       = TIM_OCMODE_PWM1;
  oc.Pulse        = SERVO_CENTER_US;        /* start at center */
  oc.OCPolarity   = TIM_OCPOLARITY_HIGH;
  oc.OCNPolarity  = TIM_OCNPOLARITY_HIGH;
  oc.OCFastMode   = TIM_OCFAST_DISABLE;
  oc.OCIdleState  = TIM_OCIDLESTATE_RESET;
  oc.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_PWM_ConfigChannel(&htim16, &oc, TIM_CHANNEL_1) != HAL_OK) { while (1); }

  /* TIM16 is break-capable: ensure the main output is enabled. */
  TIM_BreakDeadTimeConfigTypeDef bdt = {0};
  bdt.OffStateRunMode  = TIM_OSSR_DISABLE;
  bdt.OffStateIDLEMode = TIM_OSSI_DISABLE;
  bdt.LockLevel        = TIM_LOCKLEVEL_OFF;
  bdt.DeadTime         = 0;
  bdt.BreakState       = TIM_BREAK_DISABLE;
  bdt.BreakPolarity    = TIM_BREAKPOLARITY_HIGH;
  bdt.AutomaticOutput  = TIM_AUTOMATICOUTPUT_ENABLE;
  HAL_TIMEx_ConfigBreakDeadTime(&htim16, &bdt);

  if (HAL_TIM_PWM_Start(&htim16, TIM_CHANNEL_1) != HAL_OK) { while (1); }

  /* Define the nominal/home: command center (we can't read the servo back). */
  Servo_SetMicros(SERVO_CENTER_US);
  g_servo_deg = 90;
  uint32_t period_ms = (20000u * (psc)) / (timclk / 1000u);  /* sanity: should read 20 */
  printf("Servo: PCLK2=%lu timclk=%lu psc=%lu period~%lums(=>50Hz target) home=%uus(%ddeg)\n",
         (unsigned long)pclk2, (unsigned long)timclk, (unsigned long)(psc - 1u),
         (unsigned long)period_ms, (unsigned)g_servo_us, g_servo_deg);
}
