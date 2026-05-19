/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    stm32u5xx_it.c
  * @brief   Interrupt Service Routines.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "stm32u5xx_it.h"
#include "usart.h"
#include <string.h>
/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN TD */

/* USER CODE END TD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/* External variables --------------------------------------------------------*/
extern DMA_HandleTypeDef handle_GPDMA1_Channel12;
/* USER CODE BEGIN EV */

/* USER CODE END EV */

/******************************************************************************/
/*           Cortex Processor Interruption and Exception Handlers          */
/******************************************************************************/
/**
  * @brief This function handles Non maskable interrupt.
  */
void NMI_Handler(void)
{
  /* USER CODE BEGIN NonMaskableInt_IRQn 0 */

  /* USER CODE END NonMaskableInt_IRQn 0 */
  /* USER CODE BEGIN NonMaskableInt_IRQn 1 */
   while (1)
  {
  }
  /* USER CODE END NonMaskableInt_IRQn 1 */
}

/**
  * @brief This function handles Hard fault interrupt.
  */
void HardFault_Handler(void)
{
  /* USER CODE BEGIN HardFault_IRQn 0 */
  /* Stack-frame at fault: [r0,r1,r2,r3,r12,LR,PC,xPSR] */
  uint32_t *stack;
  __asm volatile (
      "tst lr, #4 \n"
      "ite eq \n"
      "mrseq %0, msp \n"
      "mrsne %0, psp \n"
      : "=r" (stack));
  const char *labels[3] = {"PC=", " LR=", " CFSR="};
  uint32_t vals[3] = {stack[6], stack[5], *(volatile uint32_t*)0xE000ED28};
  HAL_UART_Transmit(&huart1, (uint8_t*)"*** HARDFAULT ***", 17, HAL_MAX_DELAY);
  for (int v = 0; v < 3; v++) {
    HAL_UART_Transmit(&huart1, (uint8_t*)labels[v], strlen(labels[v]), HAL_MAX_DELAY);
    char hex[10]; for (int i = 7; i >= 0; i--) {
      int n = (vals[v] >> (i*4)) & 0xF; hex[7-i] = (n < 10) ? ('0'+n) : ('a'+n-10);
    }
    hex[8] = 0;
    HAL_UART_Transmit(&huart1, (uint8_t*)hex, 8, HAL_MAX_DELAY);
  }
  HAL_UART_Transmit(&huart1, (uint8_t*)"\r\n", 2, HAL_MAX_DELAY);
  /* USER CODE END HardFault_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_HardFault_IRQn 0 */
    /* USER CODE END W1_HardFault_IRQn 0 */
  }
}

/**
  * @brief This function handles Memory management fault.
  */
void MemManage_Handler(void)
{
  /* USER CODE BEGIN MemoryManagement_IRQn 0 */

  /* USER CODE END MemoryManagement_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_MemoryManagement_IRQn 0 */
    /* USER CODE END W1_MemoryManagement_IRQn 0 */
  }
}

/**
  * @brief This function handles Prefetch fault, memory access fault.
  */
void BusFault_Handler(void)
{
  /* USER CODE BEGIN BusFault_IRQn 0 */

  /* USER CODE END BusFault_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_BusFault_IRQn 0 */
    /* USER CODE END W1_BusFault_IRQn 0 */
  }
}

/**
  * @brief This function handles Undefined instruction or illegal state.
  */
void UsageFault_Handler(void)
{
  /* USER CODE BEGIN UsageFault_IRQn 0 */

  /* USER CODE END UsageFault_IRQn 0 */
  while (1)
  {
    /* USER CODE BEGIN W1_UsageFault_IRQn 0 */
    /* USER CODE END W1_UsageFault_IRQn 0 */
  }
}

/**
  * @brief This function handles System service call via SWI instruction.
  */
void SVC_Handler(void)
{
  /* USER CODE BEGIN SVCall_IRQn 0 */

  /* USER CODE END SVCall_IRQn 0 */
  /* USER CODE BEGIN SVCall_IRQn 1 */

  /* USER CODE END SVCall_IRQn 1 */
}

/**
  * @brief This function handles Debug monitor.
  */
void DebugMon_Handler(void)
{
  /* USER CODE BEGIN DebugMonitor_IRQn 0 */

  /* USER CODE END DebugMonitor_IRQn 0 */
  /* USER CODE BEGIN DebugMonitor_IRQn 1 */

  /* USER CODE END DebugMonitor_IRQn 1 */
}

/**
  * @brief This function handles Pendable request for system service.
  */
void PendSV_Handler(void)
{
  /* USER CODE BEGIN PendSV_IRQn 0 */

  /* USER CODE END PendSV_IRQn 0 */
  /* USER CODE BEGIN PendSV_IRQn 1 */

  /* USER CODE END PendSV_IRQn 1 */
}

/**
  * @brief This function handles System tick timer.
  */
void SysTick_Handler(void)
{
  /* USER CODE BEGIN SysTick_IRQn 0 */

  /* USER CODE END SysTick_IRQn 0 */
  HAL_IncTick();
  /* USER CODE BEGIN SysTick_IRQn 1 */

  /* USER CODE END SysTick_IRQn 1 */
}

/******************************************************************************/
/* STM32U5xx Peripheral Interrupt Handlers                                    */
/* Add here the Interrupt Handlers for the used peripherals.                  */
/* For the available peripheral interrupt handler names,                      */
/* please refer to the startup file (startup_stm32u5xx.s).                    */
/******************************************************************************/

/**
  * @brief This function handles EXTI Line14 interrupt.
  */
void EXTI14_IRQHandler(void)
{
  /* USER CODE BEGIN EXTI14_IRQn 0 */

  /* USER CODE END EXTI14_IRQn 0 */
  HAL_GPIO_EXTI_IRQHandler(MXCHIP_NOTIFY_Pin);
  /* USER CODE BEGIN EXTI14_IRQn 1 */

  /* USER CODE END EXTI14_IRQn 1 */
}

void EXTI15_IRQHandler(void)
{
  HAL_GPIO_EXTI_IRQHandler(MXCHIP_FLOW_Pin);
}

/**
  * @brief This function handles GPDMA1 Channel 12 global interrupt.
  */
void GPDMA1_Channel12_IRQHandler(void)
{
  /* USER CODE BEGIN GPDMA1_Channel12_IRQn 0 */
  /* BSP_CAMERA wires DCMI -> GPDMA1_Channel12 via its own DMA handle
   * (hdma_handler in b_u585i_iot02a_camera.c), so that's what the IRQ
   * must service — not the CubeMX handle, which BSP overwrote at init.
   * If you ever regenerate the .ioc, restore this redirect. */
  extern DMA_HandleTypeDef hdma_handler;
  HAL_DMA_IRQHandler(&hdma_handler);
  return;
  (void)handle_GPDMA1_Channel12;  /* keep CubeMX symbol referenced */
  /* USER CODE END GPDMA1_Channel12_IRQn 0 */
  HAL_DMA_IRQHandler(&handle_GPDMA1_Channel12);
  /* USER CODE BEGIN GPDMA1_Channel12_IRQn 1 */

  /* USER CODE END GPDMA1_Channel12_IRQn 1 */
}

/* USER CODE BEGIN 1 */

/* DCMI/PSSI global IRQ. Not generated by CubeMX because the NVIC entry
 * for DCMI_PSSI_IRQn isn't checked in the .ioc — but BSP_CAMERA_Init
 * unmasks it programmatically (HAL_NVIC_EnableIRQ in MspInit), so the
 * vector still needs a real handler or it traps to Default_Handler. */
extern DCMI_HandleTypeDef hcamera_dcmi;
void DCMI_PSSI_IRQHandler(void)
{
  HAL_DCMI_IRQHandler(&hcamera_dcmi);
}

/* USER CODE END 1 */
