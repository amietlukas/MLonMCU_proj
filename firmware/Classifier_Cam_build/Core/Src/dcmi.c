/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    dcmi.c
  * @brief   This file provides code for the configuration
  *          of the DCMI instances.
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
#include "dcmi.h"

/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

DCMI_HandleTypeDef hdcmi;
DMA_HandleTypeDef handle_GPDMA1_Channel12;

/* DCMI init function */
void MX_DCMI_Init(void)
{

  /* USER CODE BEGIN DCMI_Init 0 */

  /* USER CODE END DCMI_Init 0 */

  /* USER CODE BEGIN DCMI_Init 1 */

  /* USER CODE END DCMI_Init 1 */
  hdcmi.Instance = DCMI;
  hdcmi.Init.SynchroMode = DCMI_SYNCHRO_HARDWARE;
  hdcmi.Init.PCKPolarity = DCMI_PCKPOLARITY_FALLING;
  hdcmi.Init.VSPolarity = DCMI_VSPOLARITY_LOW;
  hdcmi.Init.HSPolarity = DCMI_HSPOLARITY_LOW;
  hdcmi.Init.CaptureRate = DCMI_CR_ALL_FRAME;
  hdcmi.Init.ExtendedDataMode = DCMI_EXTEND_DATA_8B;
  hdcmi.Init.JPEGMode = DCMI_JPEG_DISABLE;
  hdcmi.Init.ByteSelectMode = DCMI_BSM_ALL;
  hdcmi.Init.ByteSelectStart = DCMI_OEBS_ODD;
  hdcmi.Init.LineSelectMode = DCMI_LSM_ALL;
  hdcmi.Init.LineSelectStart = DCMI_OELS_ODD;
  if (HAL_DCMI_Init(&hdcmi) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN DCMI_Init 2 */

  /* USER CODE END DCMI_Init 2 */

}

void HAL_DCMI_MspInit(DCMI_HandleTypeDef* dcmiHandle)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(dcmiHandle->Instance==DCMI)
  {
  /* USER CODE BEGIN DCMI_MspInit 0 */
    /* This MspInit is invoked twice during boot:
     *   1. by MX_DCMI_Init() on CubeMX's hdcmi handle  (we don't use hdcmi)
     *   2. by BSP_MX_DCMI_Init() on hcamera_dcmi      (BSP's real handle)
     *
     * BSP's own DCMI_MspInit (b_u585i_iot02a_camera.c) is called between
     * #1 and #2 and sets up:
     *   - the correct CN2-mapped GPIOs (PA6/PB7/PH8/PH14/PC6-8/PE1/PI4/PI6/PI7)
     *   - a circular linked-list DMA queue on GPDMA1_Channel12 (hdma_handler)
     *   - NVIC for DCMI_PSSI and GPDMA1_Channel12
     *
     * If we configure GPIOs or DMA here, we either light up the WRONG
     * pins (CubeMX auto-routes by chip pin matrix, not board schematic),
     * or — worse — when call #2 runs we clobber BSP's linked-list DMA
     * setup with a single-shot normal-mode DMA, breaking continuous
     * capture and producing one DCMI overrun per frame.
     *
     * So: enable the DCMI peripheral clock and return. BSP owns the rest. */
    __HAL_RCC_DCMI_PSSI_CLK_ENABLE();
    return;
  /* USER CODE END DCMI_MspInit 0 */
    /* DCMI clock enable */
    __HAL_RCC_DCMI_PSSI_CLK_ENABLE();

    __HAL_RCC_GPIOI_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOH_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();
    /**DCMI GPIO Configuration
    PI6     ------> DCMI_D6
    PC11     ------> DCMI_D4
    PI5     ------> DCMI_VSYNC
    PB6     ------> DCMI_D5
    PH12     ------> DCMI_D3
    PI7     ------> DCMI_D7
    PH10     ------> DCMI_D1
    PE0     ------> DCMI_D2
    PH8     ------> DCMI_HSYNC
    PH9     ------> DCMI_D0
    PH5     ------> DCMI_PIXCLK
    */
    GPIO_InitStruct.Pin = GPIO_PIN_6|GPIO_PIN_5|GPIO_PIN_7;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF10_DCMI;
    HAL_GPIO_Init(GPIOI, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = GPIO_PIN_11;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF10_DCMI;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = GPIO_PIN_6;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF10_DCMI;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = GPIO_PIN_12|GPIO_PIN_10|GPIO_PIN_8|GPIO_PIN_9
                          |GPIO_PIN_5;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF10_DCMI;
    HAL_GPIO_Init(GPIOH, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = GPIO_PIN_0;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF10_DCMI;
    HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

    /* DCMI DMA Init */
    /* GPDMA1_REQUEST_DCMI_PSSI Init */
    handle_GPDMA1_Channel12.Instance = GPDMA1_Channel12;
    handle_GPDMA1_Channel12.Init.Request = GPDMA1_REQUEST_DCMI_PSSI;
    handle_GPDMA1_Channel12.Init.BlkHWRequest = DMA_BREQ_SINGLE_BURST;
    handle_GPDMA1_Channel12.Init.Direction = DMA_PERIPH_TO_MEMORY;
    handle_GPDMA1_Channel12.Init.SrcInc = DMA_SINC_INCREMENTED;
    handle_GPDMA1_Channel12.Init.DestInc = DMA_DINC_INCREMENTED;
    handle_GPDMA1_Channel12.Init.SrcDataWidth = DMA_SRC_DATAWIDTH_WORD;
    handle_GPDMA1_Channel12.Init.DestDataWidth = DMA_DEST_DATAWIDTH_WORD;
    handle_GPDMA1_Channel12.Init.Priority = DMA_LOW_PRIORITY_HIGH_WEIGHT;
    handle_GPDMA1_Channel12.Init.SrcBurstLength = 1;
    handle_GPDMA1_Channel12.Init.DestBurstLength = 1;
    handle_GPDMA1_Channel12.Init.TransferAllocatedPort = DMA_SRC_ALLOCATED_PORT0|DMA_DEST_ALLOCATED_PORT0;
    handle_GPDMA1_Channel12.Init.TransferEventMode = DMA_TCEM_BLOCK_TRANSFER;
    handle_GPDMA1_Channel12.Init.Mode = DMA_NORMAL;
    if (HAL_DMA_Init(&handle_GPDMA1_Channel12) != HAL_OK)
    {
      Error_Handler();
    }

    __HAL_LINKDMA(dcmiHandle, DMA_Handle, handle_GPDMA1_Channel12);

    if (HAL_DMA_ConfigChannelAttributes(&handle_GPDMA1_Channel12, DMA_CHANNEL_NPRIV) != HAL_OK)
    {
      Error_Handler();
    }

  /* USER CODE BEGIN DCMI_MspInit 1 */

  /* USER CODE END DCMI_MspInit 1 */
  }
}

void HAL_DCMI_MspDeInit(DCMI_HandleTypeDef* dcmiHandle)
{

  if(dcmiHandle->Instance==DCMI)
  {
  /* USER CODE BEGIN DCMI_MspDeInit 0 */

  /* USER CODE END DCMI_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_DCMI_PSSI_CLK_DISABLE();

    /**DCMI GPIO Configuration
    PI6     ------> DCMI_D6
    PC11     ------> DCMI_D4
    PI5     ------> DCMI_VSYNC
    PB6     ------> DCMI_D5
    PH12     ------> DCMI_D3
    PI7     ------> DCMI_D7
    PH10     ------> DCMI_D1
    PE0     ------> DCMI_D2
    PH8     ------> DCMI_HSYNC
    PH9     ------> DCMI_D0
    PH5     ------> DCMI_PIXCLK
    */
    HAL_GPIO_DeInit(GPIOI, GPIO_PIN_6|GPIO_PIN_5|GPIO_PIN_7);

    HAL_GPIO_DeInit(GPIOC, GPIO_PIN_11);

    HAL_GPIO_DeInit(GPIOB, GPIO_PIN_6);

    HAL_GPIO_DeInit(GPIOH, GPIO_PIN_12|GPIO_PIN_10|GPIO_PIN_8|GPIO_PIN_9
                          |GPIO_PIN_5);

    HAL_GPIO_DeInit(GPIOE, GPIO_PIN_0);

    /* DCMI DMA DeInit */
    HAL_DMA_DeInit(dcmiHandle->DMA_Handle);
  /* USER CODE BEGIN DCMI_MspDeInit 1 */

  /* USER CODE END DCMI_MspDeInit 1 */
  }
}

/* USER CODE BEGIN 1 */

/* USER CODE END 1 */

