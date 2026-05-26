/**
  ******************************************************************************
  * @file    small_net_fp32_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-05-20T03:17:49+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef SMALL_NET_FP32_DATA_PARAMS_H
#define SMALL_NET_FP32_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_SMALL_NET_FP32_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_small_net_fp32_data_weights_params[1]))
*/

#define AI_SMALL_NET_FP32_DATA_CONFIG               (NULL)


#define AI_SMALL_NET_FP32_DATA_ACTIVATIONS_SIZES \
  { 353252, }
#define AI_SMALL_NET_FP32_DATA_ACTIVATIONS_SIZE     (353252)
#define AI_SMALL_NET_FP32_DATA_ACTIVATIONS_COUNT    (1)
#define AI_SMALL_NET_FP32_DATA_ACTIVATION_1_SIZE    (353252)



#define AI_SMALL_NET_FP32_DATA_WEIGHTS_SIZES \
  { 760728, }
#define AI_SMALL_NET_FP32_DATA_WEIGHTS_SIZE         (760728)
#define AI_SMALL_NET_FP32_DATA_WEIGHTS_COUNT        (1)
#define AI_SMALL_NET_FP32_DATA_WEIGHT_1_SIZE        (760728)



#define AI_SMALL_NET_FP32_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_small_net_fp32_activations_table[1])

extern ai_handle g_small_net_fp32_activations_table[1 + 2];



#define AI_SMALL_NET_FP32_DATA_WEIGHTS_TABLE_GET() \
  (&g_small_net_fp32_weights_table[1])

extern ai_handle g_small_net_fp32_weights_table[1 + 2];


#endif    /* SMALL_NET_FP32_DATA_PARAMS_H */
