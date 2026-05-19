/**
  ******************************************************************************
  * @file    big_net_pruned_int8.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-05-10T22:55:41+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "big_net_pruned_int8.h"
#include "big_net_pruned_int8_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_big_net_pruned_int8
 
#undef AI_BIG_NET_PRUNED_INT8_MODEL_SIGNATURE
#define AI_BIG_NET_PRUNED_INT8_MODEL_SIGNATURE     "0x4fe999df9942b80a7a7bffa279e6ea0b"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-05-10T22:55:41+0200"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_BIG_NET_PRUNED_INT8_N_BATCHES
#define AI_BIG_NET_PRUNED_INT8_N_BATCHES         (1)

static ai_ptr g_big_net_pruned_int8_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_big_net_pruned_int8_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 19200, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  relu_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 24000, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  relu_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12000, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 13440, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5700, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  relu_3_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7106, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  relu_3_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2030, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  relu_4_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3132, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  relu_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 570, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  relu_4_0_0_mean_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 570, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  mean_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 38, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  mean_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 38, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 38, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 6, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  relu_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 45, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  relu_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 5, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  relu_1_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 450, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  relu_1_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 10, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1710, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 19, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  relu_3_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4959, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  relu_3_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 29, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  relu_4_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9918, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  relu_4_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 38, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  mean_Mul_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 38, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  mean_Mul_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 38, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 228, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 6, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  relu_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 196, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  relu_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1600, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  relu_1_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1220, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  relu_1_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1600, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4046, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1520, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  relu_3_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6210, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  relu_3_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1160, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  relu_4_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6696, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  relu_4_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 760, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 68, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(input_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.0f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.22004638612270355f),
    AI_PACK_INTQ_ZP(-39)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 6,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.020901571959257126f, 0.02340463548898697f, 0.020924458280205727f, 0.02051440067589283f, 0.021523844450712204f, 0.024090535938739777f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013121228665113449f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04237150400876999f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_1_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04237150400876999f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_1_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 10,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005836012773215771f, 0.003892249660566449f, 0.007999674417078495f, 0.005698434077203274f, 0.0039920322597026825f, 0.004631010349839926f, 0.003571645123884082f, 0.0062354025430977345f, 0.005376564804464579f, 0.004218680318444967f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_2_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0219194944947958f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_2_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04237150400876999f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_2_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0219194944947958f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_2_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 19,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002314995741471648f, 0.0011161563452333212f, 0.0027928738854825497f, 0.0031014643609523773f, 0.0024991659447550774f, 0.006510020233690739f, 0.0014582826988771558f, 0.0024105622433125973f, 0.003057450521737337f, 0.0036548643838614225f, 0.0017236904241144657f, 0.002495431574061513f, 0.003123129718005657f, 0.003441843669861555f, 0.002052543917670846f, 0.001169270952232182f, 0.004819630645215511f, 0.0020052737090736628f, 0.0024674558080732822f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_3_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01259827334433794f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_3_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0219194944947958f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_3_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01259827334433794f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_3_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 29,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004453886300325394f, 0.003208776004612446f, 0.0018754194024950266f, 0.005111564416438341f, 0.004323557484894991f, 0.003505998756736517f, 0.0024931360967457294f, 1.0f, 0.00429651839658618f, 0.0021091487724334f, 2.2181667702625418e-07f, 0.0025969420094043016f, 0.003920636605471373f, 0.0021732051391154528f, 0.0016428178641945124f, 1.0f, 0.004996056202799082f, 0.0038553422782570124f, 0.008184235543012619f, 0.00259189959615469f, 0.0028481422923505306f, 0.005884979851543903f, 0.0019739363342523575f, 0.00307695590890944f, 0.003916674293577671f, 0.0019990161526948214f, 0.0026530814357101917f, 0.005362675059586763f, 0.004705451894551516f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_4_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09209765493869781f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_4_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01259827334433794f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_4_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09209765493869781f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_4_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 38,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0215976070612669f, 0.023511013016104698f, 0.026097914204001427f, 0.02784409373998642f, 0.019345104694366455f, 0.020550360903143883f, 0.022840656340122223f, 0.018879063427448273f, 0.017298605293035507f, 0.019652046263217926f, 0.019351748749613762f, 0.023267606273293495f, 0.023369336500763893f, 0.022091824561357498f, 0.02022290788590908f, 0.02130640111863613f, 0.01818685606122017f, 0.021803932264447212f, 0.01816968061029911f, 0.019440243020653725f, 0.024023333564400673f, 0.017344100400805473f, 0.01783781498670578f, 0.027321798726916313f, 0.025775384157896042f, 0.017711399123072624f, 0.019241830334067345f, 0.02427227981388569f, 0.02194921486079693f, 0.02249564416706562f, 0.016549957916140556f, 0.018558040261268616f, 0.020141270011663437f, 0.022020922973752022f, 0.02607574872672558f, 0.022175202146172523f, 0.018027156591415405f, 0.015735028311610222f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03757210448384285f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03757210448384285f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 5,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0001489875139668584f, 0.00014317166642285883f, 0.00016114930622279644f, 3.0318093195091933e-05f, 0.00012955440615769476f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  input_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 160, 120), AI_STRIDE_INIT(4, 1, 1, 1, 160),
  1, &input_output_array, &input_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &logits_QuantizeLinear_Input_bias_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6, 6),
  1, &logits_QuantizeLinear_Input_output_array, &logits_QuantizeLinear_Input_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 68, 1, 1), AI_STRIDE_INIT(4, 2, 2, 136, 136),
  1, &logits_QuantizeLinear_Input_scratch0_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 38, 6, 1, 1), AI_STRIDE_INIT(4, 1, 38, 228, 228),
  1, &logits_QuantizeLinear_Input_weights_array, &logits_QuantizeLinear_Input_weights_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 38, 1, 1), AI_STRIDE_INIT(4, 1, 1, 38, 38),
  1, &mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array, &mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  mean_Mul_bias, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 38, 1, 1), AI_STRIDE_INIT(4, 4, 4, 152, 152),
  1, &mean_Mul_bias_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  mean_Mul_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 38, 1, 1), AI_STRIDE_INIT(4, 4, 4, 152, 152),
  1, &mean_Mul_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  mean_Mul_scale, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 38, 1, 1), AI_STRIDE_INIT(4, 4, 4, 152, 152),
  1, &mean_Mul_scale_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  mean_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 38, 1, 1), AI_STRIDE_INIT(4, 4, 4, 152, 152),
  1, &mean_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  relu_1_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 10, 1, 1), AI_STRIDE_INIT(4, 4, 4, 40, 40),
  1, &relu_1_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  relu_1_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 40, 30), AI_STRIDE_INIT(4, 1, 1, 10, 400),
  1, &relu_1_output_array, &relu_1_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  relu_1_scratch0, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 1220, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1220, 1220),
  1, &relu_1_scratch0_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  relu_1_scratch1, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 80, 2), AI_STRIDE_INIT(4, 1, 1, 10, 800),
  1, &relu_1_scratch1_array, &relu_1_scratch1_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  relu_1_weights, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 5, 3, 3, 10), AI_STRIDE_INIT(4, 1, 5, 50, 150),
  1, &relu_1_weights_array, &relu_1_weights_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_bias, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 19, 1, 1), AI_STRIDE_INIT(4, 4, 4, 76, 76),
  1, &relu_2_bias_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_output, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 1, 19, 20, 15), AI_STRIDE_INIT(4, 1, 1, 19, 380),
  1, &relu_2_output_array, &relu_2_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_pad_before_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 42, 32), AI_STRIDE_INIT(4, 1, 1, 10, 420),
  1, &relu_2_pad_before_output_array, &relu_2_pad_before_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_scratch0, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 4046, 1, 1), AI_STRIDE_INIT(4, 1, 1, 4046, 4046),
  1, &relu_2_scratch0_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_scratch1, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 19, 40, 2), AI_STRIDE_INIT(4, 1, 1, 19, 760),
  1, &relu_2_scratch1_array, &relu_2_scratch1_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_weights, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 10, 3, 3, 19), AI_STRIDE_INIT(4, 1, 10, 190, 570),
  1, &relu_2_weights_array, &relu_2_weights_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  relu_3_bias, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 29, 1, 1), AI_STRIDE_INIT(4, 4, 4, 116, 116),
  1, &relu_3_bias_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  relu_3_output, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 1, 29, 10, 7), AI_STRIDE_INIT(4, 1, 1, 29, 290),
  1, &relu_3_output_array, &relu_3_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  relu_3_pad_before_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 19, 22, 17), AI_STRIDE_INIT(4, 1, 1, 19, 418),
  1, &relu_3_pad_before_output_array, &relu_3_pad_before_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  relu_3_scratch0, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 6210, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6210, 6210),
  1, &relu_3_scratch0_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  relu_3_scratch1, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 29, 20, 2), AI_STRIDE_INIT(4, 1, 1, 29, 580),
  1, &relu_3_scratch1_array, &relu_3_scratch1_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  relu_3_weights, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 19, 3, 3, 29), AI_STRIDE_INIT(4, 1, 19, 551, 1653),
  1, &relu_3_weights_array, &relu_3_weights_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  relu_4_0_0_mean_conversion_output, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 38, 5, 3), AI_STRIDE_INIT(4, 4, 4, 152, 760),
  1, &relu_4_0_0_mean_conversion_output_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  relu_4_bias, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 38, 1, 1), AI_STRIDE_INIT(4, 4, 4, 152, 152),
  1, &relu_4_bias_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  relu_4_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 38, 5, 3), AI_STRIDE_INIT(4, 1, 1, 38, 190),
  1, &relu_4_output_array, &relu_4_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  relu_4_pad_before_output, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 1, 29, 12, 9), AI_STRIDE_INIT(4, 1, 1, 29, 348),
  1, &relu_4_pad_before_output_array, &relu_4_pad_before_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  relu_4_scratch0, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 6696, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6696, 6696),
  1, &relu_4_scratch0_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  relu_4_scratch1, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 1, 38, 10, 2), AI_STRIDE_INIT(4, 1, 1, 38, 380),
  1, &relu_4_scratch1_array, &relu_4_scratch1_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  relu_4_weights, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 29, 3, 3, 38), AI_STRIDE_INIT(4, 1, 29, 1102, 3306),
  1, &relu_4_weights_array, &relu_4_weights_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  relu_bias, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &relu_bias_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  relu_output, AI_STATIC,
  35, 0x1,
  AI_SHAPE_INIT(4, 1, 5, 80, 60), AI_STRIDE_INIT(4, 1, 1, 5, 400),
  1, &relu_output_array, &relu_output_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  relu_scratch0, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 196, 1, 1), AI_STRIDE_INIT(4, 1, 1, 196, 196),
  1, &relu_scratch0_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  relu_scratch1, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 5, 160, 2), AI_STRIDE_INIT(4, 1, 1, 5, 800),
  1, &relu_scratch1_array, &relu_scratch1_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  relu_weights, AI_STATIC,
  38, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 3, 5), AI_STRIDE_INIT(4, 1, 1, 5, 15),
  1, &relu_weights_array, &relu_weights_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  logits_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &logits_QuantizeLinear_Input_weights, &logits_QuantizeLinear_Input_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  logits_QuantizeLinear_Input_layer, 48,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &logits_QuantizeLinear_Input_chain,
  NULL, &logits_QuantizeLinear_Input_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mean_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_layer, 45,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_chain,
  NULL, &logits_QuantizeLinear_Input_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  mean_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mean_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mean_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &mean_Mul_scale, &mean_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mean_Mul_layer, 45,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &mean_Mul_chain,
  NULL, &mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float mean_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    mean_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    mean_neutral_value_data, mean_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  mean_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_4_0_0_mean_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mean_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mean_layer, 45,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &mean_chain,
  NULL, &mean_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &mean_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_4_0_0_mean_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_4_0_0_mean_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  relu_4_0_0_mean_conversion_layer, 42,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &relu_4_0_0_mean_conversion_chain,
  NULL, &mean_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_4_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &relu_4_weights, &relu_4_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &relu_4_scratch0, &relu_4_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  relu_4_layer, 42,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &relu_4_chain,
  NULL, &relu_4_0_0_mean_conversion_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 relu_4_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    relu_4_pad_before_value, AI_ARRAY_FORMAT_S8,
    relu_4_pad_before_value_data, relu_4_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_4_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_4_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  relu_4_pad_before_layer, 39,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &relu_4_pad_before_chain,
  NULL, &relu_4_layer, AI_STATIC, 
  .value = &relu_4_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_3_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &relu_3_weights, &relu_3_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &relu_3_scratch0, &relu_3_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  relu_3_layer, 36,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &relu_3_chain,
  NULL, &relu_4_pad_before_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 relu_3_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    relu_3_pad_before_value, AI_ARRAY_FORMAT_S8,
    relu_3_pad_before_value_data, relu_3_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_3_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_3_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  relu_3_pad_before_layer, 33,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &relu_3_pad_before_chain,
  NULL, &relu_3_layer, AI_STATIC, 
  .value = &relu_3_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_2_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &relu_2_weights, &relu_2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &relu_2_scratch0, &relu_2_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  relu_2_layer, 30,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &relu_2_chain,
  NULL, &relu_3_pad_before_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 relu_2_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    relu_2_pad_before_value, AI_ARRAY_FORMAT_S8,
    relu_2_pad_before_value_data, relu_2_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_2_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_2_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  relu_2_pad_before_layer, 27,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &relu_2_pad_before_chain,
  NULL, &relu_2_layer, AI_STATIC, 
  .value = &relu_2_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &relu_1_weights, &relu_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &relu_1_scratch0, &relu_1_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  relu_1_layer, 24,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_sssa8_ch_nl_pool,
  &relu_1_chain,
  NULL, &relu_2_pad_before_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &relu_weights, &relu_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &relu_scratch0, &relu_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  relu_layer, 18,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_sssa8_ch_nl_pool,
  &relu_chain,
  NULL, &relu_1_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 18052, 1, 1),
    18052, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 27640, 1, 1),
    27640, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_BIG_NET_PRUNED_INT8_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_BIG_NET_PRUNED_INT8_OUT_NUM, &logits_QuantizeLinear_Input_output),
  &relu_layer, 0xab092d2c, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 18052, 1, 1),
      18052, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 27640, 1, 1),
      27640, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_BIG_NET_PRUNED_INT8_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_BIG_NET_PRUNED_INT8_OUT_NUM, &logits_QuantizeLinear_Input_output),
  &relu_layer, 0xab092d2c, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool big_net_pruned_int8_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_big_net_pruned_int8_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_output_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 6432);
    input_output_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 6432);
    relu_scratch0_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 25632);
    relu_scratch0_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 25632);
    relu_scratch1_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 25828);
    relu_scratch1_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 25828);
    relu_output_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 820);
    relu_output_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 820);
    relu_1_scratch0_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 24820);
    relu_1_scratch0_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 24820);
    relu_1_scratch1_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 26040);
    relu_1_scratch1_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 26040);
    relu_1_output_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 0);
    relu_1_output_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 0);
    relu_2_pad_before_output_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 12000);
    relu_2_pad_before_output_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 12000);
    relu_2_scratch0_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 0);
    relu_2_scratch0_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 0);
    relu_2_scratch1_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 4048);
    relu_2_scratch1_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 4048);
    relu_2_output_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 5568);
    relu_2_output_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 5568);
    relu_3_pad_before_output_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 11268);
    relu_3_pad_before_output_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 11268);
    relu_3_scratch0_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 0);
    relu_3_scratch0_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 0);
    relu_3_scratch1_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 6212);
    relu_3_scratch1_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 6212);
    relu_3_output_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 7372);
    relu_3_output_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 7372);
    relu_4_pad_before_output_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 0);
    relu_4_pad_before_output_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 0);
    relu_4_scratch0_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 3132);
    relu_4_scratch0_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 3132);
    relu_4_scratch1_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 9828);
    relu_4_scratch1_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 9828);
    relu_4_output_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 10588);
    relu_4_output_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 10588);
    relu_4_0_0_mean_conversion_output_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 0);
    relu_4_0_0_mean_conversion_output_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 0);
    mean_output_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 2280);
    mean_output_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 2280);
    mean_Mul_output_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 0);
    mean_Mul_output_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 0);
    mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 152);
    mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 152);
    logits_QuantizeLinear_Input_scratch0_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 0);
    logits_QuantizeLinear_Input_scratch0_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 0);
    logits_QuantizeLinear_Input_output_array.data = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 136);
    logits_QuantizeLinear_Input_output_array.data_start = AI_PTR(g_big_net_pruned_int8_activations_map[0] + 136);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool big_net_pruned_int8_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_big_net_pruned_int8_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    relu_weights_array.format |= AI_FMT_FLAG_CONST;
    relu_weights_array.data = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 0);
    relu_weights_array.data_start = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 0);
    relu_bias_array.format |= AI_FMT_FLAG_CONST;
    relu_bias_array.data = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 48);
    relu_bias_array.data_start = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 48);
    relu_1_weights_array.format |= AI_FMT_FLAG_CONST;
    relu_1_weights_array.data = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 68);
    relu_1_weights_array.data_start = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 68);
    relu_1_bias_array.format |= AI_FMT_FLAG_CONST;
    relu_1_bias_array.data = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 520);
    relu_1_bias_array.data_start = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 520);
    relu_2_weights_array.format |= AI_FMT_FLAG_CONST;
    relu_2_weights_array.data = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 560);
    relu_2_weights_array.data_start = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 560);
    relu_2_bias_array.format |= AI_FMT_FLAG_CONST;
    relu_2_bias_array.data = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 2272);
    relu_2_bias_array.data_start = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 2272);
    relu_3_weights_array.format |= AI_FMT_FLAG_CONST;
    relu_3_weights_array.data = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 2348);
    relu_3_weights_array.data_start = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 2348);
    relu_3_bias_array.format |= AI_FMT_FLAG_CONST;
    relu_3_bias_array.data = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 7308);
    relu_3_bias_array.data_start = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 7308);
    relu_4_weights_array.format |= AI_FMT_FLAG_CONST;
    relu_4_weights_array.data = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 7424);
    relu_4_weights_array.data_start = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 7424);
    relu_4_bias_array.format |= AI_FMT_FLAG_CONST;
    relu_4_bias_array.data = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 17344);
    relu_4_bias_array.data_start = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 17344);
    mean_Mul_scale_array.format |= AI_FMT_FLAG_CONST;
    mean_Mul_scale_array.data = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 17496);
    mean_Mul_scale_array.data_start = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 17496);
    mean_Mul_bias_array.format |= AI_FMT_FLAG_CONST;
    mean_Mul_bias_array.data = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 17648);
    mean_Mul_bias_array.data_start = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 17648);
    logits_QuantizeLinear_Input_weights_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_weights_array.data = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 17800);
    logits_QuantizeLinear_Input_weights_array.data_start = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 17800);
    logits_QuantizeLinear_Input_bias_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_bias_array.data = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 18028);
    logits_QuantizeLinear_Input_bias_array.data_start = AI_PTR(g_big_net_pruned_int8_weights_map[0] + 18028);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_big_net_pruned_int8_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_BIG_NET_PRUNED_INT8_MODEL_NAME,
      .model_signature   = AI_BIG_NET_PRUNED_INT8_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 7437357,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xab092d2c,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_big_net_pruned_int8_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_BIG_NET_PRUNED_INT8_MODEL_NAME,
      .model_signature   = AI_BIG_NET_PRUNED_INT8_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 7437357,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0xab092d2c,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_big_net_pruned_int8_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_big_net_pruned_int8_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_big_net_pruned_int8_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_big_net_pruned_int8_create(network, AI_BIG_NET_PRUNED_INT8_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_big_net_pruned_int8_data_params_get(&params) != true) {
    err = ai_big_net_pruned_int8_get_error(*network);
    return err;
  }
#if defined(AI_BIG_NET_PRUNED_INT8_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_BIG_NET_PRUNED_INT8_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_big_net_pruned_int8_init(*network, &params) != true) {
    err = ai_big_net_pruned_int8_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_big_net_pruned_int8_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_big_net_pruned_int8_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_big_net_pruned_int8_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_big_net_pruned_int8_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= big_net_pruned_int8_configure_weights(net_ctx, params);
  ok &= big_net_pruned_int8_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_big_net_pruned_int8_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_big_net_pruned_int8_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_BIG_NET_PRUNED_INT8_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

