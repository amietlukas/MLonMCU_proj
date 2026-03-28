/**
  ******************************************************************************
  * @file    small_net_int8.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-03-17T22:42:29+0100
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


#include "small_net_int8.h"
#include "small_net_int8_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_small_net_int8
 
#undef AI_SMALL_NET_INT8_MODEL_SIGNATURE
#define AI_SMALL_NET_INT8_MODEL_SIGNATURE     "0x5c010cc5aa9830967d0251cbf294f927"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-03-17T22:42:29+0100"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_SMALL_NET_INT8_N_BATCHES
#define AI_SMALL_NET_INT8_N_BATCHES         (1)

static ai_ptr g_small_net_int8_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_small_net_int8_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 49152, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  input_Transpose_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 49153, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 69696, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32768, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36992, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 20736, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_0_0__ReduceMean_output_0_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _ReduceMean_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 6, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 432, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4608, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 73728, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 6, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1196, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6144, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7168, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 158, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_ReduceMean_output_0_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006670780945569277f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_0_features_0_2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03525615110993385f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_0_features_0_2_Relu_output_0_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03525615110993385f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_0_features_0_2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00830912496894598f, 0.009788244031369686f, 0.010398535057902336f, 0.005213859956711531f, 0.006062234286218882f, 0.006684453226625919f, 0.004889512434601784f, 0.01920527219772339f, 0.006813144776970148f, 0.0029991446062922478f, 0.011437103152275085f, 0.002947825938463211f, 0.021771807223558426f, 0.0272486824542284f, 0.008688490837812424f, 0.0035155501682311296f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_1_features_1_2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0252937451004982f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_1_features_1_2_Relu_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03525615110993385f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_1_features_1_2_Relu_output_0_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0252937451004982f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_1_features_1_2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0032133273780345917f, 0.002488398691639304f, 0.0020759706385433674f, 0.0027237245813012123f, 0.0040495493449270725f, 0.0036678637843579054f, 0.002977327210828662f, 0.004307878203690052f, 0.0028630082961171865f, 0.004935287404805422f, 0.002706240164116025f, 0.0038138250820338726f, 0.0029397194739431143f, 0.0017041170503944159f, 0.005062629468739033f, 0.0029711932875216007f, 0.004193982575088739f, 0.0037290670443326235f, 0.004112157039344311f, 0.002694177208468318f, 0.003246851498261094f, 0.0018934309482574463f, 0.0023837790358811617f, 0.0021341422107070684f, 0.0021762847900390625f, 0.0033096098341047764f, 0.0034563704393804073f, 0.0026423123199492693f, 0.00218344759196043f, 0.003269105451181531f, 0.0031925514340400696f, 0.002649005502462387f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_2_features_2_2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010670177638530731f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_2_features_2_2_Relu_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0252937451004982f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_2_features_2_2_Relu_output_0_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010670177638530731f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_2_features_2_2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(1.0f, 1.0f, 1.0f, 0.0006503660697489977f, 1.0f, 7.581370253806341e-11f, 0.0018055855762213469f, 5.45411438235277e-10f, 0.0015766127035021782f, 1.0f, 1.0f, 1.0f, 1.0f, 0.002652497496455908f, 1.0f, 0.0034077600575983524f, 1.0f, 0.0020693112164735794f, 3.178227681033263e-10f, 1.0f, 1.0f, 8.657068811812607e-11f, 0.002326391637325287f, 1.0f, 1.0f, 0.0012249244609847665f, 0.0010327218333259225f, 1.0f, 0.0007238506805151701f, 7.70258775446564e-05f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0005033220513723791f, 0.0024456451646983624f, 4.985520440925484e-09f, 3.4776258261981496e-11f, 0.0018890468636527658f, 0.00263780914247036f, 1.0f, 1.0f, 0.0029291363898664713f, 0.0004813329433090985f, 0.002103115664795041f, 1.0f, 0.00295922439545393f, 7.963921962073073e-05f, 0.002720039337873459f, 0.0026845077518373728f, 0.001153217046521604f, 1.0f, 1.0f, 1.0f, 0.0022499514743685722f, 1.0f, 0.0028516382444649935f, 0.0032576201483607292f, 0.003099356545135379f, 1.0f, 0.001988042378798127f, 0.0028956106398254633f, 0.002208217279985547f, 1.0f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_3_features_3_2_Relu_output_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19466465711593628f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_3_features_3_2_Relu_output_0_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010670177638530731f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_3_features_3_2_Relu_output_0_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19466465711593628f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(_features_features_3_features_3_2_Relu_output_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06324426084756851f, 0.05722503736615181f, 0.05620286241173744f, 0.05197858437895775f, 0.05320148915052414f, 0.0572923868894577f, 0.05836890637874603f, 0.044716957956552505f, 0.03897199407219887f, 0.06234801188111305f, 0.05442174896597862f, 0.05665706470608711f, 0.06854261457920074f, 0.04766403138637543f, 0.07276255637407303f, 1.0f, 0.0450068823993206f, 0.05239398404955864f, 0.07616756856441498f, 0.058265525847673416f, 0.05686821788549423f, 0.05058176442980766f, 0.039251215755939484f, 0.0684020146727562f, 0.06628961861133575f, 0.07004323601722717f, 0.047215450555086136f, 0.029515091329813004f, 0.05247564613819122f, 0.04934249818325043f, 0.03721798583865166f, 0.051358796656131744f, 0.041193123906850815f, 0.045832302421331406f, 0.048520103096961975f, 0.05663083866238594f, 0.05717337504029274f, 0.0406084842979908f, 0.0533205047249794f, 0.05667310580611229f, 0.03890584409236908f, 1.0f, 0.053512074053287506f, 0.05908622592687607f, 0.036318421363830566f, 0.04261606186628342f, 0.05218270793557167f, 0.03587751090526581f, 0.06174260750412941f, 0.051912643015384674f, 1.0f, 0.061390943825244904f, 0.03247039392590523f, 0.04316568374633789f, 0.05413832888007164f, 0.05709022283554077f, 0.07392855733633041f, 0.05268498510122299f, 0.06362644582986832f, 0.0448494516313076f, 0.042436711490154266f, 0.05364479869604111f, 0.04995473846793175f, 0.07206834107637405f, 0.04757476970553398f, 0.0455663837492466f, 0.07393640279769897f, 0.046395186334848404f, 0.05647312477231026f, 0.05314331501722336f, 0.054887883365154266f, 0.057562749832868576f, 0.05440353974699974f, 0.04840051010251045f, 0.0520566962659359f, 0.04105595499277115f, 0.04948579892516136f, 0.04600456729531288f, 0.0564415268599987f, 0.04629950970411301f, 0.059617266058921814f, 0.04981372877955437f, 0.04613195359706879f, 0.051103509962558746f, 0.08361895382404327f, 0.04478384926915169f, 0.05040229484438896f, 0.049132298678159714f, 0.05541287362575531f, 0.049318041652441025f, 1.0f, 0.03806794062256813f, 0.04839017242193222f, 0.0452144481241703f, 0.0506577230989933f, 0.07103225588798523f, 0.049829740077257156f, 0.03854832798242569f, 0.05471986532211304f, 1.0f, 0.04037042707204819f, 0.05736509710550308f, 0.06940469145774841f, 0.0747755840420723f, 0.07982563227415085f, 1.0f, 0.04603927582502365f, 1.0f, 0.05440777540206909f, 0.059379663318395615f, 0.05247056111693382f, 0.06497662514448166f, 0.04774576053023338f, 0.06875179708003998f, 1.0f, 0.0548258014023304f, 0.06580636650323868f, 0.04629504308104515f, 0.07349793612957001f, 0.05726396292448044f, 0.039178021252155304f, 0.04559376463294029f, 0.04068482667207718f, 0.06988712400197983f, 0.047414667904376984f, 0.05847998708486557f, 0.0686020702123642f, 0.058463647961616516f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(input_Transpose_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013167597353458405f),
    AI_PACK_INTQ_ZP(-37)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(input_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013167597353458405f),
    AI_PACK_INTQ_ZP(-37)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13516253232955933f),
    AI_PACK_INTQ_ZP(-43)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 6,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01051893550902605f, 0.013834615238010883f, 0.014469115063548088f, 0.012196398340165615f, 0.015407524071633816f, 0.013131944462656975f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_0_0_logits_QuantizeLinear_Input_conversion_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &_ReduceMean_output_0_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array, &_ReduceMean_output_0_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_bias, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_ReduceMean_output_0_Mul_bias_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_ReduceMean_output_0_Mul_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_scale, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_ReduceMean_output_0_Mul_scale_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _ReduceMean_output_0_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_ReduceMean_output_0_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_bias, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &_features_features_0_features_0_2_Relu_output_0_bias_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 64, 64), AI_STRIDE_INIT(4, 1, 1, 16, 1024),
  1, &_features_features_0_features_0_2_Relu_output_0_output_array, &_features_features_0_features_0_2_Relu_output_0_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_scratch0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 1196, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1196, 1196),
  1, &_features_features_0_features_0_2_Relu_output_0_scratch0_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_scratch1, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 128, 2), AI_STRIDE_INIT(4, 1, 1, 16, 2048),
  1, &_features_features_0_features_0_2_Relu_output_0_scratch1_array, &_features_features_0_features_0_2_Relu_output_0_scratch1_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_weights, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 3, 3, 3, 16), AI_STRIDE_INIT(4, 1, 3, 48, 144),
  1, &_features_features_0_features_0_2_Relu_output_0_weights_array, &_features_features_0_features_0_2_Relu_output_0_weights_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &_features_features_1_features_1_2_Relu_output_0_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 32), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &_features_features_1_features_1_2_Relu_output_0_output_array, &_features_features_1_features_1_2_Relu_output_0_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_pad_before_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 66, 66), AI_STRIDE_INIT(4, 1, 1, 16, 1056),
  1, &_features_features_1_features_1_2_Relu_output_0_pad_before_output_array, &_features_features_1_features_1_2_Relu_output_0_pad_before_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_scratch0, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 6144, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6144, 6144),
  1, &_features_features_1_features_1_2_Relu_output_0_scratch0_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_scratch1, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 64, 2), AI_STRIDE_INIT(4, 1, 1, 32, 2048),
  1, &_features_features_1_features_1_2_Relu_output_0_scratch1_array, &_features_features_1_features_1_2_Relu_output_0_scratch1_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_weights, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 1, 16, 512, 1536),
  1, &_features_features_1_features_1_2_Relu_output_0_weights_array, &_features_features_1_features_1_2_Relu_output_0_weights_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_features_features_2_features_2_2_Relu_output_0_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 16, 16), AI_STRIDE_INIT(4, 1, 1, 64, 1024),
  1, &_features_features_2_features_2_2_Relu_output_0_output_array, &_features_features_2_features_2_2_Relu_output_0_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_pad_before_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 34, 34), AI_STRIDE_INIT(4, 1, 1, 32, 1088),
  1, &_features_features_2_features_2_2_Relu_output_0_pad_before_output_array, &_features_features_2_features_2_2_Relu_output_0_pad_before_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_scratch0, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 7168, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7168, 7168),
  1, &_features_features_2_features_2_2_Relu_output_0_scratch0_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_scratch1, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 32, 2), AI_STRIDE_INIT(4, 1, 1, 64, 2048),
  1, &_features_features_2_features_2_2_Relu_output_0_scratch1_array, &_features_features_2_features_2_2_Relu_output_0_scratch1_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_weights, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 32, 3, 3, 64), AI_STRIDE_INIT(4, 1, 32, 2048, 6144),
  1, &_features_features_2_features_2_2_Relu_output_0_weights_array, &_features_features_2_features_2_2_Relu_output_0_weights_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_0_0__ReduceMean_output_0_conversion_output, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 8, 8), AI_STRIDE_INIT(4, 4, 4, 512, 4096),
  1, &_features_features_3_features_3_2_Relu_output_0_0_0__ReduceMean_output_0_conversion_output_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_bias, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_features_features_3_features_3_2_Relu_output_0_bias_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 8, 8), AI_STRIDE_INIT(4, 1, 1, 128, 1024),
  1, &_features_features_3_features_3_2_Relu_output_0_output_array, &_features_features_3_features_3_2_Relu_output_0_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_pad_before_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 18, 18), AI_STRIDE_INIT(4, 1, 1, 64, 1152),
  1, &_features_features_3_features_3_2_Relu_output_0_pad_before_output_array, &_features_features_3_features_3_2_Relu_output_0_pad_before_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_scratch0, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 9216, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9216, 9216),
  1, &_features_features_3_features_3_2_Relu_output_0_scratch0_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_scratch1, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 16, 2), AI_STRIDE_INIT(4, 1, 1, 128, 2048),
  1, &_features_features_3_features_3_2_Relu_output_0_scratch1_array, &_features_features_3_features_3_2_Relu_output_0_scratch1_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_weights, AI_STATIC,
  28, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 3, 128), AI_STRIDE_INIT(4, 1, 64, 8192, 24576),
  1, &_features_features_3_features_3_2_Relu_output_0_weights_array, &_features_features_3_features_3_2_Relu_output_0_weights_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  input_Transpose_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 128, 128), AI_STRIDE_INIT(4, 1, 1, 3, 384),
  1, &input_Transpose_output_array, &input_Transpose_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  input_output, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 128, 3), AI_STRIDE_INIT(4, 1, 1, 128, 16384),
  1, &input_output_array, &input_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &logits_QuantizeLinear_Input_bias_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output, AI_STATIC,
  32, 0x1,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6, 6),
  1, &logits_QuantizeLinear_Input_output_array, &logits_QuantizeLinear_Input_output_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 158, 1, 1), AI_STRIDE_INIT(4, 2, 2, 316, 316),
  1, &logits_QuantizeLinear_Input_scratch0_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights, AI_STATIC,
  34, 0x1,
  AI_SHAPE_INIT(4, 128, 6, 1, 1), AI_STRIDE_INIT(4, 1, 128, 768, 768),
  1, &logits_QuantizeLinear_Input_weights_array, &logits_QuantizeLinear_Input_weights_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  logits_QuantizeLinear_Input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceMean_output_0_Mul_0_0_logits_QuantizeLinear_Input_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &logits_QuantizeLinear_Input_weights, &logits_QuantizeLinear_Input_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &logits_QuantizeLinear_Input_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  logits_QuantizeLinear_Input_layer, 40,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &logits_QuantizeLinear_Input_chain,
  NULL, &logits_QuantizeLinear_Input_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_0_0_logits_QuantizeLinear_Input_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceMean_output_0_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceMean_output_0_Mul_0_0_logits_QuantizeLinear_Input_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_0_0_logits_QuantizeLinear_Input_conversion_layer, 37,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_ReduceMean_output_0_Mul_0_0_logits_QuantizeLinear_Input_conversion_chain,
  NULL, &logits_QuantizeLinear_Input_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceMean_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceMean_output_0_Mul_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_ReduceMean_output_0_Mul_scale, &_ReduceMean_output_0_Mul_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _ReduceMean_output_0_Mul_layer, 37,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_ReduceMean_output_0_Mul_chain,
  NULL, &_ReduceMean_output_0_Mul_0_0_logits_QuantizeLinear_Input_conversion_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_float _ReduceMean_output_0_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    _ReduceMean_output_0_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    _ReduceMean_output_0_neutral_value_data, _ReduceMean_output_0_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _ReduceMean_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_features_3_2_Relu_output_0_0_0__ReduceMean_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceMean_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _ReduceMean_output_0_layer, 37,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &_ReduceMean_output_0_chain,
  NULL, &_ReduceMean_output_0_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &_ReduceMean_output_0_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_0_0__ReduceMean_output_0_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_features_3_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_features_3_2_Relu_output_0_0_0__ReduceMean_output_0_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_0_0__ReduceMean_output_0_conversion_layer, 34,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &_features_features_3_features_3_2_Relu_output_0_0_0__ReduceMean_output_0_conversion_chain,
  NULL, &_ReduceMean_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_features_3_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_features_3_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_3_features_3_2_Relu_output_0_weights, &_features_features_3_features_3_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_3_features_3_2_Relu_output_0_scratch0, &_features_features_3_features_3_2_Relu_output_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_layer, 34,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &_features_features_3_features_3_2_Relu_output_0_chain,
  NULL, &_features_features_3_features_3_2_Relu_output_0_0_0__ReduceMean_output_0_conversion_layer, AI_STATIC, 
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


AI_STATIC_CONST ai_i8 _features_features_3_features_3_2_Relu_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _features_features_3_features_3_2_Relu_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _features_features_3_features_3_2_Relu_output_0_pad_before_value_data, _features_features_3_features_3_2_Relu_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_features_2_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_3_features_3_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_3_features_3_2_Relu_output_0_pad_before_layer, 31,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_features_features_3_features_3_2_Relu_output_0_pad_before_chain,
  NULL, &_features_features_3_features_3_2_Relu_output_0_layer, AI_STATIC, 
  .value = &_features_features_3_features_3_2_Relu_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_features_2_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_features_2_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_2_features_2_2_Relu_output_0_weights, &_features_features_2_features_2_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_2_features_2_2_Relu_output_0_scratch0, &_features_features_2_features_2_2_Relu_output_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_layer, 28,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &_features_features_2_features_2_2_Relu_output_0_chain,
  NULL, &_features_features_3_features_3_2_Relu_output_0_pad_before_layer, AI_STATIC, 
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


AI_STATIC_CONST ai_i8 _features_features_2_features_2_2_Relu_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _features_features_2_features_2_2_Relu_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _features_features_2_features_2_2_Relu_output_0_pad_before_value_data, _features_features_2_features_2_2_Relu_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_features_1_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_2_features_2_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_2_features_2_2_Relu_output_0_pad_before_layer, 25,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_features_features_2_features_2_2_Relu_output_0_pad_before_chain,
  NULL, &_features_features_2_features_2_2_Relu_output_0_layer, AI_STATIC, 
  .value = &_features_features_2_features_2_2_Relu_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_features_1_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_features_1_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_1_features_1_2_Relu_output_0_weights, &_features_features_1_features_1_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_1_features_1_2_Relu_output_0_scratch0, &_features_features_1_features_1_2_Relu_output_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_layer, 22,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &_features_features_1_features_1_2_Relu_output_0_chain,
  NULL, &_features_features_2_features_2_2_Relu_output_0_pad_before_layer, AI_STATIC, 
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


AI_STATIC_CONST ai_i8 _features_features_1_features_1_2_Relu_output_0_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    _features_features_1_features_1_2_Relu_output_0_pad_before_value, AI_ARRAY_FORMAT_S8,
    _features_features_1_features_1_2_Relu_output_0_pad_before_value_data, _features_features_1_features_1_2_Relu_output_0_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_features_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_1_features_1_2_Relu_output_0_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _features_features_1_features_1_2_Relu_output_0_pad_before_layer, 19,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &_features_features_1_features_1_2_Relu_output_0_pad_before_chain,
  NULL, &_features_features_1_features_1_2_Relu_output_0_layer, AI_STATIC, 
  .value = &_features_features_1_features_1_2_Relu_output_0_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_features_features_0_features_0_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_features_features_0_features_0_2_Relu_output_0_weights, &_features_features_0_features_0_2_Relu_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_features_features_0_features_0_2_Relu_output_0_scratch0, &_features_features_0_features_0_2_Relu_output_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  _features_features_0_features_0_2_Relu_output_0_layer, 16,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_sssa8_ch_nl_pool,
  &_features_features_0_features_0_2_Relu_output_0_chain,
  NULL, &_features_features_1_features_1_2_Relu_output_0_pad_before_layer, AI_STATIC, 
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
  input_Transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_Transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_Transpose_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &input_Transpose_chain,
  NULL, &_features_features_0_features_0_2_Relu_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 99976, 1, 1),
    99976, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 98308, 1, 1),
    98308, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_SMALL_NET_INT8_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_SMALL_NET_INT8_OUT_NUM, &logits_QuantizeLinear_Input_output),
  &input_Transpose_layer, 0x02d294b3, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 99976, 1, 1),
      99976, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 98308, 1, 1),
      98308, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_SMALL_NET_INT8_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_SMALL_NET_INT8_OUT_NUM, &logits_QuantizeLinear_Input_output),
  &input_Transpose_layer, 0x02d294b3, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool small_net_int8_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_small_net_int8_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    input_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    input_Transpose_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 49152);
    input_Transpose_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 49152);
    _features_features_0_features_0_2_Relu_output_0_scratch0_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    _features_features_0_features_0_2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    _features_features_0_features_0_2_Relu_output_0_scratch1_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 1196);
    _features_features_0_features_0_2_Relu_output_0_scratch1_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 1196);
    _features_features_0_features_0_2_Relu_output_0_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 30688);
    _features_features_0_features_0_2_Relu_output_0_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 30688);
    _features_features_1_features_1_2_Relu_output_0_pad_before_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 26528);
    _features_features_1_features_1_2_Relu_output_0_pad_before_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 26528);
    _features_features_1_features_1_2_Relu_output_0_scratch0_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    _features_features_1_features_1_2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    _features_features_1_features_1_2_Relu_output_0_scratch1_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 6144);
    _features_features_1_features_1_2_Relu_output_0_scratch1_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 6144);
    _features_features_1_features_1_2_Relu_output_0_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 25504);
    _features_features_1_features_1_2_Relu_output_0_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 25504);
    _features_features_2_features_2_2_Relu_output_0_pad_before_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 58272);
    _features_features_2_features_2_2_Relu_output_0_pad_before_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 58272);
    _features_features_2_features_2_2_Relu_output_0_scratch0_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    _features_features_2_features_2_2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    _features_features_2_features_2_2_Relu_output_0_scratch1_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 7168);
    _features_features_2_features_2_2_Relu_output_0_scratch1_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 7168);
    _features_features_2_features_2_2_Relu_output_0_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 11264);
    _features_features_2_features_2_2_Relu_output_0_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 11264);
    _features_features_3_features_3_2_Relu_output_0_pad_before_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 27648);
    _features_features_3_features_3_2_Relu_output_0_pad_before_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 27648);
    _features_features_3_features_3_2_Relu_output_0_scratch0_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    _features_features_3_features_3_2_Relu_output_0_scratch0_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    _features_features_3_features_3_2_Relu_output_0_scratch1_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 9216);
    _features_features_3_features_3_2_Relu_output_0_scratch1_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 9216);
    _features_features_3_features_3_2_Relu_output_0_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 13312);
    _features_features_3_features_3_2_Relu_output_0_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 13312);
    _features_features_3_features_3_2_Relu_output_0_0_0__ReduceMean_output_0_conversion_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 21504);
    _features_features_3_features_3_2_Relu_output_0_0_0__ReduceMean_output_0_conversion_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 21504);
    _ReduceMean_output_0_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    _ReduceMean_output_0_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    _ReduceMean_output_0_Mul_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 512);
    _ReduceMean_output_0_Mul_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 512);
    _ReduceMean_output_0_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    _ReduceMean_output_0_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    logits_QuantizeLinear_Input_scratch0_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 128);
    logits_QuantizeLinear_Input_scratch0_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 128);
    logits_QuantizeLinear_Input_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 444);
    logits_QuantizeLinear_Input_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 444);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool small_net_int8_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_small_net_int8_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _features_features_0_features_0_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_0_features_0_2_Relu_output_0_weights_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 0);
    _features_features_0_features_0_2_Relu_output_0_weights_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 0);
    _features_features_0_features_0_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_0_features_0_2_Relu_output_0_bias_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 432);
    _features_features_0_features_0_2_Relu_output_0_bias_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 432);
    _features_features_1_features_1_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_1_features_1_2_Relu_output_0_weights_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 496);
    _features_features_1_features_1_2_Relu_output_0_weights_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 496);
    _features_features_1_features_1_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_1_features_1_2_Relu_output_0_bias_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 5104);
    _features_features_1_features_1_2_Relu_output_0_bias_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 5104);
    _features_features_2_features_2_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_2_features_2_2_Relu_output_0_weights_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 5232);
    _features_features_2_features_2_2_Relu_output_0_weights_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 5232);
    _features_features_2_features_2_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_2_features_2_2_Relu_output_0_bias_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 23664);
    _features_features_2_features_2_2_Relu_output_0_bias_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 23664);
    _features_features_3_features_3_2_Relu_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _features_features_3_features_3_2_Relu_output_0_weights_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 23920);
    _features_features_3_features_3_2_Relu_output_0_weights_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 23920);
    _features_features_3_features_3_2_Relu_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _features_features_3_features_3_2_Relu_output_0_bias_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 97648);
    _features_features_3_features_3_2_Relu_output_0_bias_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 97648);
    _ReduceMean_output_0_Mul_scale_array.format |= AI_FMT_FLAG_CONST;
    _ReduceMean_output_0_Mul_scale_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 98160);
    _ReduceMean_output_0_Mul_scale_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 98160);
    _ReduceMean_output_0_Mul_bias_array.format |= AI_FMT_FLAG_CONST;
    _ReduceMean_output_0_Mul_bias_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 98672);
    _ReduceMean_output_0_Mul_bias_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 98672);
    logits_QuantizeLinear_Input_weights_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_weights_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 99184);
    logits_QuantizeLinear_Input_weights_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 99184);
    logits_QuantizeLinear_Input_bias_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_bias_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 99952);
    logits_QuantizeLinear_Input_bias_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 99952);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_small_net_int8_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_SMALL_NET_INT8_MODEL_NAME,
      .model_signature   = AI_SMALL_NET_INT8_MODEL_SIGNATURE,
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
      
      .n_macc            = 64243190,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x02d294b3,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_small_net_int8_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_SMALL_NET_INT8_MODEL_NAME,
      .model_signature   = AI_SMALL_NET_INT8_MODEL_SIGNATURE,
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
      
      .n_macc            = 64243190,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x02d294b3,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_small_net_int8_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_small_net_int8_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_small_net_int8_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_small_net_int8_create(network, AI_SMALL_NET_INT8_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_small_net_int8_data_params_get(&params) != true) {
    err = ai_small_net_int8_get_error(*network);
    return err;
  }
#if defined(AI_SMALL_NET_INT8_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_SMALL_NET_INT8_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_small_net_int8_init(*network, &params) != true) {
    err = ai_small_net_int8_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_small_net_int8_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_small_net_int8_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_small_net_int8_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_small_net_int8_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= small_net_int8_configure_weights(net_ctx, params);
  ok &= small_net_int8_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_small_net_int8_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_small_net_int8_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_SMALL_NET_INT8_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

