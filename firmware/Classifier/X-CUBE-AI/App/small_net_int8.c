/**
  ******************************************************************************
  * @file    small_net_int8.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-05-18T15:00:37+0200
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
#define AI_SMALL_NET_INT8_MODEL_SIGNATURE     "0x340f0f3482d35f0f3e576978248dd4ab"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-05-18T15:00:37+0200"

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
  NULL, NULL, 19200, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  relu_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 153600, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  relu_1_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 162688, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  relu_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 76800, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_pad_before_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 86016, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 38400, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_0_0_mean_conversion_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 38400, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  mean_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  mean_Mul_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 6, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  relu_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  relu_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  relu_1_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  relu_1_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 73728, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  mean_Mul_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  mean_Mul_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 6, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  relu_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1060, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  relu_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 10240, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  relu_1_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7168, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  relu_1_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 10240, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  relu_2_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 10240, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  logits_QuantizeLinear_Input_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 158, AI_STATIC)

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
    AI_PACK_INTQ_SCALE(0.16875247657299042f),
    AI_PACK_INTQ_ZP(25)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(logits_QuantizeLinear_Input_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 6,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02964905835688114f, 0.03223033621907234f, 0.025518637150526047f, 0.027716774493455887f, 0.029141319915652275f, 0.02521609701216221f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.004278772044926882f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01282228622585535f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_1_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06996529549360275f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_1_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01282228622585535f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_1_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007701183203607798f, 0.002451922744512558f, 0.0007130678277462721f, 0.002417676616460085f, 0.001451952150091529f, 0.00201168330386281f, 0.0017543663270771503f, 0.0029284656047821045f, 0.0007799169979989529f, 0.0008495361544191837f, 0.0012617302127182484f, 0.001110969576984644f, 0.002024636138230562f, 0.0009423884912393987f, 0.0019048009999096394f, 0.0026706268545240164f, 0.002202813047915697f, 0.0027215981390327215f, 0.0009473918471485376f, 0.0020907805301249027f, 0.002393139060586691f, 0.0011010132730007172f, 0.002290974836796522f, 0.0009707134449854493f, 0.002890432719141245f, 0.0010046525858342648f, 0.00038354290882125497f, 0.0027594983112066984f, 0.0022184185218065977f, 0.0023770187981426716f, 0.002084884559735656f, 0.002379286102950573f, 0.0003744321293197572f, 0.0019719949923455715f, 0.001083495793864131f, 0.0013547830749303102f, 0.0017490737373009324f, 0.0017991961212828755f, 0.0028522599022835493f, 0.0026843026280403137f, 0.0018838838441297412f, 0.0025796417612582445f, 0.003500324673950672f, 0.0016101222718134522f, 0.001980618806555867f, 0.0019886912778019905f, 0.0017500084359198809f, 0.0019611131865531206f, 1.0f, 0.0015164089854806662f, 0.003890829626470804f, 0.0012385437730699778f, 0.0021611556876450777f, 0.002260429784655571f, 0.00129612791351974f, 0.0018485181499272585f, 0.0008896776125766337f, 0.0019302534637972713f, 0.0020371933933347464f, 0.0011835274053737521f, 0.0018660168861970305f, 0.0020255516283214092f, 0.0018633611034601927f, 0.0014744860818609595f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_2_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15299513936042786f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_2_pad_before_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01282228622585535f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_2_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.15299513936042786f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_2_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03581182286143303f, 0.0503058098256588f, 0.05270405113697052f, 0.037275779992341995f, 0.052999354898929596f, 0.04869711399078369f, 0.033215347677469254f, 0.03987458348274231f, 0.042281121015548706f, 0.04884995520114899f, 0.034256692975759506f, 0.03579697757959366f, 0.057824742048978806f, 0.04378026723861694f, 0.0471554771065712f, 0.04717652127146721f, 0.03207671642303467f, 0.03944374620914459f, 0.05073569715023041f, 0.04787333309650421f, 0.05056995153427124f, 0.04192347452044487f, 0.04367433860898018f, 0.042611535638570786f, 0.06321115791797638f, 0.04258756339550018f, 0.0640842467546463f, 0.036841657012701035f, 0.04422628879547119f, 0.03325890004634857f, 0.0544312559068203f, 0.06299437582492828f, 0.03966333344578743f, 0.05008792504668236f, 0.06139615923166275f, 0.04666929319500923f, 0.04732906073331833f, 0.022580605000257492f, 0.0498795211315155f, 0.050742194056510925f, 0.04438479617238045f, 0.044585149735212326f, 0.04580765217542648f, 0.024510128423571587f, 0.042751822620630264f, 0.06312105804681778f, 0.039216868579387665f, 0.04317238926887512f, 0.04686863720417023f, 0.042639147490262985f, 0.04236472398042679f, 0.047703370451927185f, 0.045204926282167435f, 0.02516336925327778f, 0.03709346055984497f, 0.04502519965171814f, 0.051567140966653824f, 0.03734862059354782f, 0.04486710578203201f, 0.05561875179409981f, 0.03175182268023491f, 0.03723021224141121f, 0.04862745478749275f, 0.042070165276527405f, 0.04616960883140564f, 0.047199297696352005f, 0.04809025302529335f, 0.03374427184462547f, 0.03944738209247589f, 0.037726689130067825f, 0.06527204066514969f, 0.05999099090695381f, 0.04744718223810196f, 0.042250532656908035f, 0.05585476756095886f, 0.038708098232746124f, 0.03926851600408554f, 0.038989193737506866f, 0.05133870989084244f, 0.04786843806505203f, 0.04938173294067383f, 0.030984869226813316f, 0.04653563350439072f, 0.04520973190665245f, 0.04803874343633652f, 0.05173385515809059f, 0.04759624972939491f, 0.0411229208111763f, 0.06196436658501625f, 0.05957882106304169f, 0.04659270867705345f, 0.022788146510720253f, 0.04525643214583397f, 0.047608476132154465f, 0.04470483586192131f, 0.034433815628290176f, 0.0606573224067688f, 0.03703894466161728f, 0.026355469599366188f, 0.049387235194444656f, 0.03889724984765053f, 0.04195976257324219f, 0.039104163646698f, 0.04588983952999115f, 0.04922160878777504f, 0.07381950318813324f, 0.050831180065870285f, 0.04222269728779793f, 0.04614676162600517f, 0.03645540028810501f, 0.04294469207525253f, 0.09678973257541656f, 0.04181953892111778f, 0.05127928406000137f, 0.06335138529539108f, 0.048918917775154114f, 0.042488373816013336f, 0.04368419572710991f, 0.04151752218604088f, 0.018021637573838234f, 0.036184683442115784f, 0.05497440695762634f, 0.04064807668328285f, 0.04401438310742378f, 0.04252950847148895f, 0.04431905597448349f, 0.055196963250637054f, 0.04805539920926094f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06996529549360275f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06996529549360275f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(relu_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00013433658750727773f, 0.00013422452320810407f, 8.885352144716308e-05f, 0.00013805381604470313f, 0.00021666452812496573f, 0.00016329248319379985f, 0.0001552831381559372f, 0.00017044875130522996f, 0.0001020787822199054f, 0.00011874474148498848f, 9.350483014713973e-05f, 0.00010078357445308939f, 0.00012943528417963535f, 0.00012653767771553248f, 0.00012000386777799577f, 0.0001996540231630206f, 0.00022995797917246819f, 0.00013485722593031824f, 0.00021168109378777444f, 0.00010089603892993182f, 0.00017983584257308394f, 0.0003662600356619805f, 0.0001429829717380926f, 0.00013507904077414423f, 0.0001674472150625661f, 0.0001435523445252329f, 0.00021432088396977633f, 0.0002443522389512509f, 0.0001243160804733634f, 0.00015616726886946708f, 0.00016324309399351478f, 0.00022873285342939198f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

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
  AI_SHAPE_INIT(4, 1, 158, 1, 1), AI_STRIDE_INIT(4, 2, 2, 316, 316),
  1, &logits_QuantizeLinear_Input_scratch0_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  logits_QuantizeLinear_Input_weights, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 128, 6, 1, 1), AI_STRIDE_INIT(4, 1, 128, 768, 768),
  1, &logits_QuantizeLinear_Input_weights_array, &logits_QuantizeLinear_Input_weights_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array, &mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  mean_Mul_bias, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &mean_Mul_bias_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  mean_Mul_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &mean_Mul_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  mean_Mul_scale, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &mean_Mul_scale_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  mean_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &mean_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  relu_1_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &relu_1_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  relu_1_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 40, 30), AI_STRIDE_INIT(4, 1, 1, 64, 2560),
  1, &relu_1_output_array, &relu_1_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  relu_1_pad_before_output, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 82, 62), AI_STRIDE_INIT(4, 1, 1, 32, 2624),
  1, &relu_1_pad_before_output_array, &relu_1_pad_before_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  relu_1_scratch0, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 7168, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7168, 7168),
  1, &relu_1_scratch0_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  relu_1_scratch1, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 80, 2), AI_STRIDE_INIT(4, 1, 1, 64, 5120),
  1, &relu_1_scratch1_array, &relu_1_scratch1_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  relu_1_weights, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 32, 3, 3, 64), AI_STRIDE_INIT(4, 1, 32, 2048, 6144),
  1, &relu_1_weights_array, &relu_1_weights_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_0_0_mean_conversion_output, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 20, 15), AI_STRIDE_INIT(4, 4, 4, 512, 10240),
  1, &relu_2_0_0_mean_conversion_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_bias, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &relu_2_bias_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_output, AI_STATIC,
  18, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 20, 15), AI_STRIDE_INIT(4, 1, 1, 128, 2560),
  1, &relu_2_output_array, &relu_2_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_pad_before_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 42, 32), AI_STRIDE_INIT(4, 1, 1, 64, 2688),
  1, &relu_2_pad_before_output_array, &relu_2_pad_before_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_scratch0, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 9216, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9216, 9216),
  1, &relu_2_scratch0_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_scratch1, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 40, 2), AI_STRIDE_INIT(4, 1, 1, 128, 5120),
  1, &relu_2_scratch1_array, &relu_2_scratch1_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  relu_2_weights, AI_STATIC,
  22, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 3, 128), AI_STRIDE_INIT(4, 1, 64, 8192, 24576),
  1, &relu_2_weights_array, &relu_2_weights_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  relu_bias, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &relu_bias_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  relu_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 80, 60), AI_STRIDE_INIT(4, 1, 1, 32, 2560),
  1, &relu_output_array, &relu_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  relu_scratch0, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 1060, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1060, 1060),
  1, &relu_scratch0_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  relu_scratch1, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 160, 2), AI_STRIDE_INIT(4, 1, 1, 32, 5120),
  1, &relu_scratch1_array, &relu_scratch1_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  relu_weights, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 3, 32), AI_STRIDE_INIT(4, 1, 1, 32, 96),
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
  logits_QuantizeLinear_Input_layer, 32,
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
  mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_layer, 29,
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
  mean_Mul_layer, 29,
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
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_2_0_0_mean_conversion_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &mean_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  mean_layer, 29,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &mean_chain,
  NULL, &mean_Mul_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &mean_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_2_0_0_mean_conversion_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_2_0_0_mean_conversion_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  relu_2_0_0_mean_conversion_layer, 26,
  NL_TYPE, 0x0, NULL,
  nl, node_convert,
  &relu_2_0_0_mean_conversion_chain,
  NULL, &mean_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_2_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &relu_2_weights, &relu_2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &relu_2_scratch0, &relu_2_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  relu_2_layer, 26,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &relu_2_chain,
  NULL, &relu_2_0_0_mean_conversion_layer, AI_STATIC, 
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
  relu_2_pad_before_layer, 23,
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
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_1_pad_before_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &relu_1_weights, &relu_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &relu_1_scratch0, &relu_1_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  relu_1_layer, 20,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &relu_1_chain,
  NULL, &relu_2_pad_before_layer, AI_STATIC, 
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


AI_STATIC_CONST ai_i8 relu_1_pad_before_value_data[] = { -128 };
AI_ARRAY_OBJ_DECLARE(
    relu_1_pad_before_value, AI_ARRAY_FORMAT_S8,
    relu_1_pad_before_value_data, relu_1_pad_before_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_1_pad_before_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_1_pad_before_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  relu_1_pad_before_layer, 17,
  PAD_TYPE, 0x0, NULL,
  pad, forward_pad,
  &relu_1_pad_before_chain,
  NULL, &relu_1_layer, AI_STATIC, 
  .value = &relu_1_pad_before_value, 
  .mode = AI_PAD_CONSTANT, 
  .pads = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  relu_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &relu_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &relu_weights, &relu_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &relu_scratch0, &relu_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  relu_layer, 14,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_sssa8_ch_nl_pool,
  &relu_chain,
  NULL, &relu_1_pad_before_layer, AI_STATIC, 
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
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 95160, 1, 1),
    95160, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 184100, 1, 1),
    184100, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_SMALL_NET_INT8_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_SMALL_NET_INT8_OUT_NUM, &logits_QuantizeLinear_Input_output),
  &relu_layer, 0x75263b16, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 95160, 1, 1),
      95160, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 184100, 1, 1),
      184100, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_SMALL_NET_INT8_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_SMALL_NET_INT8_OUT_NUM, &logits_QuantizeLinear_Input_output),
  &relu_layer, 0x75263b16, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool small_net_int8_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_small_net_int8_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 10240);
    input_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 10240);
    relu_scratch0_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 29440);
    relu_scratch0_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 29440);
    relu_scratch1_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    relu_scratch1_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    relu_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 30500);
    relu_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 30500);
    relu_1_pad_before_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 21412);
    relu_1_pad_before_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 21412);
    relu_1_scratch0_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    relu_1_scratch0_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    relu_1_scratch1_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 7168);
    relu_1_scratch1_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 7168);
    relu_1_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 18852);
    relu_1_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 18852);
    relu_2_pad_before_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 9636);
    relu_2_pad_before_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 9636);
    relu_2_scratch0_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    relu_2_scratch0_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    relu_2_scratch1_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 95652);
    relu_2_scratch1_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 95652);
    relu_2_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 145700);
    relu_2_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 145700);
    relu_2_0_0_mean_conversion_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 30500);
    relu_2_0_0_mean_conversion_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 30500);
    mean_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    mean_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    mean_Mul_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 512);
    mean_Mul_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 512);
    mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array.data = AI_PTR(g_small_net_int8_activations_map[0] + 0);
    mean_Mul_0_0_logits_QuantizeLinear_Input_conversion_output_array.data_start = AI_PTR(g_small_net_int8_activations_map[0] + 0);
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
    
    relu_weights_array.format |= AI_FMT_FLAG_CONST;
    relu_weights_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 0);
    relu_weights_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 0);
    relu_bias_array.format |= AI_FMT_FLAG_CONST;
    relu_bias_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 288);
    relu_bias_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 288);
    relu_1_weights_array.format |= AI_FMT_FLAG_CONST;
    relu_1_weights_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 416);
    relu_1_weights_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 416);
    relu_1_bias_array.format |= AI_FMT_FLAG_CONST;
    relu_1_bias_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 18848);
    relu_1_bias_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 18848);
    relu_2_weights_array.format |= AI_FMT_FLAG_CONST;
    relu_2_weights_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 19104);
    relu_2_weights_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 19104);
    relu_2_bias_array.format |= AI_FMT_FLAG_CONST;
    relu_2_bias_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 92832);
    relu_2_bias_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 92832);
    mean_Mul_scale_array.format |= AI_FMT_FLAG_CONST;
    mean_Mul_scale_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 93344);
    mean_Mul_scale_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 93344);
    mean_Mul_bias_array.format |= AI_FMT_FLAG_CONST;
    mean_Mul_bias_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 93856);
    mean_Mul_bias_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 93856);
    logits_QuantizeLinear_Input_weights_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_weights_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 94368);
    logits_QuantizeLinear_Input_weights_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 94368);
    logits_QuantizeLinear_Input_bias_array.format |= AI_FMT_FLAG_CONST;
    logits_QuantizeLinear_Input_bias_array.data = AI_PTR(g_small_net_int8_weights_map[0] + 95136);
    logits_QuantizeLinear_Input_bias_array.data_start = AI_PTR(g_small_net_int8_weights_map[0] + 95136);
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
      
      .n_macc            = 183668710,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x75263b16,
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
      
      .n_macc            = 183668710,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x75263b16,
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

