 /**
 ******************************************************************************
 * @file    app_camerapipeline.c
 * @author  GPM Application Team
 *
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2023 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

#include <assert.h>
#include "cmw_camera.h"
#include "app_camerapipeline.h"
#include "app_config.h"
#include "crop_img.h"
#if APP_UVC
#include "scrl.h"
#endif
#include "main.h"
#include "stai_network.h"


/* Leave the driver use the default resolution */
#define CAMERA_WIDTH 0
#define CAMERA_HEIGHT 0

extern int32_t cameraFrameReceived;

#if APP_UVC
static void DCMIPP_PipeInitDisplay(CMW_CameraInit_t *camConf, uint32_t *layers_width[2], uint32_t *layers_height[2])
{
  CMW_Aspect_Ratio_Mode_t aspect_ratio;
  CMW_DCMIPP_Conf_t dcmipp_conf = {0};
  int ret;

  if (ASPECT_RATIO_MODE == ASPECT_RATIO_CROP)
  {
    aspect_ratio = CMW_Aspect_ratio_crop;
  }
  else if (ASPECT_RATIO_MODE == ASPECT_RATIO_FIT)
  {
    aspect_ratio = CMW_Aspect_ratio_fit;
  }
  else if (ASPECT_RATIO_MODE == ASPECT_RATIO_FULLSCREEN)
  {
    aspect_ratio = CMW_Aspect_ratio_fullscreen;
  }

  int lcd_bg_width;
  int lcd_bg_height;

  lcd_bg_height = (camConf->height <= SCREEN_HEIGHT) ? camConf->height : SCREEN_HEIGHT;

#if ASPECT_RATIO_MODE == ASPECT_RATIO_FULLSCREEN
  lcd_bg_width = (((camConf->width*lcd_bg_height)/camConf->height) - ((camConf->width*lcd_bg_height)/camConf->height) % 16);
#else
  lcd_bg_width = (camConf->height <= SCREEN_HEIGHT) ? camConf->height : SCREEN_HEIGHT;
#endif

  *layers_width[0] = lcd_bg_width;
  *layers_height[0] = lcd_bg_height;
  *layers_width[1] = lcd_bg_width;
  *layers_height[1] = lcd_bg_height;

  dcmipp_conf.output_width = lcd_bg_width;
  dcmipp_conf.output_height = lcd_bg_height;
  dcmipp_conf.output_format = DCMIPP_PIXEL_PACKER_FORMAT_RGB565_1;
  dcmipp_conf.output_bpp = 2;
  dcmipp_conf.mode = aspect_ratio;
  dcmipp_conf.enable_gamma_conversion = 0;
  uint32_t pitch;
  ret = CMW_CAMERA_SetPipeConfig(DCMIPP_PIPE1, &dcmipp_conf, &pitch);
  assert(ret == HAL_OK);
  assert(dcmipp_conf.output_width * dcmipp_conf.output_bpp == pitch);
}
#endif /* APP_UVC */

static void DCMIPP_PipeInitNn(uint32_t *pitch)
{
  CMW_Aspect_Ratio_Mode_t aspect_ratio;
  CMW_DCMIPP_Conf_t dcmipp_conf;
  int ret;

  if (ASPECT_RATIO_MODE == ASPECT_RATIO_CROP)
  {
    aspect_ratio = CMW_Aspect_ratio_crop;
  }
  else if (ASPECT_RATIO_MODE == ASPECT_RATIO_FIT)
  {
    aspect_ratio = CMW_Aspect_ratio_fit;
  }
  else if (ASPECT_RATIO_MODE == ASPECT_RATIO_FULLSCREEN)
  {
    aspect_ratio = CMW_Aspect_ratio_fit;
  }

#if NN_ROTATE_90
  /* Camera is mounted 90deg rotated: capture in the sensor's native 3:4
   * orientation (model HEIGHT x WIDTH) and rotate 90deg CCW in software (see
   * rotate_ccw_hwc_to_chw). Crop (not fit/stretch) keeps the 4:3 sensor's
   * aspect so balls stay round; main.c rotates this into the 4:3 NN input. */
  dcmipp_conf.output_width = STAI_NETWORK_IN_1_HEIGHT;   /* 288 */
  dcmipp_conf.output_height = STAI_NETWORK_IN_1_WIDTH;   /* 384 */
  dcmipp_conf.mode = CMW_Aspect_ratio_crop;
  (void)aspect_ratio;
#else
  dcmipp_conf.output_width = STAI_NETWORK_IN_1_WIDTH;
  dcmipp_conf.output_height = STAI_NETWORK_IN_1_HEIGHT;
  dcmipp_conf.mode = aspect_ratio;
#endif
  dcmipp_conf.output_format = DCMIPP_PIXEL_PACKER_FORMAT_RGB888_YUV444_1;
  dcmipp_conf.output_bpp = STAI_NETWORK_IN_1_CHANNEL;
  dcmipp_conf.enable_swap = COLOR_MODE;
  dcmipp_conf.enable_gamma_conversion = 0;
  ret = CMW_CAMERA_SetPipeConfig(DCMIPP_PIPE2, &dcmipp_conf, pitch);
  assert(ret == HAL_OK);
}

/**
* @brief Init the camera and the 2 DCMIPP pipes
* @param lcd_bg_width display width
* @param lcd_bg_height display height
* @param pitch_nn output pitch computed by the CMW
*/
void CameraPipeline_Init(uint32_t *layers_width[2], uint32_t *layers_height[2], uint32_t *pitch_nn)
{
  int ret;
  CMW_CameraInit_t cam_conf;

  cam_conf.width = CAMERA_WIDTH;
  cam_conf.height = CAMERA_HEIGHT;
  cam_conf.fps = CAMERA_FPS;
  cam_conf.mirror_flip = CAMERA_FLIP;

  ret = CMW_CAMERA_Init(&cam_conf, NULL);
  assert(ret == CMW_ERROR_NONE);
#if APP_UVC
  DCMIPP_PipeInitDisplay(&cam_conf, layers_width, layers_height);
#else
  (void)layers_width;
  (void)layers_height;
#endif
  DCMIPP_PipeInitNn(pitch_nn);
}

void CameraPipeline_DeInit(void)
{
  int ret;
  ret = CMW_CAMERA_DeInit();
  assert(ret == CMW_ERROR_NONE);
}

void CameraPipeline_DisplayPipe_Start(uint8_t *display_pipe_dst, uint32_t cam_mode)
{
  int ret;
  ret = CMW_CAMERA_Start(DCMIPP_PIPE1, display_pipe_dst, cam_mode);
  assert(ret == CMW_ERROR_NONE);
}

void CameraPipeline_NNPipe_Start(uint8_t *nn_pipe_dst, uint32_t cam_mode)
{
  int ret;

  ret = CMW_CAMERA_Start(DCMIPP_PIPE2, nn_pipe_dst, cam_mode);
  assert(ret == CMW_ERROR_NONE);
}

void CameraPipeline_DisplayPipe_Stop()
{
  int ret;
  ret = CMW_CAMERA_Suspend(DCMIPP_PIPE1);
  assert(ret == CMW_ERROR_NONE);
}

void CameraPipeline_IspUpdate(void)
{
  int ret = CMW_ERROR_NONE;
  ret = CMW_CAMERA_Run();
  assert(ret == CMW_ERROR_NONE);
}

/**
  * @brief  Frame event callback
  * @param  hdcmipp pointer to the DCMIPP handle
  * @retval None
  */
int CMW_CAMERA_PIPE_FrameEventCallback(uint32_t pipe)
{
  switch (pipe)
  {
    case DCMIPP_PIPE2 :
      cameraFrameReceived++;

#if APP_UVC
      {
        int ret = SRCL_Update();
        assert(ret == 0);
      }
#endif
      break;
  }
  return 0;
}

void CMW_CAMERA_PIPE_ErrorCallback(uint32_t pipe)
{
  /* FIXME : Need to tune sensor/ipplug so we can remove this implementation */
}
