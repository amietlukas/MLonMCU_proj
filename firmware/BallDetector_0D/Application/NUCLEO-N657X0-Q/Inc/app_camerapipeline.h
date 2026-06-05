 /**
 ******************************************************************************
 * @file    app_camerapipeline.h
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
#ifndef APP_CAMERAPIPELINE
#define APP_CAMERAPIPELINE

#define CAMERA_FPS 30
/* UVC/display resolution. Matched to the model's 4:3 working resolution
 * (384x288) so the streamed video has the same field of view / aspect the NN
 * sees. The display pipe is configured FULLSCREEN from the 4:3 sensor, so
 * lcd_bg_width = SCREEN_HEIGHT*4/3 = 384 lines up exactly (multiple of 16).
 * Buffers grow 1.44x vs 320x240 (~864KB of the 1647KB AXISRAM1_S). */
#define SCREEN_HEIGHT (288)
#define SCREEN_WIDTH  (384)

void CameraPipeline_Init(uint32_t *layers_width[2], uint32_t *layers_height[2], uint32_t *pitch_nn);
void CameraPipeline_DeInit(void);
void CameraPipeline_Start(void);
void CameraPipeline_DisplayPipe_Start(uint8_t *display_pipe_dst, uint32_t cam_mode);
void CameraPipeline_DisplayPipe_Stop(void);
void CameraPipeline_NNPipe_Start(uint8_t *nn_pipe_dst, uint32_t cam_mode);
void CameraPipeline_IspUpdate(void);

#endif