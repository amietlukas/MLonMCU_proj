/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __APP_AI_H
#define __APP_AI_H

#ifdef __cplusplus
extern "C" {
#endif

/**
  ******************************************************************************
  * @file    app_x-cube-ai.h
  * @brief   AI entry function definitions
  *
  *          Backend selection (set ONE in your build defines or constants.h):
  *            USE_BACKEND_FP32        — X-CUBE-AI FP32  (default)
  *            USE_BACKEND_INT8_XCUBE  — X-CUBE-AI INT8
  *            USE_BACKEND_INT8_CMSIS  — CMSIS-NN  INT8
  ******************************************************************************
  */

void MX_X_CUBE_AI_Init(void);
void MX_X_CUBE_AI_Process(void);

#ifdef __cplusplus
}
#endif

#endif /* __APP_AI_H */
