#ifndef CONSTANTS_H
#define CONSTANTS_H

// MODEL CONFIGURATION
#define NUM_CLASSES        6


// INPUT / IMAGE SETTINGS
#define IN_W               320
#define IN_H               240

#define OUT_SIZE           128   // model input size

#define CHANNELS           3
#define INPUT_IMAGE_SIZE   (IN_W * IN_H * CHANNELS)
#define MODEL_INPUT_SIZE   (OUT_SIZE * OUT_SIZE * CHANNELS)


// PREPROCESSING
#define SCALE              0.4f
#define PAD_Y              16

// Normalization (from training)
static const float norm_mean[3] = {0.3912f, 0.3612f, 0.3425f};
static const float norm_std[3]  = {0.3270f, 0.3131f, 0.3042f};


// UART PROTOCOL
#define UART_TIMEOUT       HAL_MAX_DELAY


// DEBUG / CONTROL
#define SEND_FULL_LOGITS   0   // 1 = send all 6 outputs over UART
#define ENABLE_TIMING      1


#endif
