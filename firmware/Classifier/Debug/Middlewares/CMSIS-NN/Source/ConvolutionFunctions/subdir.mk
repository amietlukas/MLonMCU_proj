################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s4.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s4.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s4_fast.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_even_s4.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s16.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s4.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s16.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s4.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s16.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s4.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_fast_s16.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s16.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s4.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s16.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s4.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s4_opt.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s16.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s4.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s16.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s4_s16.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_get_buffer_sizes_s8.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_s8.c \
../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_wrapper_s8.c 

OBJS += \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s4.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s4.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s4_fast.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_even_s4.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s16.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s4.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s16.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s4.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s16.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s4.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_fast_s16.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s16.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s4.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s16.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s4.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s4_opt.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s16.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s4.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s16.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s4_s16.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_get_buffer_sizes_s8.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_s8.o \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_wrapper_s8.o 

C_DEPS += \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s4.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s4.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s4_fast.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_even_s4.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s16.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s4.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s16.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s4.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s16.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s4.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_fast_s16.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s16.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s4.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s16.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s4.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s4_opt.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s16.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s4.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s16.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s4_s16.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_get_buffer_sizes_s8.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_s8.d \
./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_wrapper_s8.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/CMSIS-NN/Source/ConvolutionFunctions/%.o Middlewares/CMSIS-NN/Source/ConvolutionFunctions/%.su Middlewares/CMSIS-NN/Source/ConvolutionFunctions/%.cyclo: ../Middlewares/CMSIS-NN/Source/ConvolutionFunctions/%.c Middlewares/CMSIS-NN/Source/ConvolutionFunctions/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m33 -std=gnu11 -g3 -DDEBUG -DARM_MATH_DSP -DUSE_HAL_DRIVER -DSTM32U585xx -c -I../Core/Inc -I../Middlewares/ST/AI/Inc -I../Middlewares/CMSIS-NN/Include -I../Drivers/STM32U5xx_HAL_Driver/Inc -I../Drivers/STM32U5xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32U5xx/Include -I../Drivers/CMSIS/Include -I../X-CUBE-AI/App -I../X-CUBE-AI -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-CMSIS-2d-NN-2f-Source-2f-ConvolutionFunctions

clean-Middlewares-2f-CMSIS-2d-NN-2f-Source-2f-ConvolutionFunctions:
	-$(RM) ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s4.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s4.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s4.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s4.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1_x_n_s8.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s4.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s4.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s4.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s4.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s4_fast.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s4_fast.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s4_fast.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s4_fast.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_1x1_s8_fast.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_even_s4.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_even_s4.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_even_s4.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_even_s4.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s16.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s16.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s16.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s16.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s4.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s4.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s4.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s4.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s16.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s16.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s16.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s16.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s4.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s4.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s4.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s4.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s16.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s16.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s16.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s16.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s4.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s4.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s4.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s4.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_wrapper_s8.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_fast_s16.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_fast_s16.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_fast_s16.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_fast_s16.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s16.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s16.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s16.o
	-$(RM) ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s16.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s4.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s4.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s4.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s4.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s16.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s16.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s16.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s16.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s4.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s4.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s4.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s4.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s4_opt.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s4_opt.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s4_opt.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s4_opt.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_s8_opt.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s16.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s16.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s16.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s16.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s4.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s4.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s4.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s4.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_depthwise_conv_wrapper_s8.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_depthwise_conv_s8_core.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_row_offset_s8_s16.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s16.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s16.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s16.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s16.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s4_s16.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s4_s16.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s4_s16.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s4_s16.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_nn_mat_mult_s8.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_get_buffer_sizes_s8.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_get_buffer_sizes_s8.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_get_buffer_sizes_s8.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_get_buffer_sizes_s8.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_s8.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_s8.d
	-$(RM) ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_s8.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_s8.su ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_wrapper_s8.cyclo ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_wrapper_s8.d ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_wrapper_s8.o ./Middlewares/CMSIS-NN/Source/ConvolutionFunctions/arm_transpose_conv_wrapper_s8.su

.PHONY: clean-Middlewares-2f-CMSIS-2d-NN-2f-Source-2f-ConvolutionFunctions

