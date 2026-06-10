################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s16.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s4.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s16.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s8_s16.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s16.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s8.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s4.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_interleaved_t_even_s4.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s16.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s4.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8_s32.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_transpose_conv_row_s8_s32.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s16.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s8_s16.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_per_ch_s8.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16_s16.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s4.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_svdf_s8.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nntables.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c \
../Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.c 

OBJS += \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s16.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s4.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s16.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s8_s16.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s16.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s8.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s4.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_interleaved_t_even_s4.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s16.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s4.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8_s32.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_transpose_conv_row_s8_s32.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s16.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s8_s16.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_per_ch_s8.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16_s16.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s4.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_svdf_s8.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nntables.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.o \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.o 

C_DEPS += \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s16.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s4.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s16.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s8_s16.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s16.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s8.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s4.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_interleaved_t_even_s4.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s16.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s4.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8_s32.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_transpose_conv_row_s8_s32.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s16.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s8_s16.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_per_ch_s8.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16_s16.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s4.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_svdf_s8.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nntables.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.d \
./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/CMSIS-NN/Source/NNSupportFunctions/%.o Middlewares/CMSIS-NN/Source/NNSupportFunctions/%.su Middlewares/CMSIS-NN/Source/NNSupportFunctions/%.cyclo: ../Middlewares/CMSIS-NN/Source/NNSupportFunctions/%.c Middlewares/CMSIS-NN/Source/NNSupportFunctions/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m33 -std=gnu11 -g3 -DDEBUG -DARM_MATH_DSP -DUSE_HAL_DRIVER -DSTM32U585xx -c -I../Core/Inc -I../Middlewares/ST/AI/Inc -I../Middlewares/CMSIS-NN/Include -I../Drivers/STM32U5xx_HAL_Driver/Inc -I../Drivers/STM32U5xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32U5xx/Include -I../Drivers/CMSIS/Include -I../X-CUBE-AI/App -I../X-CUBE-AI -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-CMSIS-2d-NN-2f-Source-2f-NNSupportFunctions

clean-Middlewares-2f-CMSIS-2d-NN-2f-Source-2f-NNSupportFunctions:
	-$(RM) ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_padded_s8.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s16.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s16.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s16.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s16.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s4.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s4.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s4.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s4.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_depthwise_conv_nt_t_s8.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s16.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s16.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s16.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s16.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s8_s16.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s8_s16.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s8_s16.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_calculate_gate_s8_s16.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s16.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s16.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s16.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s16.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s8.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s8.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s8.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s8.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s4.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s4.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s4.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s4.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_1x_s8.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mul_core_4x_s8.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_interleaved_t_even_s4.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_interleaved_t_even_s4.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_interleaved_t_even_s4.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_interleaved_t_even_s4.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s16.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s16.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s16.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s16.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s4.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s4.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s4.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s4.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8_s32.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8_s32.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8_s32.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8_s32.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_transpose_conv_row_s8_s32.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_transpose_conv_row_s8_s32.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_transpose_conv_row_s8_s32.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_transpose_conv_row_s8_s32.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s16.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s16.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s16.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s16.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s8_s16.cyclo
	-$(RM) ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s8_s16.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s8_s16.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mul_result_acc_s8_s16.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_per_ch_s8.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_per_ch_s8.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_per_ch_s8.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_per_ch_s8.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16_s16.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16_s16.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16_s16.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s16_s16.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s4.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s4.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s4.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s4.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_svdf_s8.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_svdf_s8.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_svdf_s8.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_svdf_s8.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nntables.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nntables.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nntables.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_nntables.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.su ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.cyclo ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.d ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.o ./Middlewares/CMSIS-NN/Source/NNSupportFunctions/arm_s8_to_s16_unordered_with_offset.su

.PHONY: clean-Middlewares-2f-CMSIS-2d-NN-2f-Source-2f-NNSupportFunctions

