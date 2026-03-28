################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_batch_matmul_s16.c \
../Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_batch_matmul_s8.c \
../Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s16.c \
../Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s8.c \
../Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_per_channel_s8.c \
../Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s16.c \
../Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s4.c \
../Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c \
../Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_wrapper_s8.c \
../Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_vector_sum_s8.c \
../Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_vector_sum_s8_s64.c 

OBJS += \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_batch_matmul_s16.o \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_batch_matmul_s8.o \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s16.o \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s8.o \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_per_channel_s8.o \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s16.o \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s4.o \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.o \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_wrapper_s8.o \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_vector_sum_s8.o \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_vector_sum_s8_s64.o 

C_DEPS += \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_batch_matmul_s16.d \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_batch_matmul_s8.d \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s16.d \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s8.d \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_per_channel_s8.d \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s16.d \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s4.d \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.d \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_wrapper_s8.d \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_vector_sum_s8.d \
./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_vector_sum_s8_s64.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/%.o Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/%.su Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/%.cyclo: ../Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/%.c Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m33 -std=gnu11 -g3 -DDEBUG -DARM_MATH_DSP -DUSE_HAL_DRIVER -DSTM32U585xx -c -I../Core/Inc -I../Middlewares/ST/AI/Inc -I../Middlewares/CMSIS-NN/Include -I../Drivers/STM32U5xx_HAL_Driver/Inc -I../Drivers/STM32U5xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32U5xx/Include -I../Drivers/CMSIS/Include -I../X-CUBE-AI/App -I../X-CUBE-AI -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-CMSIS-2d-NN-2f-Source-2f-FullyConnectedFunctions

clean-Middlewares-2f-CMSIS-2d-NN-2f-Source-2f-FullyConnectedFunctions:
	-$(RM) ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_batch_matmul_s16.cyclo ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_batch_matmul_s16.d ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_batch_matmul_s16.o ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_batch_matmul_s16.su ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_batch_matmul_s8.cyclo ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_batch_matmul_s8.d ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_batch_matmul_s8.o ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_batch_matmul_s8.su ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s16.cyclo ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s16.d ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s16.o ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s16.su ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s8.cyclo ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s8.d ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s8.o ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_get_buffer_sizes_s8.su ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_per_channel_s8.cyclo ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_per_channel_s8.d ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_per_channel_s8.o ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_per_channel_s8.su ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s16.cyclo ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s16.d ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s16.o ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s16.su ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s4.cyclo ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s4.d ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s4.o ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s4.su ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.cyclo ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.d ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.o ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.su ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_wrapper_s8.cyclo ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_wrapper_s8.d ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_wrapper_s8.o ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_fully_connected_wrapper_s8.su ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_vector_sum_s8.cyclo ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_vector_sum_s8.d ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_vector_sum_s8.o ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_vector_sum_s8.su ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_vector_sum_s8_s64.cyclo ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_vector_sum_s8_s64.d ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_vector_sum_s8_s64.o ./Middlewares/CMSIS-NN/Source/FullyConnectedFunctions/arm_vector_sum_s8_s64.su

.PHONY: clean-Middlewares-2f-CMSIS-2d-NN-2f-Source-2f-FullyConnectedFunctions

