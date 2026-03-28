################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_nn_activation_s16.c \
../Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu6_s8.c \
../Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu_q15.c \
../Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu_q7.c 

OBJS += \
./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_nn_activation_s16.o \
./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu6_s8.o \
./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu_q15.o \
./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu_q7.o 

C_DEPS += \
./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_nn_activation_s16.d \
./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu6_s8.d \
./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu_q15.d \
./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu_q7.d 


# Each subdirectory must supply rules for building sources it contributes
Middlewares/CMSIS-NN/Source/ActivationFunctions/%.o Middlewares/CMSIS-NN/Source/ActivationFunctions/%.su Middlewares/CMSIS-NN/Source/ActivationFunctions/%.cyclo: ../Middlewares/CMSIS-NN/Source/ActivationFunctions/%.c Middlewares/CMSIS-NN/Source/ActivationFunctions/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m33 -std=gnu11 -g3 -DDEBUG -DARM_MATH_DSP -DUSE_HAL_DRIVER -DSTM32U585xx -c -I../Core/Inc -I../Middlewares/ST/AI/Inc -I../Middlewares/CMSIS-NN/Include -I../Drivers/STM32U5xx_HAL_Driver/Inc -I../Drivers/STM32U5xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32U5xx/Include -I../Drivers/CMSIS/Include -I../X-CUBE-AI/App -I../X-CUBE-AI -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Middlewares-2f-CMSIS-2d-NN-2f-Source-2f-ActivationFunctions

clean-Middlewares-2f-CMSIS-2d-NN-2f-Source-2f-ActivationFunctions:
	-$(RM) ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_nn_activation_s16.cyclo ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_nn_activation_s16.d ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_nn_activation_s16.o ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_nn_activation_s16.su ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu6_s8.cyclo ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu6_s8.d ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu6_s8.o ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu6_s8.su ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu_q15.cyclo ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu_q15.d ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu_q15.o ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu_q15.su ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu_q7.cyclo ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu_q7.d ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu_q7.o ./Middlewares/CMSIS-NN/Source/ActivationFunctions/arm_relu_q7.su

.PHONY: clean-Middlewares-2f-CMSIS-2d-NN-2f-Source-2f-ActivationFunctions

