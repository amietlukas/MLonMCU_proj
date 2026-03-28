################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../X-CUBE-AI/App/app_x-cube-ai.c \
../X-CUBE-AI/App/small_net_fp32.c \
../X-CUBE-AI/App/small_net_fp32_data.c \
../X-CUBE-AI/App/small_net_fp32_data_params.c \
../X-CUBE-AI/App/small_net_int8.c \
../X-CUBE-AI/App/small_net_int8_data.c \
../X-CUBE-AI/App/small_net_int8_data_params.c 

OBJS += \
./X-CUBE-AI/App/app_x-cube-ai.o \
./X-CUBE-AI/App/small_net_fp32.o \
./X-CUBE-AI/App/small_net_fp32_data.o \
./X-CUBE-AI/App/small_net_fp32_data_params.o \
./X-CUBE-AI/App/small_net_int8.o \
./X-CUBE-AI/App/small_net_int8_data.o \
./X-CUBE-AI/App/small_net_int8_data_params.o 

C_DEPS += \
./X-CUBE-AI/App/app_x-cube-ai.d \
./X-CUBE-AI/App/small_net_fp32.d \
./X-CUBE-AI/App/small_net_fp32_data.d \
./X-CUBE-AI/App/small_net_fp32_data_params.d \
./X-CUBE-AI/App/small_net_int8.d \
./X-CUBE-AI/App/small_net_int8_data.d \
./X-CUBE-AI/App/small_net_int8_data_params.d 


# Each subdirectory must supply rules for building sources it contributes
X-CUBE-AI/App/%.o X-CUBE-AI/App/%.su X-CUBE-AI/App/%.cyclo: ../X-CUBE-AI/App/%.c X-CUBE-AI/App/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m33 -std=gnu11 -g3 -DDEBUG -DARM_MATH_DSP -DUSE_HAL_DRIVER -DSTM32U585xx -c -I../Core/Inc -I../Middlewares/ST/AI/Inc -I../Middlewares/CMSIS-NN/Include -I../Drivers/STM32U5xx_HAL_Driver/Inc -I../Drivers/STM32U5xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32U5xx/Include -I../Drivers/CMSIS/Include -I../X-CUBE-AI/App -I../X-CUBE-AI -Ofast -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-X-2d-CUBE-2d-AI-2f-App

clean-X-2d-CUBE-2d-AI-2f-App:
	-$(RM) ./X-CUBE-AI/App/app_x-cube-ai.cyclo ./X-CUBE-AI/App/app_x-cube-ai.d ./X-CUBE-AI/App/app_x-cube-ai.o ./X-CUBE-AI/App/app_x-cube-ai.su ./X-CUBE-AI/App/small_net_fp32.cyclo ./X-CUBE-AI/App/small_net_fp32.d ./X-CUBE-AI/App/small_net_fp32.o ./X-CUBE-AI/App/small_net_fp32.su ./X-CUBE-AI/App/small_net_fp32_data.cyclo ./X-CUBE-AI/App/small_net_fp32_data.d ./X-CUBE-AI/App/small_net_fp32_data.o ./X-CUBE-AI/App/small_net_fp32_data.su ./X-CUBE-AI/App/small_net_fp32_data_params.cyclo ./X-CUBE-AI/App/small_net_fp32_data_params.d ./X-CUBE-AI/App/small_net_fp32_data_params.o ./X-CUBE-AI/App/small_net_fp32_data_params.su ./X-CUBE-AI/App/small_net_int8.cyclo ./X-CUBE-AI/App/small_net_int8.d ./X-CUBE-AI/App/small_net_int8.o ./X-CUBE-AI/App/small_net_int8.su ./X-CUBE-AI/App/small_net_int8_data.cyclo ./X-CUBE-AI/App/small_net_int8_data.d ./X-CUBE-AI/App/small_net_int8_data.o ./X-CUBE-AI/App/small_net_int8_data.su ./X-CUBE-AI/App/small_net_int8_data_params.cyclo ./X-CUBE-AI/App/small_net_int8_data_params.d ./X-CUBE-AI/App/small_net_int8_data_params.o ./X-CUBE-AI/App/small_net_int8_data_params.su

.PHONY: clean-X-2d-CUBE-2d-AI-2f-App

