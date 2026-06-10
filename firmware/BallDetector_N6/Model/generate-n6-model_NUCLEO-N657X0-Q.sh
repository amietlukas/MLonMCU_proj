#!/bin/bash

set -eu # Exit on any error, Exit on unset variable

# Always run from this script's directory (the Model/ folder), so relative
# paths work no matter where it is invoked from.
cd "$(dirname "$0")"

MODEL=${STEDGEAI_MODEL:-balldet_int8.onnx}
NEURAL_ART=${STEDGEAI_NEURAL_ART:-default@user_neuralart_NUCLEO-N657X0-Q.json}

mkdir -p NUCLEO-N657X0-Q

stedgeai generate --model "$MODEL" --target stm32n6 --st-neural-art "$NEURAL_ART" --input-data-type uint8 --output-data-type int8

cp st_ai_output/network.c NUCLEO-N657X0-Q/
cp st_ai_output/network_ecblobs.h NUCLEO-N657X0-Q/
cp st_ai_output/stai_network.c NUCLEO-N657X0-Q/
cp st_ai_output/stai_network.h NUCLEO-N657X0-Q/
cp st_ai_output/network_atonbuf.xSPI2.raw NUCLEO-N657X0-Q/network_data.xSPI2.bin
arm-none-eabi-objcopy -I binary NUCLEO-N657X0-Q/network_data.xSPI2.bin --change-addresses 0x70380000 -O ihex NUCLEO-N657X0-Q/network_data.hex
