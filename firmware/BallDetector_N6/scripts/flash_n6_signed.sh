#!/usr/bin/env bash
# Flash signed app + weights using ST's Makefile targets:
#   make flash          - programs Project_sign.bin to 0x70100000
#   make flash_weights  - programs network_data.hex (which encodes 0x70380000)
#
# Requires `make all sign` to have run (build_n6.sh does that) and a fresh
# model/weights in ../Model/NUCLEO-N657X0-Q/network_data.hex
# (stedgeai_compile.sh produces those).

set -euo pipefail

# --- adjust me ----------------------------------------------------------------
GCC_BIN="${GCC_BIN:-/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.linux64_1.0.100.202602081740/tools/bin}"
CUBEPROG_BIN="${CUBEPROG_BIN:-/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_2.2.400.202601091506/tools/bin}"
# ------------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
APP_DIR="$PROJ/stm32ai-modelzoo-services/application_code/object_detection/STM32N6/Application/NUCLEO-N657X0-Q"

if [[ ! -x "$CUBEPROG_BIN/STM32_Programmer_CLI" ]]; then
  echo "STM32_Programmer_CLI not at: $CUBEPROG_BIN" >&2; exit 1
fi

export PATH="$GCC_BIN:$CUBEPROG_BIN:$PATH"

echo "==> make flash         (Project_sign.bin -> 0x70100000)"
make -C "$APP_DIR" flash

echo
echo "==> make flash_weights (network_data.hex)"
make -C "$APP_DIR" flash_weights
