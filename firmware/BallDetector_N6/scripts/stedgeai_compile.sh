#!/usr/bin/env bash
# Compile the int8 ball-detector ONNX for Neural-ART and drop the
# generated files into the ST reference app's Model directory, ready
# for `make`. Mirrors ST's own Model/generate-n6-model_NUCLEO-N657X0-Q.sh.

set -euo pipefail

# --- adjust me ----------------------------------------------------------------
STEDGEAI="${STEDGEAI:-$HOME/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0/Utilities/linux/stedgeai}"
GCC_BIN="${GCC_BIN:-/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.linux64_1.0.100.202602081740/tools/bin}"

# Weights base address in external OSPI — must match Application linker
# script (Application/NUCLEO-N657X0-Q/STM32CubeIDE/STM32N657xx.ld).
# ST's reference uses 0x70380000.
WEIGHTS_ADDR="${WEIGHTS_ADDR:-0x70380000}"
# ------------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$PROJ/../.." && pwd)"

ST_APP="$PROJ/stm32ai-modelzoo-services/application_code/object_detection/STM32N6"
MODEL_DIR_REL="Model/NUCLEO-N657X0-Q"
NEURAL_ART_JSON="$ST_APP/Model/user_neuralart_NUCLEO-N657X0-Q.json"

MODEL_ONNX="$REPO/software/ball_detector_n6/model/int8_ptq_qdq.onnx"
WORK_DIR="$PROJ/build/st_ai_output"
DST_DIR="$ST_APP/Model/NUCLEO-N657X0-Q"

if [[ ! -x "$STEDGEAI" ]]; then
  echo "stedgeai not found: $STEDGEAI" >&2; exit 1
fi
if [[ ! -d "$ST_APP" ]]; then
  echo "ST reference app missing: $ST_APP" >&2
  echo "init the submodule:  cd stm32ai-modelzoo-services && git submodule update --init application_code/object_detection/STM32N6" >&2
  exit 1
fi
if [[ ! -f "$NEURAL_ART_JSON" ]]; then
  echo "Neural-ART JSON missing: $NEURAL_ART_JSON" >&2; exit 1
fi
if [[ ! -f "$MODEL_ONNX" ]]; then
  echo "model missing: $MODEL_ONNX" >&2
  echo "run: python software/ball_detector_n6/prepare_model.py" >&2; exit 1
fi
if [[ ! -x "$GCC_BIN/arm-none-eabi-objcopy" ]]; then
  echo "arm-none-eabi-objcopy not found at: $GCC_BIN" >&2; exit 1
fi

mkdir -p "$WORK_DIR" "$DST_DIR"

echo "==> stedgeai generate (ST recipe: epoch-controller, int8 activations)"
( cd "$WORK_DIR" && \
  "$STEDGEAI" generate \
    --model "$MODEL_ONNX" \
    --target stm32n6 \
    --st-neural-art "default@$NEURAL_ART_JSON" \
    --input-data-type uint8 \
    --output-data-type int8 \
    --output . )

echo
echo "==> copy generated files into $MODEL_DIR_REL/"
for f in network.c network_ecblobs.h stai_network.c stai_network.h; do
  cp -v "$WORK_DIR/$f" "$DST_DIR/$f"
done
cp -v "$WORK_DIR/network_atonbuf.xSPI2.raw" "$DST_DIR/network_data.xSPI2.bin"

echo
echo "==> convert weights to ihex at $WEIGHTS_ADDR"
"$GCC_BIN/arm-none-eabi-objcopy" \
  -I binary "$DST_DIR/network_data.xSPI2.bin" \
  --change-addresses "$WEIGHTS_ADDR" \
  -O ihex "$DST_DIR/network_data.hex"

echo
echo "ready in $DST_DIR:"
ls -1 "$DST_DIR" | sed 's/^/  /'
