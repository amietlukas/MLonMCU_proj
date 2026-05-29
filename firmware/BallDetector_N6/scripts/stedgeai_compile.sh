#!/usr/bin/env bash
# Regenerate the Neural-ART network code + weight blob from the int8 ONNX
# straight into this project's X-CUBE-AI/App/ directory.
#
# Run this whenever model/int8_ptq_qdq.onnx changes. CubeMX runs the same
# stedgeai step at project-generation time; this script lets you re-roll the
# network without re-opening CubeMX. Output dir matches what CubeMX uses, so
# the next `build.sh` picks the new network.{c,h} + weights up.
#
# Empirically confirmed on X-CUBE-AI 10.2.0 (ST Edge AI Core v2.2.0): this
# path emits network.{c,h} (the LL_ATON C API), NOT stai_network.{c,h}. The
# app calls LL_ATON directly — see Core/Src/balldetector_app.c.

set -euo pipefail

# --- adjust me ----------------------------------------------------------------
XAI_ROOT="${XAI_ROOT:-$HOME/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/10.2.0}"
STEDGEAI="${STEDGEAI:-$XAI_ROOT/Utilities/linux/stedgeai}"
# Neural-ART profile JSON + profile name. The pack ships a working JSON with
# named profiles; n6-allmems-O3 -> stm32n6.mpool (on-chip-only, no hyperRAM),
# which is what the Nucleo-N657X0-Q actually has. If CubeMX generated its own
# user_neuralart.json in the project, point NEURAL_ART_JSON at that instead.
NEURAL_ART_JSON="${NEURAL_ART_JSON:-$XAI_ROOT/scripts/N6_scripts/user_neuralart.json}"
NEURAL_ART_PROFILE="${NEURAL_ART_PROFILE:-n6-allmems-O3}"
# ------------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"

MODEL_ONNX="$PROJ/model/int8_ptq_qdq.onnx"
OUT_DIR="$PROJ/X-CUBE-AI/App"

if [[ ! -x "$STEDGEAI" ]]; then
  echo "stedgeai not found/executable: $STEDGEAI" >&2; exit 1
fi
if [[ ! -f "$NEURAL_ART_JSON" ]]; then
  echo "Neural-ART JSON missing: $NEURAL_ART_JSON" >&2; exit 1
fi
if [[ ! -f "$MODEL_ONNX" ]]; then
  echo "model missing: $MODEL_ONNX" >&2
  echo "run: python scripts/prepare_model.py" >&2; exit 1
fi

mkdir -p "$OUT_DIR"

echo "==> stedgeai generate  (profile: $NEURAL_ART_PROFILE)"
echo "    model : $MODEL_ONNX"
echo "    out   : $OUT_DIR"
# uint8 input (DCMIPP delivers RGB888), int8 outputs (the three YOLO heads).
# --c-api st-ai is harmless here; the N6 path emits LL_ATON regardless.
"$STEDGEAI" generate \
  --model "$MODEL_ONNX" \
  --target stm32n6 \
  --st-neural-art "$NEURAL_ART_PROFILE@$NEURAL_ART_JSON" \
  --c-api st-ai \
  --input-data-type uint8 \
  --output-data-type int8 \
  --output "$OUT_DIR"

echo
echo "==> generated in $OUT_DIR:"
ls -1 "$OUT_DIR"/network.c "$OUT_DIR"/network.h "$OUT_DIR"/network_atonbuf.xSPI2.raw 2>/dev/null | sed 's/^/  /'
echo
echo "weights blob -> flash.sh programs network_atonbuf.xSPI2.raw at 0x70380000"
