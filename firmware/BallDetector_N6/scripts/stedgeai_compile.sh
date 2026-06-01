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
# Neural-ART profile JSON + profile name.
#   n6-noextmem-fsbl (DEFAULT) -> model/stm32n6__noextmem_fsbl.mpool: internal
#     AXISRAM only (NO XSPI2 in the NPU path), with AXISRAM2 capped to 512K so the
#     NPU stays below 0x34180000 and never collides with the FSBL code/data
#     (ROM 0x34180400, RAM 0x341C0000). Weights land as 3 raw blobs in AXISRAM3/4/5.
#   n6-allmems-O3 (legacy) -> pack json, weights in XSPI2 @0x71000000.
# Override either via the env vars to switch back.
NEURAL_ART_JSON="${NEURAL_ART_JSON:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/model/user_neuralart_fsbl.json}"
NEURAL_ART_PROFILE="${NEURAL_ART_PROFILE:-n6-noextmem-fsbl}"
# ------------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"

MODEL_ONNX="$PROJ/model/int8_ptq_qdq.onnx"
# The buildable project is the FSBL sub-project; its X-CUBE-AI/App is what the
# build compiles. (Top-level X-CUBE-AI/App is NOT in the build.)
OUT_DIR="$PROJ/FSBL/X-CUBE-AI/App"

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
# --name MUST match the CubeMX network name so the emitted network_prunedint8.*
# overwrite the project's (CubeMX uses the .ioc network name "network_prunedint8").
# --c-api st-ai is harmless here; the N6 path emits LL_ATON regardless.
NET_NAME="${NET_NAME:-network_prunedint8}"
"$STEDGEAI" generate \
  --model "$MODEL_ONNX" \
  --name "$NET_NAME" \
  --target stm32n6 \
  --st-neural-art "$NEURAL_ART_PROFILE@$NEURAL_ART_JSON" \
  --c-api st-ai \
  --input-data-type uint8 \
  --output-data-type int8 \
  --output "$OUT_DIR"

echo
echo "==> generated in $OUT_DIR:"
ls -1 "$OUT_DIR/$NET_NAME.c" "$OUT_DIR/$NET_NAME.h" "$OUT_DIR/${NET_NAME}_atonbuf."*.raw 2>/dev/null | sed 's/^/  /'
echo
echo "weight blobs -> flash with scripts/flash_weights.sh:"
echo "  internal profile (n6-noextmem-fsbl): 3 blobs AXISRAM3/4/5 -> XSPI2 staging, copied to AXISRAM at boot."
echo "  legacy profile   (n6-allmems-O3):    1 blob  xSPI2        -> XSPI2 @0x71000000, NPU reads it directly."
echo "NOTE: stedgeai_compile.sh is authoritative — CubeMX's NPU profile is overwritten by this. Re-run after any model or profile change."
