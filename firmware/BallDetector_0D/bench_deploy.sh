#!/usr/bin/env bash
# Deploy one final ball-detection model to the BallDetector_0D benchmark firmware.
#
#   generate (stedgeai) -> copy network.c/weights -> build -> sign [-> flash]
#   then prints the host bench command for the 676-image val run.
#
# The YOLO decode is model-agnostic at runtime (per-head scale/zp come from the
# generated STAI_NETWORK_OUT_* macros), so NO source edits are needed per model
# -- as long as the input is 384x288 (the grids/strides are still fixed to that).
#
# Usage:
#   ./bench_deploy.sh <model_run_dir> [--tag NAME] [--flash] [--run]
# Example:
#   ./bench_deploy.sh \
#     ../../software/final_models/ball_detection/20260527-153020-smallimgsize_v1_pruned_30 \
#     --tag smallimgsize_v1_int8 --flash
set -euo pipefail

# ---- paths (edit here if your toolchain moves) ----------------------------
REPO=/mnt/core/MLonMCU_proj
APP="$REPO/firmware/BallDetector_0D/Application/NUCLEO-N657X0-Q"
MODELD="$REPO/firmware/BallDetector_0D/Model"
GCC=/opt/st/stm32cubeide_1.17.0/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.12.3.rel1.linux64_1.1.0.202410170702/tools/bin
STEDGEAI=/opt/ST/STEdgeAI/4.0/Utilities/linux
CUBEPROG=/opt/st/stm32cubeide_1.17.0/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_2.2.0.202409170845/tools/bin
SIGN="$CUBEPROG/STM32_SigningTool_CLI"
PROG="$CUBEPROG/STM32_Programmer_CLI"
NUEL="$CUBEPROG/ExternalLoader/MX25UM51245G_STM32N6570-NUCLEO.stldr"
VAL="$REPO/software/ball_detection/splits/val.txt"
IMGROOT="$REPO/datasets/BALL"
PORT="${PORT:-/dev/ttyACM1}"

# ---- args -----------------------------------------------------------------
[ $# -ge 1 ] || { echo "usage: $0 <model_run_dir> [--tag NAME] [--flash] [--run]"; exit 2; }
MODEL_DIR="$(cd "$1" && pwd)"; shift
TAG=""; DO_FLASH=0; DO_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2;;
    --flash) DO_FLASH=1; shift;;
    --run) DO_RUN=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
[ -n "$TAG" ] || TAG="$(basename "$MODEL_DIR" | sed -E 's/^[0-9]{8}-[0-9]{6}-//')_int8"

ONNX="$MODEL_DIR/exports/int8_ptq_qdq.onnx"
[ -f "$ONNX" ] || { echo "ERROR: no int8 export at $ONNX (fp32-only model? quantize it first)"; exit 1; }

# ---- input-size guard: this firmware/decoder is fixed to 384x288 ----------
W=$(grep -E '^\s*width:'  "$MODEL_DIR/config_snapshot.yaml" | head -1 | grep -oE '[0-9]+' || echo "?")
H=$(grep -E '^\s*height:' "$MODEL_DIR/config_snapshot.yaml" | head -1 | grep -oE '[0-9]+' || echo "?")
if [ "$W" != "384" ] || [ "$H" != "288" ]; then
  echo "ERROR: model input ${W}x${H} != 384x288."
  echo "  The host resize/GT-scaling (MODEL_W/H) and yolo_postproc grids are fixed to 384x288."
  echo "  Deploying ${W}x${H} needs host input-size handling + runtime grids (+ a RAM check)."
  exit 1
fi

echo "=== deploy: $(basename "$MODEL_DIR")  tag=$TAG  (${W}x${H}) ==="
export PATH="$STEDGEAI:$GCC:$PATH"

# ---- 1. generate ----------------------------------------------------------
cd "$MODELD"
echo "[1/4] stedgeai generate ..."
stedgeai generate --model "$ONNX" --target stm32n6 \
  --st-neural-art default@user_neuralart_NUCLEO-N657X0-Q.json \
  --input-data-type uint8 --output-data-type int8 >/tmp/stedgeai_$TAG.log 2>&1 \
  || { echo "stedgeai FAILED — see /tmp/stedgeai_$TAG.log"; tail -20 /tmp/stedgeai_$TAG.log; exit 1; }
cp st_ai_output/network.c          NUCLEO-N657X0-Q/
cp st_ai_output/network_ecblobs.h  NUCLEO-N657X0-Q/
cp st_ai_output/stai_network.c     NUCLEO-N657X0-Q/
cp st_ai_output/stai_network.h     NUCLEO-N657X0-Q/
cp st_ai_output/network_atonbuf.xSPI2.raw NUCLEO-N657X0-Q/network_data.xSPI2.bin
arm-none-eabi-objcopy -I binary NUCLEO-N657X0-Q/network_data.xSPI2.bin \
  --change-addresses 0x70380000 -O ihex NUCLEO-N657X0-Q/network_data.hex
echo "      OUT scales: $(grep -A1 'STAI_NETWORK_OUT_._SCALES' NUCLEO-N657X0-Q/stai_network.h | grep -oE '0\.[0-9]+' | tr '\n' ' ')"

# ---- 2. build -------------------------------------------------------------
cd "$APP"
echo "[2/4] make ..."
make -j4 GCC_PATH="$GCC" >/tmp/make_$TAG.log 2>&1 \
  || { echo "build FAILED — see /tmp/make_$TAG.log"; tail -20 /tmp/make_$TAG.log; exit 1; }

# ---- 3. sign --------------------------------------------------------------
cd "$APP/build/Application/NUCLEO-N657X0-Q"
echo "[3/4] sign ..."
chmod u+w Project_sign.bin 2>/dev/null || true
"$SIGN" -s -bin Project.bin -nk -t ssbl -hv 2.3 -o Project_sign.bin </dev/null >/dev/null 2>&1 \
  || { echo "sign FAILED"; exit 1; }
echo "      signed: $(pwd)/Project_sign.bin"

# ---- 4. flash (optional; board must be in dev mode) -----------------------
if [ "$DO_FLASH" -eq 1 ]; then
  echo "[4/4] flash app + weights (board must be in DEV mode) ..."
  "$PROG" -c port=SWD mode=HOTPLUG -el "$NUEL" -hardRst -w Project_sign.bin 0x70100000
  "$PROG" -c port=SWD mode=HOTPLUG -el "$NUEL" -hardRst -w "$MODELD/NUCLEO-N657X0-Q/network_data.hex"
  echo "      flashed. Set boot back to boot-from-flash + power-cycle."
else
  echo "[4/4] skipped flash (pass --flash; needs dev mode). Manual:"
  echo "  $PROG -c port=SWD mode=HOTPLUG -el \"$NUEL\" -hardRst -w $(pwd)/Project_sign.bin 0x70100000"
  echo "  $PROG -c port=SWD mode=HOTPLUG -el \"$NUEL\" -hardRst -w $MODELD/NUCLEO-N657X0-Q/network_data.hex"
fi

# ---- host run command -----------------------------------------------------
RUNCMD="python3 $REPO/firmware/Host/host_balldetector_n6.py --port $PORT bench \
  --image-list $VAL --image-root $IMGROOT \
  --model-name $TAG --out-dir $MODEL_DIR --chunk-delay 0.002"
echo ""
echo "=== host val run (after flashing + power-cycle) ==="
echo "$RUNCMD"
if [ "$DO_RUN" -eq 1 ]; then
  echo "--- running now ---"
  eval "$RUNCMD"
fi
