#!/usr/bin/env bash
# Generate, build, sign and (optionally) flash a ball-detection model onto the
# BallDetector_N6 firmware.
#
#   generate (stedgeai) -> copy network.c/weights -> build -> sign [-> flash]
#
# The YOLO decode is model-agnostic at runtime (per-head scale/zp come from the
# generated STAI_NETWORK_OUT_* macros), so NO source edits are needed per model
# -- as long as the input is 384x288.
#
# Usage:
#   ./deploy.sh <onnx_path> [--flash]
# Example:
#   ./deploy.sh Model/balldet_int8.onnx --flash
#
# Toolchain locations are auto-detected from PATH, but you can override any of
# them via environment variables (point at the *bin* dir or the tool itself):
#   GCC          arm-none-eabi GCC bin dir   (else `arm-none-eabi-gcc` on PATH)
#   STEDGEAI     ST Edge AI bin dir          (else `stedgeai` on PATH)
#   CUBEPROG     STM32CubeProgrammer bin dir (else STM32_Programmer_CLI on PATH)
#   SIGN / PROG  full path to signing tool / programmer (else derived/CUBEPROG)
#   NUEL         external-loader .stldr for the MX25UM51245G flash
# Example:
#   STEDGEAI=/opt/ST/STEdgeAI/4.0/Utilities/linux ./deploy.sh Model/balldet_int8.onnx
set -euo pipefail

# ---- repo paths (derived from this script's location) ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # .../firmware/BallDetector_N6
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP="$SCRIPT_DIR/Application/NUCLEO-N657X0-Q"
MODELD="$SCRIPT_DIR/Model"

# ---- toolchain (env-overridable; auto-detected from PATH otherwise) --------
# Resolve a tool to its containing bin dir: prefer $1 if set, else `which`.
resolve_dir() {  # <env_value> <tool_on_path>
  if [ -n "$1" ]; then echo "$1"; else
    local p; p="$(command -v "$2" 2>/dev/null || true)"; [ -n "$p" ] && dirname "$p" || echo ""
  fi
}
GCC="$(resolve_dir "${GCC:-}" arm-none-eabi-gcc)"
STEDGEAI="$(resolve_dir "${STEDGEAI:-}" stedgeai)"
CUBEPROG="$(resolve_dir "${CUBEPROG:-}" STM32_Programmer_CLI)"
SIGN="${SIGN:-${CUBEPROG:+$CUBEPROG/STM32_SigningTool_CLI}}"
PROG="${PROG:-${CUBEPROG:+$CUBEPROG/STM32_Programmer_CLI}}"
# External loader for the on-board MX25UM51245G flash. Ships with CubeProgrammer.
NUEL="${NUEL:-${CUBEPROG:+$CUBEPROG/ExternalLoader/MX25UM51245G_STM32N6570-NUCLEO.stldr}}"

# ---- preflight: fail early with a helpful message if a tool is missing -----
[ -n "$STEDGEAI" ] && [ -x "$STEDGEAI/stedgeai" ] || {
  echo "ERROR: stedgeai not found. Install ST Edge AI and add it to PATH, or set STEDGEAI=<bin dir>."; exit 1; }
[ -n "$GCC" ] && [ -x "$GCC/arm-none-eabi-gcc" ] || {
  echo "ERROR: arm-none-eabi-gcc not found. Use the STM32CubeIDE GCC 12.3 toolchain (set GCC=<bin dir>)."; exit 1; }

# ---- args -----------------------------------------------------------------
[ $# -ge 1 ] || { echo "usage: $0 <onnx_path> [--flash]"; exit 2; }
ONNX="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
DO_FLASH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --flash) DO_FLASH=1; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done
[ -f "$ONNX" ] || { echo "ERROR: no ONNX at $ONNX"; exit 1; }

echo "=== deploy: $(basename "$ONNX") ==="
export PATH="$STEDGEAI:$GCC:$PATH"

# ---- 1. generate ----------------------------------------------------------
cd "$MODELD"
mkdir -p NUCLEO-N657X0-Q
echo "[1/4] stedgeai generate ..."
stedgeai generate --model "$ONNX" --target stm32n6 \
  --st-neural-art default@user_neuralart_NUCLEO-N657X0-Q.json \
  --input-data-type uint8 --output-data-type int8 >/tmp/stedgeai_deploy.log 2>&1 \
  || { echo "stedgeai FAILED — see /tmp/stedgeai_deploy.log"; tail -20 /tmp/stedgeai_deploy.log; exit 1; }
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
make -j4 GCC_PATH="$GCC" >/tmp/make_deploy.log 2>&1 \
  || { echo "build FAILED — see /tmp/make_deploy.log"; tail -20 /tmp/make_deploy.log; exit 1; }

# ---- 3. sign --------------------------------------------------------------
[ -n "$SIGN" ] && [ -x "$SIGN" ] || {
  echo "ERROR: STM32_SigningTool_CLI not found. Install STM32CubeProgrammer (set CUBEPROG=<bin dir> or SIGN=<full path>)."; exit 1; }
cd "$APP/build/Application/NUCLEO-N657X0-Q"
echo "[3/4] sign ..."
chmod u+w Project_sign.bin 2>/dev/null || true
"$SIGN" -s -bin Project.bin -nk -t ssbl -hv 2.3 -o Project_sign.bin </dev/null >/dev/null 2>&1 \
  || { echo "sign FAILED"; exit 1; }
echo "      signed: $(pwd)/Project_sign.bin"

# ---- 4. flash (optional; board must be in dev mode) -----------------------
if [ "$DO_FLASH" -eq 1 ]; then
  { [ -n "$PROG" ] && [ -x "$PROG" ]; } || {
    echo "ERROR: STM32_Programmer_CLI not found. Install STM32CubeProgrammer (set CUBEPROG=<bin dir> or PROG=<full path>)."; exit 1; }
  [ -f "$NUEL" ] || {
    echo "ERROR: external loader not found at: $NUEL"
    echo "       Set NUEL=<path to MX25UM51245G_STM32N6570-NUCLEO.stldr> (ships with CubeProgrammer)."; exit 1; }
  echo "[4/4] flash app + weights (board must be in DEV mode) ..."
  "$PROG" -c port=SWD mode=HOTPLUG -el "$NUEL" -hardRst -w Project_sign.bin 0x70100000
  "$PROG" -c port=SWD mode=HOTPLUG -el "$NUEL" -hardRst -w "$MODELD/NUCLEO-N657X0-Q/network_data.hex"
  echo "      flashed. Set boot back to boot-from-flash + power-cycle."
else
  echo "[4/4] skipped flash (pass --flash; needs dev mode). Manual:"
  echo "  $PROG -c port=SWD mode=HOTPLUG -el \"$NUEL\" -hardRst -w $(pwd)/Project_sign.bin 0x70100000"
  echo "  $PROG -c port=SWD mode=HOTPLUG -el \"$NUEL\" -hardRst -w $MODELD/NUCLEO-N657X0-Q/network_data.hex"
fi
