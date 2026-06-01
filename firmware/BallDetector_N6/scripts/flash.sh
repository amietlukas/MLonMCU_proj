#!/usr/bin/env bash
# Sign the app binary and flash app + weights to the Nucleo-N657X0-Q external
# OctoFlash (xSPI2). The N6 boots signed from external flash, so the app must
# be signed; the Neural-ART weight blob is programmed raw at its base address.
#
#   app (signed) -> 0x70000000
#   weights blob -> 0x71000000   (network_prunedint8_atonbuf.xSPI2.raw)
#
# NOTE: this must match the absolute xSPI2 base emitted in
# FSBL/X-CUBE-AI/App/network_prunedint8.c.

set -euo pipefail

# --- adjust me ----------------------------------------------------------------
CUBEPROG_BIN="${CUBEPROG_BIN:-/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_2.2.400.202601091506/tools/bin}"
SIGNING_TOOL="${SIGNING_TOOL:-$CUBEPROG_BIN/STM32_SigningTool_CLI}"
PROG_TOOL="${PROG_TOOL:-$CUBEPROG_BIN/STM32_Programmer_CLI}"
EXT_LOADER="${EXT_LOADER:-$CUBEPROG_BIN/ExternalLoader/MX25UM51245G_STM32N6570-NUCLEO.stldr}"
CONFIG="${CONFIG:-Debug}"
PROJ_NAME="${PROJ_NAME:-BallDetector_N6_FSBL}"
APP_ADDR="${APP_ADDR:-0x70000000}"
WEIGHTS_ADDR="${WEIGHTS_ADDR:-0x71000000}"
# ------------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
FSBL="$PROJ/FSBL"

APP_BIN="$FSBL/$CONFIG/$PROJ_NAME.bin"
APP_SIGNED="$FSBL/$CONFIG/$PROJ_NAME-signed.bin"
WEIGHTS_RAW="$PROJ/network_prunedint8_atonbuf.xSPI2.raw"
WEIGHTS_BIN="${WEIGHTS_RAW%.raw}.bin"

for f in "$SIGNING_TOOL" "$PROG_TOOL"; do
  [[ -x "$f" ]] || { echo "missing tool: $f" >&2; exit 1; }
done
[[ -f "$EXT_LOADER" ]] || { echo "external loader missing: $EXT_LOADER" >&2; exit 1; }
[[ -f "$APP_BIN"   ]] || { echo "app binary missing: $APP_BIN — run scripts/build.sh" >&2; exit 1; }
[[ -f "$WEIGHTS_RAW" ]] || { echo "weights blob missing: $WEIGHTS_RAW — run scripts/stedgeai_compile.sh" >&2; exit 1; }
cp -f "$WEIGHTS_RAW" "$WEIGHTS_BIN"
rm -f "$APP_SIGNED"

echo "==> sign app  ($APP_BIN -> $APP_SIGNED)"
"$SIGNING_TOOL" -bin "$APP_BIN" -nk -t ssbl -hv 2.3 -o "$APP_SIGNED"

echo
echo "==> flash app (signed) @ $APP_ADDR  +  weights @ $WEIGHTS_ADDR"
"$PROG_TOOL" -c port=SWD mode=HOTPLUG -el "$EXT_LOADER" \
  -w "$APP_SIGNED"   "$APP_ADDR" \
  -w "$WEIGHTS_BIN"  "$WEIGHTS_ADDR" \
  -hardRst

echo
echo "done — open the host: python firmware/Host/host_balldetector_n6.py --port /dev/ttyACM0 cam"
