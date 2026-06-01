#!/usr/bin/env bash
# Program the Neural-ART weight blob into the external OctoFlash (xSPI2) at the
# address the generated network expects (0x71000000). Needed once (persists
# across power cycles / RAM-boot app reloads) and again whenever the model is
# re-generated. Board must be in DEV boot mode (BOOT1 = 2-3), USB connected.

set -euo pipefail

# --- adjust me ----------------------------------------------------------------
CUBEPROG_BIN="${CUBEPROG_BIN:-/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_2.2.400.202601091506/tools/bin}"
PROG_TOOL="${PROG_TOOL:-$CUBEPROG_BIN/STM32_Programmer_CLI}"
EXT_LOADER="${EXT_LOADER:-$CUBEPROG_BIN/ExternalLoader/MX25UM51245G_STM32N6570-NUCLEO.stldr}"
WEIGHTS_ADDR="${WEIGHTS_ADDR:-0x71000000}"   # == network_prunedint8.c .addr_base
# ------------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"

# Locate the weight blob CubeMX/stedgeai produced (network name = network_prunedint8).
RAW="${RAW:-}"
if [[ -z "$RAW" ]]; then
  RAW="$(find "$PROJ" -name 'network_prunedint8_atonbuf.xSPI2.raw' 2>/dev/null | head -1)"
fi
[[ -x "$PROG_TOOL"  ]] || { echo "STM32_Programmer_CLI not at: $PROG_TOOL" >&2; exit 1; }
[[ -f "$EXT_LOADER" ]] || { echo "external loader missing: $EXT_LOADER" >&2; exit 1; }
[[ -n "$RAW" && -f "$RAW" ]] || {
  echo "weight blob network_prunedint8_atonbuf.xSPI2.raw not found under $PROJ" >&2
  echo "regenerate the network in CubeMX (or run scripts/stedgeai_compile.sh)." >&2; exit 1; }

# STM32_Programmer_CLI's -w rejects the .raw extension (accepts .bin/.hex/...),
# so present the blob as a .bin (identical bytes, just a name the tool likes).
BIN="${RAW%.raw}.bin"
cp -f "$RAW" "$BIN"

echo "==> programming weights"
echo "    blob : $BIN ($(du -h "$BIN" | cut -f1))"
echo "    addr : $WEIGHTS_ADDR (OctoFlash via xSPI2)"
"$PROG_TOOL" -c port=SWD mode=HOTPLUG ap=1 \
  -el "$EXT_LOADER" \
  -w "$BIN" "$WEIGHTS_ADDR" \
  -v

echo
echo "done — weights resident in external flash. Now load+run the app (dev boot)"
echo "and feed images: python firmware/Host/host_balldetector_n6.py --port /dev/ttyACM0 img --images <dir>"
