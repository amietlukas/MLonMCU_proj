#!/usr/bin/env bash
# Stage the Neural-ART weight blobs in external OctoFlash (xSPI2). With the
# internal-only profile (n6-noextmem-fsbl) the NPU reads weights from AXISRAM,
# not flash — so balldetector_init copies these XSPI2 staging regions into
# AXISRAM3/4/5 at boot. We keep them in flash only so they survive power cycles
# (the alternative would be re-loading 1.3MB via the debugger every session).
# The 3 blobs and their XSPI2 staging offsets MUST match the memcpy in
# balldetector_app.c (copy_weights_to_axisram). Board in DEV boot, USB connected.
#
# Legacy n6-allmems-O3 profile: set RAW=<...xSPI2.raw> WEIGHTS_ADDR=0x71000000 and
# this still flashes the single direct-read blob.

set -euo pipefail

# --- adjust me ----------------------------------------------------------------
CUBEPROG_BIN="${CUBEPROG_BIN:-/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_2.2.400.202601091506/tools/bin}"
PROG_TOOL="${PROG_TOOL:-$CUBEPROG_BIN/STM32_Programmer_CLI}"
EXT_LOADER="${EXT_LOADER:-$CUBEPROG_BIN/ExternalLoader/MX25UM51245G_STM32N6570-NUCLEO.stldr}"
# ------------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
APP="$PROJ/FSBL/X-CUBE-AI/App"

[[ -x "$PROG_TOOL"  ]] || { echo "STM32_Programmer_CLI not at: $PROG_TOOL" >&2; exit 1; }
[[ -f "$EXT_LOADER" ]] || { echo "external loader missing: $EXT_LOADER" >&2; exit 1; }

# blob name  ->  XSPI2 staging address  (must equal copy_weights_to_axisram())
#   AXISRAM3 blob -> 0x71000000 -> copied to 0x34200000
#   AXISRAM4 blob -> 0x71070000 -> copied to 0x34270000  (+448K)
#   AXISRAM5 blob -> 0x710E0000 -> copied to 0x342E0000  (+896K)
BLOBS=( "AXISRAM3:0x71000000" "AXISRAM4:0x71070000" "AXISRAM5:0x710E0000" )

# Legacy single-blob override (n6-allmems-O3 direct-read from XSPI2).
if [[ -n "${RAW:-}" ]]; then
  BLOBS=( )
  BIN="${RAW%.raw}.bin"; cp -f "$RAW" "$BIN"
  echo "==> programming legacy weights $BIN -> ${WEIGHTS_ADDR:-0x71000000}"
  "$PROG_TOOL" -c port=SWD mode=HOTPLUG ap=1 -el "$EXT_LOADER" -w "$BIN" "${WEIGHTS_ADDR:-0x71000000}" -v
  exit 0
fi

for entry in "${BLOBS[@]}"; do
  tag="${entry%%:*}"; addr="${entry##*:}"
  raw="$APP/network_prunedint8_atonbuf.${tag}.raw"
  [[ -f "$raw" ]] || { echo "missing blob: $raw (run scripts/stedgeai_compile.sh)" >&2; exit 1; }
  # STM32_Programmer_CLI's -w rejects .raw; present as .bin (identical bytes).
  bin="${raw%.raw}.bin"; cp -f "$raw" "$bin"
  echo "==> programming $tag : $(du -h "$bin" | cut -f1) -> $addr (XSPI2 staging)"
  "$PROG_TOOL" -c port=SWD mode=HOTPLUG ap=1 -el "$EXT_LOADER" -w "$bin" "$addr" -v
done

echo
echo "done — 3 weight blobs staged in external flash. balldetector_init copies them"
echo "into AXISRAM3/4/5 at boot. Load+run the app (dev boot) and feed images:"
echo "  python firmware/Host/host_balldetector_n6.py --port /dev/ttyACM0 img --images <dir>"
