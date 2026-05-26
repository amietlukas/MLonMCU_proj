#!/usr/bin/env bash
# Build BallDetector_N6 via ST's GNU-Make project (no CubeIDE needed).
# Produces build/Application/NUCLEO-N657X0-Q/Project.{elf,bin,_sign.bin,_sign.hex}.

set -euo pipefail

# --- adjust me ----------------------------------------------------------------
GCC_BIN="${GCC_BIN:-/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.linux64_1.0.100.202602081740/tools/bin}"
CUBEPROG_BIN="${CUBEPROG_BIN:-/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.linux64_2.2.400.202601091506/tools/bin}"
JOBS="${JOBS:-$(nproc)}"
# ------------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
APP_DIR="$PROJ/stm32ai-modelzoo-services/application_code/object_detection/STM32N6/Application/NUCLEO-N657X0-Q"

if [[ ! -f "$APP_DIR/Makefile" ]]; then
  echo "ST app Makefile missing: $APP_DIR/Makefile" >&2
  echo "init the submodule first (see README)" >&2; exit 1
fi
if [[ ! -x "$GCC_BIN/arm-none-eabi-gcc" ]]; then
  echo "arm-none-eabi-gcc not at: $GCC_BIN" >&2; exit 1
fi

export PATH="$GCC_BIN:$CUBEPROG_BIN:$PATH"

echo "==> make -j$JOBS  (cwd: $APP_DIR)"
make -C "$APP_DIR" -j"$JOBS" all sign V=1 | tail -40 || {
  echo "build failed — re-run without 'V=1 | tail' to see full output:" >&2
  echo "  make -C $APP_DIR -j$JOBS all sign" >&2
  exit 1
}

echo
echo "==> artifacts:"
ls -lh "$APP_DIR/build/Application/NUCLEO-N657X0-Q/"*.elf \
       "$APP_DIR/build/Application/NUCLEO-N657X0-Q/"*.bin 2>/dev/null | sed 's/^/  /' || true
