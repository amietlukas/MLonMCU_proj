#!/usr/bin/env bash
# One-shot: prepare model -> stedgeai generate -> headless build -> sign+flash.
# Each step is also runnable on its own; see the individual scripts.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$PROJ/../.." && pwd)"

echo "==> [1/4] prepare model (ONNX sanity-check + copy into model/)"
if [[ -f "$REPO/software/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  ( cd "$REPO" && source software/venv/bin/activate && python "$HERE/prepare_model.py" )
else
  ( cd "$REPO" && python "$HERE/prepare_model.py" )
fi

echo
echo "==> [2/4] stedgeai generate -> X-CUBE-AI/App/"
"$HERE/stedgeai_compile.sh"

echo
echo "==> [3/4] headless CubeIDE build"
"$HERE/build.sh"

echo
echo "==> [4/4] sign + flash app + weights"
"$HERE/flash.sh"

echo
echo "done — run: python $REPO/firmware/Host/host_balldetector_n6.py --port /dev/ttyACM0 cam"
