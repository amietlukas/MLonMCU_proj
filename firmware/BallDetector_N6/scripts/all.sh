#!/usr/bin/env bash
# One-shot: prepare model -> stedgeai -> make -> flash app + weights.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

echo "==> [1/4] prepare model (ONNX sanity-check + copy)"
( cd "$REPO" && source software/venv/bin/activate \
  && python software/ball_detector_n6/prepare_model.py )

echo
echo "==> [2/4] stedgeai generate -> ST app Model/"
"$HERE/stedgeai_compile.sh"

echo
echo "==> [3/4] make all sign"
"$HERE/build_n6.sh"

echo
echo "==> [4/4] flash app + weights"
"$HERE/flash_n6_signed.sh"

echo
echo "done — connect over USB (UVC video device) to see the live feed."
