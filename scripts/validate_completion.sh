#!/usr/bin/env bash
# Functional validation only. Does not start a benchmark campaign.
set -euo pipefail
cd "$(dirname "$0")/.."
validation_build="${1:-build-completion}"
validation_gpu="${VANTAGE_VALIDATE_CUDA:-OFF}"
cmake -S . -B "$validation_build" -DCMAKE_BUILD_TYPE=Release -DVANTAGE_CUDA="$validation_gpu" ${VANTAGE_CUDA_ARCHITECTURES:+-DCMAKE_CUDA_ARCHITECTURES=$VANTAGE_CUDA_ARCHITECTURES}
cmake --build "$validation_build" -j "${VANTAGE_BUILD_JOBS:-4}"
ctest --test-dir "$validation_build" --output-on-failure
python3 tests/test_cli.py --binary "$validation_build/vantage"
VANTAGE_TEST_BINARY="$validation_build/vantage" python3 dashboard/tests/test_dashboard.py
"$validation_build/vantage" devices
# Browser check requires an independently running local server, Chromium and npm dependencies.
if [[ "${VANTAGE_VALIDATE_BROWSER:-0}" == 1 ]]; then
  node dashboard/browser-tests/check.cjs
fi
