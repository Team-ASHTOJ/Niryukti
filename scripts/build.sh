#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
args=(-DCMAKE_BUILD_TYPE=Release)
if command -v nvcc >/dev/null 2>&1; then
  args+=(-DVANTAGE_CUDA=ON)
  # CUDA 13 supports GCC 15; rolling-release hosts can have a newer default GCC.
  if command -v g++-15 >/dev/null 2>&1; then args+=(-DCMAKE_CUDA_HOST_COMPILER="$(command -v g++-15)"); fi
else
  args+=(-DVANTAGE_CUDA=OFF)
fi
cmake -S . -B build "${args[@]}" "$@"
cmake --build build --parallel "${VANTAGE_BUILD_JOBS:-4}"
