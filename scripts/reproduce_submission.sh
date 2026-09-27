#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# No downloads or hidden baseline installation. Cache datasets explicitly first.
repro_build="${VANTAGE_REPRO_BUILD:-build-cpu}"
repro_output="${1:-results/reproduced-submission}"
cmake -S . -B "$repro_build" -DCMAKE_BUILD_TYPE=Release -DVANTAGE_CUDA="${VANTAGE_REPRO_CUDA:-OFF}"
cmake --build "$repro_build" -j "${VANTAGE_BUILD_JOBS:-4}"
./scripts/run_tests.sh "$repro_build"
repro_python="${VANTAGE_BENCH_PYTHON:-python3}"
"$repro_python" benchmark/run.py examples/toy.lp examples/refinery.json examples/production.json examples/dispatch.json examples/coupled_dispatch.json examples/integer_dispatch.json examples/supply_chain.json examples/scheduling.json \
  --binary "$repro_build/vantage" --method auto --solvers "${VANTAGE_REPRO_SOLVERS:-cpu}" --runs 3 --time-limit 10 --output "$repro_output"
printf 'Raw runs, checksums, manifest, CSV and HTML: %s\n' "$repro_output"
