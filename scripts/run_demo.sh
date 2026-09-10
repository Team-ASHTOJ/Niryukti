#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -x build/vantage ]] || ./scripts/build.sh
mkdir -p results/demo
./build/vantage devices
./build/vantage explain examples/refinery.json
# All measured statuses (including limits) are retained by the benchmark harness.
python_bin=python3
solvers=cpu,cuda
if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import highspy' 2>/dev/null; then
  python_bin=.venv/bin/python
  solvers=cpu,cuda,highs
fi
"$python_bin" benchmark/run.py examples/refinery.json examples/dispatch.json examples/supply_chain.json datasets/afiro.mps --solvers "$solvers" --runs 3 --time-limit 10 --output results/demo
"$python_bin" examples/warm_resolve.py --output results/demo/warm_resolve.json
printf '\nOffline report: %s/results/demo/index.html\n' "$PWD"
