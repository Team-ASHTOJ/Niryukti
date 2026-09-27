#!/usr/bin/env bash
# Public comparisons stay outside the solver. Requires benchmark/requirements.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
bench_python="${VANTAGE_BENCH_PYTHON:-.venv/bin/python}"
bench_binary="${VANTAGE_BINARY:-build/vantage}"
bench_output="${VANTAGE_STRESS_OUTPUT:-results/stress}"
"$bench_python" benchmark/download_public.py --suite netlib dfl001 stocfor2
"$bench_python" benchmark/download_public.py --suite miplib --output datasets/stress_public rail03 ns1644855
"$bench_python" examples/generate_sparse_stress.py --variables 1000000 --seed 42 --output datasets/public/stress/planted_1m.mps
"$bench_python" - <<'PYCODE'
from pathlib import Path
import highspy
source=Path('datasets/stress_public/miplib/rail03.mps')
if not source.exists(): raise SystemExit('rail03 download failed; consult dataset manifest')
Path('datasets/public/relaxations').mkdir(parents=True,exist_ok=True)
h=highspy.Highs();h.setOptionValue('output_flag',False)
assert h.readModel(str(source))==highspy.HighsStatus.kOk
assert h.changeColsIntegrality(h.getNumCol(),list(range(h.getNumCol())),[highspy.HighsVarType.kContinuous]*h.getNumCol())==highspy.HighsStatus.kOk
assert h.writeModel('datasets/public/relaxations/rail03_lp.mps')==highspy.HighsStatus.kOk
# Conversion only: no optimization is performed here.
PYCODE
"$bench_python" benchmark/run.py datasets/public/netlib/dfl001.mps datasets/public/netlib/stocfor2.mps datasets/public/relaxations/rail03_lp.mps datasets/public/stress/planted_1m.mps \
  --binary "$bench_binary" --solvers cpu,cuda,highs --method pdhg --runs 1 \
  --time-limit 60 --iterations 500000 --threads 4 --gpu-monitor --cuda-graphs --output "$bench_output-lp"
"$bench_python" benchmark/run.py datasets/stress_public/miplib/rail03.mps datasets/stress_public/miplib/ns1644855.mps \
  --binary "$bench_binary" --solvers cpu,cuda,highs,scip --method auto --runs 1 \
  --time-limit 60 --iterations 500000 --threads 4 --branching reliability --cuts --primal-heuristic all --gpu-monitor --output "$bench_output-mip"
