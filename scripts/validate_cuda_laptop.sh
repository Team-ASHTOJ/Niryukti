#!/usr/bin/env bash
# Reproducible validation, not a benchmark guarantee. No system configuration changes.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPORT=${NIRYUKTI_VALIDATION_DIR:-"$ROOT/results/laptop-$STAMP"}
BUILD=${NIRYUKTI_BUILD_DIR:-"$REPORT/build"}
mkdir -p "$REPORT"
exec > >(tee "$REPORT/validation.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$REPORT/exit-code.txt"; echo "Validation exit: $rc. Evidence: $REPORT"' EXIT
for tool in cmake python3 nvcc nvidia-smi; do
 command -v "$tool" >/dev/null || { echo "Missing $tool. Install compiler/CUDA prerequisites, then rerun."; exit 1; }
done
export CCACHE_DIR=${CCACHE_DIR:-"$REPORT/ccache"}
nvidia-smi > "$REPORT/nvidia-smi.txt"
nvcc --version > "$REPORT/cuda-version.txt"
cmake --version > "$REPORT/cmake-version.txt"
git rev-parse HEAD > "$REPORT/commit.txt"
git status --short > "$REPORT/working-tree.txt"
args=(-DCMAKE_BUILD_TYPE=Release -DNIRYUKTI_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native)
if command -v g++-15 >/dev/null 2>&1; then args+=(-DCMAKE_CUDA_HOST_COMPILER="$(command -v g++-15)"); fi
cmake -S . -B "$BUILD" "${args[@]}"
cmake --build "$BUILD" --parallel "${NIRYUKTI_BUILD_JOBS:-2}"
bash scripts/run_tests.sh "$BUILD"
"$BUILD/niryukti" devices > "$REPORT/devices.txt"
for model in refinery coupled_dispatch supply_chain scheduling; do
 for device in cpu cuda auto; do
  result="$REPORT/$model-$device.json"
  "$BUILD/niryukti" solve "examples/$model.json" --method auto --device "$device" --threads 2 --time-limit 60 --json-out "$result"
  "$BUILD/niryukti" verify "examples/$model.json" "$result" > "$REPORT/$model-$device-verification.json"
 done
done
export NIRYUKTI_BINARY="$BUILD/niryukti" NIRYUKTI_LIBRARY="$BUILD/libniryukti_c.so" PYTHONPATH="$ROOT/python"
python3 tests/test_service.py
python3 -m niryukti report "$REPORT/refinery-auto.json" --output "$REPORT/refinery-report.html"
if [[ ${1:-} == --docker ]]; then
 command -v docker >/dev/null || { echo 'Docker requested but unavailable'; exit 1; }
 docker build -f Dockerfile.cuda -t niryukti:laptop-cuda .
 # Requires a working NVIDIA Container Toolkit/CDI; failure is retained, not bypassed.
 docker run --rm --gpus all niryukti:laptop-cuda devices
 docker run --rm --gpus all niryukti:laptop-cuda solve examples/coupled_dispatch.json --device cuda
fi
python3 - "$REPORT" <<'PY'
import hashlib,json,platform,sys
from pathlib import Path
root=Path(sys.argv[1]);results={}
for p in root.glob('*-*.json'):
 if '-verification' in p.name:continue
 try:
  d=json.loads(p.read_text());results[p.name]={"status":d.get("status"),"selection":d.get("selection"),"accuracy":d.get("accuracy"),"performance":d.get("performance")}
 except (ValueError,OSError):pass
manifest={"platform":platform.platform(),"python":platform.python_version(),"results":results,
 "checksums":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file() and p.name!='validation.log'}}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
echo "PASS: core/research/API/CPU/CUDA/auto verification. Share $REPORT (exclude build/ccache)."
