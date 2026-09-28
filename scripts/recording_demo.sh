#!/usr/bin/env bash
# A short, offline sequence for recording. No benchmark winners are invented.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
BINARY=${NIRYUKTI_BINARY:-"$ROOT/build/niryukti"}
OUTPUT=${NIRYUKTI_DEMO_DIR:-"$ROOT/results/recording-$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$OUTPUT"
[[ -x "$BINARY" ]] || { echo 'Build first, or set NIRYUKTI_BINARY.'; exit 1; }
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
"$BINARY" devices | tee "$OUTPUT/devices.json"
"$BINARY" explain examples/refinery.json | tee "$OUTPUT/refinery-analysis.json"
for model in refinery coupled_dispatch supply_chain scheduling; do
  "$BINARY" solve "examples/$model.json" --method auto --device auto --threads 2 --iterations 1000000 --time-limit 30 --json-out "$OUTPUT/$model.json"
  "$BINARY" verify "examples/$model.json" "$OUTPUT/$model.json" > "$OUTPUT/$model-verification.json"
  python3 -m niryukti report "$OUTPUT/$model.json" --output "$OUTPUT/$model.html"
done
# Demonstrate the actual GPU path explicitly; auto intentionally chooses CPU for tiny models.
if grep -q '^CUDA: NVIDIA' "$OUTPUT/devices.json"; then
  "$BINARY" solve examples/refinery.json --method pdhg --device cuda --cuda-graphs --gpu-monitor --iterations 1000000 --time-limit 30 --json-out "$OUTPUT/refinery-cuda.json"
  "$BINARY" verify examples/refinery.json "$OUTPUT/refinery-cuda.json" > "$OUTPUT/refinery-cuda-verification.json"
fi
# Demonstrate real Newton-state continuation, not a fresh solve renamed as resume.
"$BINARY" solve examples/coupled_dispatch.json --method barrier --device cpu --no-presolve --iterations 2 --checkpoint-out "$OUTPUT/barrier-state.json" --json-out "$OUTPUT/barrier-limited.json" || [[ $? == 2 ]]
"$BINARY" solve examples/coupled_dispatch.json --method auto --device cpu --no-presolve --iterations 10000 --resume "$OUTPUT/barrier-state.json" --json-out "$OUTPUT/barrier-resumed.json"
"$BINARY" verify examples/coupled_dispatch.json "$OUTPUT/barrier-resumed.json" > "$OUTPUT/barrier-resumed-verification.json"
python3 - "$OUTPUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); c={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file()}
(p/'checksums.json').write_text(json.dumps(c,indent=2)+'\n')
print('Recording evidence:',p)
print('Integer verification proves incumbents, not a replay of the complete search tree.')
PY
printf '\nDashboard: NIRYUKTI_BINARY=%q python3 dashboard/server.py --host 127.0.0.1 --port 8080\n' "$BINARY"
