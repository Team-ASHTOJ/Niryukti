#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
binary="${VANTAGE_BINARY:-build/vantage}"
folder="$(mktemp -d)"
trap 'rm -rf "$folder"' EXIT
"$binary" solve examples/toy.lp --device cpu --method auto --certificate "$folder/certificate.json" > "$folder/result.json"
echo 'Original certificate: VALID'
"$binary" verify examples/toy.lp "$folder/certificate.json"
cp "$folder/certificate.json" "$folder/original.json"
python3 - "$folder/certificate.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);c=json.loads(p.read_text());c['objective']+=10;p.write_text(json.dumps(c))
PY
echo 'Tampered certificate: expected INVALID'
if "$binary" verify examples/toy.lp "$folder/certificate.json"; then
  echo 'ERROR: tampering accepted' >&2
  exit 1
fi
cp "$folder/original.json" "$folder/certificate.json"
echo 'Restored certificate: VALID'
"$binary" verify examples/toy.lp "$folder/certificate.json"
