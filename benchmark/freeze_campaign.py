#!/usr/bin/env python3
"""Execute the declared suite, retain failures, then seal its evidence checksums."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--binary', required=True)
p.add_argument('--output', default='results/prototype_0_2_1')
p.add_argument('--config', default=str(ROOT/'benchmark/campaigns/prototype_0_2_1.json'))
a = p.parse_args()
c = json.loads(Path(a.config).read_text())
out = Path(a.output).resolve()
command = [sys.executable, str(ROOT/'benchmark/run.py'), *[str(ROOT/m) for m in c['models']],
           '--binary', str(Path(a.binary).resolve()), '--output', str(out), '--solvers', c['solvers'],
           '--runs', str(c['runs']), '--time-limit', str(c['time_limit_seconds']),
           '--iterations', str(c['iteration_limit']), '--threads', str(c['threads']),
           '--tol', str(c['tolerance']), '--method', c['method']]
subprocess.run(command, check=True)
(out/'campaign.json').write_text(json.dumps(c, indent=2)+'\n')
checksums = {str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
             for f in sorted(out.rglob('*')) if f.is_file() and f.name != 'checksums.json'}
(out/'checksums.json').write_text(json.dumps(checksums, indent=2)+'\n')
print(f'Frozen evidence: {out} ({len(checksums)} files)')
