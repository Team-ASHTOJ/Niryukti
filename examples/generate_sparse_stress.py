#!/usr/bin/env python3
"""Streaming synthetic sparse LP with a planted primal/dual optimum.

This is a memory/throughput stress model, not a public industrial benchmark.
For each block i, e_i + .25 e_(i+1) + .5 o_i = 1.25, 0<=e,o<=2.
Choose d_i in 1..10; c_e=d_i+.25*d_(i-1), c_o=.5*d_i+3.
y_i=-d_i and reduced costs (0,3) prove optimum e=1,o=0.
"""
import argparse, hashlib, json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--variables',type=int,default=1000000)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.variables<4 or a.variables%2:p.error('variables must be even and at least four')
b=a.variables//2
def d(i):return 1+((i*1103515245+a.seed*12345)&0x7fffffff)%10
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w') as f:
 f.write('NAME SYNTHETIC_PLANTED_SPARSE\nROWS\n N OBJ\n')
 for i in range(b):f.write(f' E R{i}\n')
 f.write('COLUMNS\n')
 for i in range(b):
  f.write(f' E{i} OBJ {d(i)+.25*d((i-1)%b):.17g} R{i} 1\n')
  f.write(f' E{i} R{(i-1)%b} .25\n')
  f.write(f' O{i} OBJ {.5*d(i)+3:.17g} R{i} .5\n')
 f.write('RHS\n')
 for i in range(b):f.write(f' RHS R{i} 1.25\n')
 f.write('BOUNDS\n')
 for i in range(b):f.write(f' UP BND E{i} 2\n UP BND O{i} 2\n')
 f.write('ENDATA\n')
h=hashlib.sha256()
with a.output.open('rb') as f:
 for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
record=dict(synthetic=True,kind='planted sparse cyclic LP',variables=a.variables,rows=b,
 nonzeros=3*b,seed=a.seed,expected_objective=1.25*sum(d(i) for i in range(b)),
 primal='e_i=1,o_i=0',dual='y_i=-d_i',reduced_costs='e_i=0,o_i=3',sha256=h.hexdigest(),bytes=a.output.stat().st_size)
a.output.with_suffix('.manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
