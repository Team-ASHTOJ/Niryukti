#!/usr/bin/env python3
"""Synthetic sparse PSD stress case with a known optimum, never industrial data."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--variables',type=int,default=1200)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.variables<2: p.error('--variables must be at least 2')
n=a.variables
# Q = sum_{j>0}(e0 - 2ej)(e0 - 2ej)^T; Q is singular and PSD.
# c = -Q e0; bounds [0,1] leave unique minimizer e0, objective -(n-1)/2.
q=[[0,0,n-1]]
for j in range(1,n): q.extend([[0,j,-2],[j,0,-2],[j,j,4]])
model={'name':'synthetic_singular_weighted_star','sense':'min',
       'variables':[{'name':f'x{j}','lb':0,'ub':1} for j in range(n)],
       'objective':{'linear':[-(n-1)]+[2]*(n-1),'quadratic_sparse':q},
       'constraints':[]}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(model)+'\n')
print(f'Synthetic singular PSD QP: {n} variables, {len(q)} Q nonzeros; known optimum {-(n-1)/2}')
