#!/usr/bin/env python3
"""Prevent accidental solver-package dependencies in production code."""
import re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
forbidden=re.compile(r'\b(highspy|highs|scip|glpk|gurobi|cplex|xpress|ortools|or-tools|cvxpy|osqp|scipy\.optimize|coinor|cbc|clp|ecos|mosek)\b',re.I)
files=[root/'CMakeLists.txt',root/'pyproject.toml']
for folder in ['src','include','app','python']:
    files.extend(p for p in (root/folder).rglob('*') if p.suffix in {'.cpp','.hpp','.cu','.py'})
violations=[]
for p in files:
    for i,line in enumerate(p.read_text().splitlines(),1):
        if forbidden.search(line):violations.append(f'{p.relative_to(root)}:{i}: {line.strip()}')
if violations:raise SystemExit('\n'.join(violations))
print(f'Dependency policy passed: {len(files)} production files; comparison solvers confined to benchmark/')
