#!/usr/bin/env python3
"""External SCIP comparison process only; never imported by the solving core."""
import argparse, json, resource, time
from pyscipopt import Model, __version__
p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--time-limit',type=float,default=60);p.add_argument('--threads',type=int,default=4);p.add_argument('--tol',type=float,default=1e-6)
a=p.parse_args();start=time.perf_counter()
model=Model();model.hideOutput()
model.setParam('limits/time',a.time_limit);model.setParam('numerics/feastol',a.tol);model.setParam('limits/gap',1e-4)
model.setParam('parallel/maxnthreads',a.threads);model.setParam('lp/threads',a.threads)
model.readProblem(a.model);parse=time.perf_counter()-start
# Preserve original MPS column order for the independent VANTAGE verifier.
names=[];seen=set();columns=False
with open(a.model) as f:
 for line in f:
  fields=line.split()
  if not fields or fields[0].startswith('*'):continue
  if fields[0]=='COLUMNS' and len(fields)==1:columns=True;continue
  if columns and len(fields)==1:columns=False
  if columns and not any('MARKER' in v for v in fields) and fields[0] not in seen:
   names.append(fields[0]);seen.add(fields[0])
tick=time.perf_counter();model.optimize();seconds=time.perf_counter()-tick
solution=model.getBestSol();variables={v.name:v for v in model.getVars(transformed=False)}
status=str(model.getStatus()).upper();status={'TIMELIMIT':'TIME_LIMIT','NODELIMIT':'NODE_LIMIT','GAPLIMIT':'APPROXIMATE_OPTIMAL'}.get(status,status)
primal=[model.getSolVal(solution,variables[name]) for name in names] if solution is not None else []
bound=model.getDualbound();gap=model.getGap()
print(json.dumps(dict(solver='SCIP',version=f'{model.getMajorVersion()}.{model.getMinorVersion()}.{model.getTechVersion()}',pyscipopt_version=__version__,status=status,
 objective=model.getSolObjVal(solution) if solution is not None else None,primal=primal,row_dual=[],
 best_bound=bound if abs(bound)<1e19 else None,mip_gap=gap if gap<1e19 else None,nodes=model.getNNodes(),
 performance=dict(parse_seconds=parse,iteration_seconds=seconds,end_to_end_seconds=time.perf_counter()-start),
 peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 dual_verification='Not supplied by this adapter; use on MILP/MIQP for independent incumbent verification')))
