#!/usr/bin/env python3
"""External comparison process only. Never imported by the NIRYUKTI solver or API."""
import argparse
import json
import time
import resource
import highspy

p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--time-limit',type=float,default=30);p.add_argument('--threads',type=int,default=1);p.add_argument('--tol',type=float,default=1e-6)
a=p.parse_args();start=time.perf_counter()
h=highspy.Highs();h.setOptionValue('output_flag',False);h.setOptionValue('threads',a.threads);h.setOptionValue('time_limit',a.time_limit)
for name in ['primal_feasibility_tolerance','dual_feasibility_tolerance','mip_feasibility_tolerance']:h.setOptionValue(name,a.tol)
h.setOptionValue('mip_rel_gap',1e-4)
load=h.readModel(a.model)
if load==highspy.HighsStatus.kError:raise RuntimeError('HiGHS could not read model')
parse=time.perf_counter()-start;tick=time.perf_counter();h.run();seconds=time.perf_counter()-tick
info=h.getInfo();solution=h.getSolution();status=h.modelStatusToString(h.getModelStatus())
status={'Optimal':'OPTIMAL','Infeasible':'INFEASIBLE','Unbounded':'UNBOUNDED','Time limit reached':'TIME_LIMIT','Iteration limit reached':'ITERATION_LIMIT'}.get(status,status.upper().replace(' ','_'))
print(json.dumps(dict(solver='HiGHS',version=h.version(),status=status,objective=h.getObjectiveValue() if solution.value_valid else None,
 primal=list(solution.col_value) if solution.value_valid else [],row_dual=list(solution.row_dual) if solution.dual_valid else [],
 performance=dict(parse_seconds=parse,iteration_seconds=seconds,end_to_end_seconds=time.perf_counter()-start),
 nodes=info.mip_node_count,mip_gap=info.mip_gap if info.mip_gap<float('inf') else None,
 peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
