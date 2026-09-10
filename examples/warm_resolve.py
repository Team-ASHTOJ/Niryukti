#!/usr/bin/env python3
import argparse
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from vantage import solve
from generate import refinery

p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--output',default='results/warm_resolve.json');a=p.parse_args()
with tempfile.TemporaryDirectory(prefix='vantage-parametric-') as tmp:
    before=Path(tmp)/'before.json';after=Path(tmp)/'after.json'
    before.write_text(json.dumps(refinery()));after.write_text(json.dumps(refinery(demand_multiplier=1.05)))
    first=solve(before,device=a.device);cold=solve(after,device=a.device)
    warm=solve(after,device=a.device,warm_start=first,allow_model_change=True)
    result=dict(description='Synthetic refinery demand increased by 5%; no assumed speedup',initial=first,cold=cold,warm=warm)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2))
    for key,r in [('initial',first),('cold',cold),('warm',warm)]:
        print(key,r['status'],'objective',r['objective'],'iterations',r['performance']['iterations'],'seconds',r['performance']['end_to_end_seconds'])
    if any(r['status']!='OPTIMAL' for r in (first,cold,warm)):raise SystemExit(2)
    if abs(cold['objective']-warm['objective'])>1e-5*(1+abs(cold['objective'])):raise SystemExit('Warm/cold objectives disagree')
