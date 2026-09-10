#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--binary',default='build/vantage');a=p.parse_args();binary=str(Path(a.binary).resolve())
sys.path.insert(0,str(root/'python'));from vantage import Model

def run(*args):return subprocess.run([binary,*map(str,args)],text=True,capture_output=True)

with tempfile.TemporaryDirectory(prefix='vantage-cli-tests-') as tmp:
    tmp=Path(tmp);sol=tmp/'solution.json'
    r=run('solve',root/'examples/toy.lp','--json-out',sol);assert r.returncode==0,r.stderr
    data=json.loads(sol.read_text());assert data['status']=='OPTIMAL'
    assert run('verify',root/'examples/toy.lp',sol).returncode==0
    data['primal']=[0,0];sol.write_text(json.dumps(data));assert run('verify',root/'examples/toy.lp',sol).returncode==2
    assert run('solve',root/'examples/toy.lp','--device','imaginary').returncode==1
    assert run('solve',root/'examples/toy.lp','--tol','nan').returncode==1
    # Malformed sparse JSON and malformed MPS may not crash or enter the engine.
    malformed=[{'variables':[],'objective':{'linear':[1]},'constraints':[]},
               {'variables':[{'name':'x'}],'objective':{'linear':[1]},'constraints':[{'coefficients':{'missing':1},'lb':2}]}]
    for i,data in enumerate(malformed):
        path=tmp/f'bad{i}.json';path.write_text(json.dumps(data));r=run('solve',path);assert r.returncode==1,(r.returncode,r.stderr)
    path=tmp/'bad.mps';path.write_text("NAME BAD\nROWS\n N OBJ\nCOLUMNS\n M 'MARKER' 'INTEND'\nENDATA\n");assert run('solve',path).returncode==1
    # A changed model is rejected unless parametric warm-start use is explicit.
    model=Model();x=model.add_var('x',ub=10);y=model.add_var('y',ub=10)
    model.add_constraint({x:1,y:2},'>=',4);model.set_objective({x:1,y:1})
    first=model.solve(binary=binary,device='cpu');assert first['status']=='OPTIMAL'
    model.data['constraints'][0]['lb']=4.2
    try:model.solve(binary=binary,warm_start=first);raise AssertionError('mismatched warm start accepted')
    except RuntimeError:pass
    second=model.solve(binary=binary,warm_start=first,allow_model_change=True);assert second['status']=='OPTIMAL'
    assert abs(second['objective']-2.1)<1e-5
print('CLI, malformed-input rejection, verification, Python API and parametric warm-start tests passed')
