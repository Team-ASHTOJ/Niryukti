"""Explicit economic trade-off between operating cost and changing an approved plan."""
import copy
import math
import json
from pathlib import Path
import subprocess
import tempfile
from . import Model, _binary, solve

def stable_plan_model(model, reference, penalties, *, locked=()):
    """LP/MILP objective plus weighted absolute deviations; retain hard constraints.

    penalties maps selected variable names to positive cost per unit of change.
    locked variables are fixed exactly to their reference values.
    """
    data=copy.deepcopy(model.data if hasattr(model,'data') else model)
    locked=tuple(locked)
    if set(data['objective'])-{'linear','offset'}:raise ValueError('Plan stability currently supports linear objectives')
    names=[v['name'] for v in data['variables']];used=set(names)
    if not isinstance(data['objective']['linear'],list) or len(data['objective']['linear'])!=len(names):raise ValueError('Expected one objective coefficient per variable')
    if len(used)!=len(names):raise ValueError('Unique variable names required')
    if not (set(penalties)|set(locked))<=used:raise ValueError('Unknown plan variable')
    if not (set(penalties)|set(locked))<=set(reference):raise ValueError('Missing reference value')
    if any(not math.isfinite(reference[n]) for n in set(penalties)|set(locked)):raise ValueError('Nonfinite reference')
    if any(not math.isfinite(w) or w<=0 for w in penalties.values()):raise ValueError('Penalties must be finite and positive')
    for v in data['variables']:
        if v['name'] in locked:
            value=reference[v['name']]
            if (v.get('lb') is not None and value<v['lb']) or (v.get('ub') is not None and value>v['ub']):raise ValueError('Locked plan violates current bounds')
            if v.get('type','continuous')!='continuous' and value!=round(value):raise ValueError('Locked integer plan must be integral')
            v['lb']=v['ub']=value
    row_names={r['name'] for r in data['constraints']}
    for index,(name,weight) in enumerate(penalties.items()):
        aux=f'__change_{index}'
        while aux in used or aux+'_up' in row_names or aux+'_down' in row_names:aux+='_' 
        used.add(aux);row_names.update([aux+'_up',aux+'_down'])
        data['variables'].append(dict(name=aux,lb=0,ub=None))
        data['objective']['linear'].append(weight if data.get('sense','min')=='min' else -weight)
        data['constraints'] += [dict(name=aux+'_up',coefficients={name:1,aux:-1},ub=reference[name]),
                                dict(name=aux+'_down',coefficients={name:-1,aux:-1},ub=-reference[name])]
    result=Model();result.data=data;return result

def replan(model, reference, penalties, *, locked=(), binary=None, **options):
    original=model.data if hasattr(model,'data') else model
    augmented=stable_plan_model(model,reference,penalties,locked=locked)
    with tempfile.TemporaryDirectory(prefix='vantage-stable-plan-') as temp:
        folder=Path(temp);path=folder/'model.json';augmented.write(path)
        result=solve(path,binary=binary,**options)
        solution=folder/'solution.json';solution.write_text(json.dumps(result))
        proc=subprocess.run([_binary(binary),'verify',str(path),str(solution)],capture_output=True,text=True,timeout=120)
        try:verification=json.loads(proc.stdout)
        except ValueError:verification=dict(status='UNAVAILABLE')
    accepted=proc.returncode==0 and verification.get('status') in ('VERIFIED_OPTIMAL','VERIFIED_FEASIBLE')
    plan={v['name']:x for v,x in zip(original['variables'],result.get('primal',[]))} if accepted else {}
    operating_cost=None;disruption_cost=None;changes={}
    if accepted:
        operating_cost=original['objective'].get('offset',0)+sum(c*plan[v['name']] for c,v in zip(original['objective']['linear'],original['variables']))
        changes={n:plan[n]-reference[n] for n in penalties}
        disruption_cost=sum(penalties[n]*abs(delta) for n,delta in changes.items())
    return dict(status='VERIFIED_PLAN' if accepted else 'NO_VERIFIED_PLAN',plan=plan,changes=changes,
                operating_objective=operating_cost,disruption_penalty=disruption_cost,result=result,
                verification=verification,model=augmented.data,
                scope='Weighted cost/change trade-off, not lexicographic optimization. MIP verification checks incumbent feasibility.')
