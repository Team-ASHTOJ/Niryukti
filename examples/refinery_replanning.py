"""Synthetic seven-day linear blending/inventory plan and disruption replay."""
import argparse
import copy
import json
from pathlib import Path
import sys
import subprocess
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from vantage import Model, SolverSession, propose_repair, diagnose_infeasibility, _binary


def refinery_model(days=7):
    m=Model('synthetic_coupled_refinery')
    costs={}
    for t in range(days):
        for crude,price in [('clean',70),('sour',45)]:
            m.add_var(f'buy_{crude}_{t}',ub=40)
            m.add_var(f'feed_{crude}_{t}',ub=100)
            m.add_var(f'tank_{crude}_{t}',ub=100)
            costs[f'buy_{crude}_{t}']=price
            costs[f'tank_{crude}_{t}']=.1
        m.add_var(f'product_{t}',ub=200); costs[f'product_{t}']=.2
    for t in range(days):
        for crude in ('clean','sour'):
            terms={f'buy_{crude}_{t}':1,f'feed_{crude}_{t}':-1,f'tank_{crude}_{t}':-1}
            if t: terms[f'tank_{crude}_{t-1}']=1
            m.add_constraint(terms,'=',-(40 if crude=='clean' else 20) if t==0 else 0,name=f'crude_balance_{crude}_{t}')
        terms={f'feed_clean_{t}':1,f'feed_sour_{t}':1,f'product_{t}':-1}
        if t: terms[f'product_{t-1}']=1
        m.add_constraint(terms,'=',60,name=f'demand_{t}')
        m.add_constraint({f'feed_clean_{t}':1,f'feed_sour_{t}':1},'<=',100,name=f'capacity_{t}')
        # Sulfur mass balance: clean 0.5%, sour 2%, product at most 1%.
        m.add_constraint({f'feed_clean_{t}':-.005,f'feed_sour_{t}':.01},'<=',0,name=f'sulfur_{t}')
    m.set_objective(costs)
    return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/refinery-replanning');p.add_argument('--binary');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    baseline=refinery_model(); baseline.write(out/'baseline.json')
    scenarios=[('baseline',{},{}),('delivery_delay',{'buy_clean_1':{'ub':0},'buy_clean_3':{'ub':80}},{}),
               ('unit_outage',{}, {'capacity_3':{'ub':0}})]
    records=[]
    with SolverSession(baseline,binary=args.binary) as session:
        for name,variables,rows in scenarios:
            # Every disruption is relative to baseline, not cumulative.
            session.update_variable_bounds({v['name']:{'lb':v['lb'],'ub':v['ub']} for v in baseline.data['variables']})
            session.update_row_bounds({r['name']:{'lb':r['lb'],'ub':r['ub']} for r in baseline.data['constraints']})
            session.update_variable_bounds(variables);session.update_row_bounds(rows)
            data=copy.deepcopy(baseline.data)
            for v in data['variables']:v.update(variables.get(v['name'],{}))
            for r in data['constraints']:r.update(rows.get(r['name'],{}))
            (out/f'{name}.json').write_text(json.dumps(data,indent=2))
            tick=time.perf_counter();warm=session.solve();warm_wall=time.perf_counter()-tick
            with SolverSession(data,binary=args.binary) as cold:
                tick=time.perf_counter();fresh=cold.solve(warm_start=False);cold_wall=time.perf_counter()-tick
            (out/f'{name}.solution.json').write_text(json.dumps(warm,indent=2))
            (out/f'{name}.cold.solution.json').write_text(json.dumps(fresh,indent=2))
            checks={}
            for suffix in ('solution','cold.solution'):
                proc=subprocess.run([_binary(args.binary),'verify',str(out/f'{name}.json'),str(out/f'{name}.{suffix}.json')],capture_output=True,text=True)
                checked=json.loads(proc.stdout)
                (out/f'{name}.{suffix}.verification.json').write_text(json.dumps(checked,indent=2))
                checks[suffix]=checked['status']
            records.append(dict(scenario=name,status=warm['status'],objective=warm.get('objective'),
                                cold_status=fresh['status'],cold_objective=fresh.get('objective'),
                                verification=checks,
                                warm_request_seconds=warm_wall,cold_request_seconds=cold_wall,
                                timing_scope='Request/response only; session startup excluded for both'))
    impossible=copy.deepcopy(baseline.data)
    for v in impossible['variables']:
        if v['name'].startswith('buy_'):v['ub']=0
    (out/'supply_failure.json').write_text(json.dumps(impossible,indent=2))
    diagnostic=diagnose_infeasibility(impossible,binary=args.binary,method='auto',device='cpu')
    (out/'infeasibility.json').write_text(json.dumps(diagnostic,indent=2))
    repair=propose_repair(impossible,{f'demand_{t}':1 for t in range(7)},
                          violation_limits={f'demand_{t}':{'lb':60,'ub':0} for t in range(7)},
                          binary=args.binary,method='auto',device='cpu')
    (out/'repair-proposal.json').write_text(json.dumps(repair,indent=2))
    (out/'summary.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))

if __name__=='__main__':main()
