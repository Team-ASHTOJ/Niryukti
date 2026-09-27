"""Compare cheap recovery with a recovery that protects a sensitive day's plan."""
import argparse
import json
from pathlib import Path
from refinery_replanning import refinery_model
from vantage import NativeSession,replan

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/stable-refinery-plan');args=parser.parse_args()
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    model=refinery_model()
    with NativeSession(model) as session:baseline=session.solve()
    if baseline['status']!='OPTIMAL':raise RuntimeError('Baseline did not reach optimality')
    reference=dict(zip([v['name'] for v in model.data['variables']],baseline['primal']))
    next(row for row in model.data['constraints'] if row['name']=='capacity_3')['ub']=0
    (folder/'baseline.json').write_text(json.dumps(baseline,indent=2))
    summary=[]
    for label,weight in [('cost_focused',.01),('stability_focused',5.)]:
        # Illustrative policy: day 2 has ten times the disruption cost.
        penalties={f'feed_{crude}_{day}':weight*(10 if day==2 else 1)
                   for day in range(7) for crude in ('clean','sour')}
        report=replan(model,reference,penalties,device='cpu',method='auto')
        (folder/f'{label}.json').write_text(json.dumps(report,indent=2))
        summary.append(dict(policy=label,status=report['status'],operating_cost=report['operating_objective'],
                            weighted_change_cost=report['disruption_penalty'],changes=report['changes']))
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
