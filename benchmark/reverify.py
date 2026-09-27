#!/usr/bin/env python3
"""Replay verification with a larger budget; preserve the original benchmark CSV."""
import argparse, csv, hashlib, json, subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('campaign',type=Path);p.add_argument('--binary',required=True);p.add_argument('--timeout',type=float,default=90)
a=p.parse_args();root=a.campaign;manifest=json.loads((root/'manifest.json').read_text());raw=root/'raw'
rows=list(csv.DictReader((root/'runs.csv').open()));checks=[];models={}
for row in rows:
 if row['reported_status']!='OPTIMAL' or row['status']=='OPTIMAL':continue
 files=list(raw.glob(f'*_{row["solver"]}_{row["run"]}.json'))
 entry=next((f for f in files if json.loads(f.read_text())['record']['instance']==row['instance']),None)
 if entry is None:continue
 index=int(entry.name.split('_')[0]);model=Path(manifest['arguments']['models'][index]).resolve()
 if index not in models:
  inspected=subprocess.run([a.binary,'inspect',str(model)],capture_output=True,text=True,timeout=a.timeout,check=True)
  info=json.loads(inspected.stdout)
  sense=1
  if model.suffix=='.json':sense=1 if json.loads(model.read_text()).get('sense','min')=='min' else -1
  else:
   import highspy
   h=highspy.Highs();h.setOptionValue('output_flag',False);h.readModel(str(model));sense=1 if h.getObjectiveSense()[1]==highspy.ObjSense.kMinimize else -1
  models[index]=(info,sense)
 info,sense=models[index];wrapped=json.loads(entry.read_text());data=wrapped['result']
 solution=raw/(entry.stem+'.recheck.solution.json')
 if row['solver'] in ('cpu','cuda'):
  solution=raw/(entry.stem+'.solution.json')
 else:
  dual=[-sense*v for v in data.get('row_dual',[])]
  if len(dual)!=info['rows']:dual=[0.]*info['rows']
  solution.write_text(json.dumps(dict(model=dict(fingerprint=info['fingerprint']),status=data['status'],objective=data.get('objective'),primal=data['primal'],dual=dual)))
 command=[a.binary,'verify',str(model),str(solution)]
 result=subprocess.run(command,capture_output=True,text=True,timeout=a.timeout)
 (raw/(entry.stem+'.recheck.stdout')).write_text(result.stdout)
 (raw/(entry.stem+'.recheck.stderr')).write_text(result.stderr)
 verification=json.loads(result.stdout);status=verification.get('status')
 accepted='VERIFIED_FEASIBLE' if info['type'] in ('MILP','MIQP') else 'VERIFIED_OPTIMAL'
 row.update(status='OPTIMAL' if status==accepted else 'VERIFICATION_FAILED',verification_status=status,rows=info['rows'],columns=info['columns'],nonzeros=info['nonzeros'],problem_type=info['type'])
 for field in ('primal_residual','dual_residual','kkt_error'):row[field]=verification.get('accuracy',{}).get(field)
 checks.append(dict(instance=row['instance'],solver=row['solver'],run=row['run'],status=status,command=command,exit_code=result.returncode,timeout=a.timeout,measured_entry_sha256=hashlib.sha256(entry.read_bytes()).hexdigest()))
 print(row['instance'],row['solver'],status,flush=True)
with (root/'runs.reverified.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
(root/'verification_rechecks.json').write_text(json.dumps(dict(original_csv_sha256=hashlib.sha256((root/'runs.csv').read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(Path(a.binary).read_bytes()).hexdigest(),checks=checks,note='Verification replay only; no optimization reruns or timing edits.'),indent=2)+'\n')
