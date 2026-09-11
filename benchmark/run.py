#!/usr/bin/env python3
"""Retain every selected instance/run, including process errors and limits."""
import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import uuid
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def execute(command, timeout):
    tick=time.perf_counter()
    try:
        p=subprocess.run(command,text=True,capture_output=True,timeout=timeout)
        try:data=json.loads(p.stdout)
        except (ValueError,TypeError):data=dict(status='PROCESS_ERROR',message=p.stderr[-4000:])
        return data,p.stdout,p.stderr,p.returncode,time.perf_counter()-tick
    except subprocess.TimeoutExpired as e:
        return dict(status='PROCESS_TIMEOUT'),str(e.stdout or ''),str(e.stderr or ''),None,time.perf_counter()-tick
    except OSError as e:return dict(status='UNAVAILABLE',message=str(e)),'',str(e),None,time.perf_counter()-tick

def main():
    p=argparse.ArgumentParser();p.add_argument('models',nargs='+');p.add_argument('--binary',default=str(ROOT/'build/vantage'));p.add_argument('--solvers',default='cpu,cuda,highs');p.add_argument('--runs',type=int,default=3);p.add_argument('--iterations',type=int,default=100000);p.add_argument('--time-limit',type=float,default=10);p.add_argument('--threads',type=int,default=1);p.add_argument('--tol',type=float,default=1e-6);p.add_argument('--output',default='results/latest');p.add_argument('--no-scaling',action='store_true');p.add_argument('--no-restart',action='store_true');p.add_argument('--no-adaptive',action='store_true')
    a=p.parse_args()
    if a.runs<1:p.error('--runs must be positive')
    if a.iterations<1:p.error('--iterations must be positive')
    out=Path(a.output)
    if (out/'manifest.json').exists():
        archive=out.with_name(out.name+'-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
        shutil.move(str(out),str(archive));print(f'Previous complete/partial run retained at {archive}',flush=True)
    out.mkdir(parents=True,exist_ok=True);raw=out/'raw';raw.mkdir(exist_ok=True)
    shutil.copy2(a.binary,raw/'vantage-measured')
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True).stdout.strip() or 'uncommitted'
    metadata=dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),platform=platform.platform(),cpu=platform.processor(),python=sys.version,git_commit=commit,arguments=vars(a),binary_sha256=hashlib.sha256(Path(a.binary).read_bytes()).hexdigest())
    metadata['cpu_info']=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor()) if Path('/proc/cpuinfo').exists() else platform.processor()
    metadata['ram_bytes']=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES') if hasattr(os,'sysconf') else None
    metadata['compiler']=execute(['c++','--version'],10)[1]
    metadata['cuda_toolkit']=execute(['nvcc','--version'],10)[1]
    metadata['gpu_state']=execute(['nvidia-smi','--query-gpu=name,driver_version,memory.total,power.limit','--format=csv'],10)[1]
    cache=Path(a.binary).parent/'CMakeCache.txt'
    if cache.exists():metadata['cmake_cache']=cache.read_text()
    metadata['git_dirty']=bool(subprocess.run(['git','status','--porcelain'],cwd=ROOT,text=True,capture_output=True).stdout.strip())
    metadata['devices']=execute([a.binary,'devices'],10)[1];(out/'manifest.json').write_text(json.dumps(metadata,indent=2))
    rows=[]
    for model_index,model in enumerate(a.models):
        path=Path(model).resolve();tag=f'{model_index:03d}_{path.stem}'
        try:checksum=hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:checksum=None
        inspect=execute([a.binary,'inspect',str(path)],10)[0]
        mps=raw/f'{tag}.mps'
        if path.suffix.lower() in ('.mps','.qps'):mps=path
        else:execute([a.binary,'convert',str(path),str(mps)],10)
        for solver in a.solvers.split(','):
            for run in range(-1 if solver=='cuda' else 0,a.runs):
                name=f'{tag}_{solver}_{"warmup" if run<0 else run}'
                if solver in ('cpu','cuda'):
                    command=[a.binary,'solve',str(path),'--device',solver,'--time-limit',str(a.time_limit),'--threads',str(a.threads),'--tol',str(a.tol),'--iterations',str(a.iterations)]
                    if a.no_scaling:command+=['--scaling-passes','0']
                    if a.no_restart:command+=['--no-restart']
                    if a.no_adaptive:command+=['--no-adaptive']
                elif solver=='highs':command=[sys.executable,str(ROOT/'benchmark/adapters/highs.py'),str(mps),'--time-limit',str(a.time_limit),'--threads',str(a.threads),'--tol',str(a.tol)]
                else:raise ValueError(f'Unknown solver {solver}')
                data,stdout,stderr,code,wall=execute(command,a.time_limit+30)
                (raw/f'{name}.stdout').write_text(stdout);(raw/f'{name}.stderr').write_text(stderr)
                accuracy=data.get('accuracy',{})
                if solver=='highs' and data.get('primal') and inspect.get('fingerprint'):
                    # Verify baseline primal using VANTAGE's independent original-space checker.
                    sense=1
                    if path.suffix.lower()=='.json':sense=1 if json.loads(path.read_text()).get('sense','min')=='min' else -1
                    else:
                        converted=raw/f'{tag}.json';execute([a.binary,'convert',str(path),str(converted)],10)
                        if converted.exists():sense=1 if json.loads(converted.read_text())['sense']=='min' else -1
                    dual=[-sense*v for v in data.get('row_dual',[])]
                    if len(dual)!=inspect['rows']:dual=[0.0]*inspect['rows']
                    sol=raw/f'{name}.solution.json';sol.write_text(json.dumps(dict(model=dict(fingerprint=inspect['fingerprint']),status=data['status'],primal=data['primal'],dual=dual)))
                    v=execute([a.binary,'verify',str(path),str(sol)],20)[0];accuracy=v.get('accuracy',{});data['independent_verification']=v
                perf=data.get('performance',{})
                record=dict(instance=path.name,solver=solver,run=run,warmup=run<0,status=data.get('status','UNKNOWN'),objective=data.get('objective'),primal_residual=accuracy.get('primal_residual'),dual_residual=accuracy.get('dual_residual'),kkt_error=accuracy.get('kkt_error'),iterations=perf.get('iterations'),end_to_end_seconds=perf.get('end_to_end_seconds'),iteration_seconds=perf.get('iteration_seconds'),process_wall_seconds=wall,rows=inspect.get('rows'),columns=inspect.get('columns'),nonzeros=inspect.get('nonzeros'),dataset_sha256=checksum)
                (raw/f'{name}.json').write_text(json.dumps(dict(record=record,command=command,exit_code=code,result=data),indent=2))
                if run>=0:rows.append(record)
                print(f"{path.name:24} {solver:6} run={run:2} {record['status']:18} objective={record['objective']} wall={wall:.4f}s",flush=True)
    with (out/'runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for instance,solver in dict.fromkeys((r['instance'],r['solver']) for r in rows):
        group=[r for r in rows if (r['instance'],r['solver'])==(instance,solver)]
        vals=[r['end_to_end_seconds'] for r in group if r['end_to_end_seconds'] is not None]
        summary.append(dict(instance=instance,solver=solver,optimal_runs=sum(r['status']=='OPTIMAL' for r in group),total_runs=len(group),statuses=[r['status'] for r in group],median_end_to_end_seconds=statistics.median(vals) if vals else None))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    subprocess.run([sys.executable,str(ROOT/'benchmark/report.py'),str(out)],check=True)

if __name__=='__main__':main()
