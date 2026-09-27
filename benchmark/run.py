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
    p=argparse.ArgumentParser();p.add_argument('models',nargs='+');p.add_argument('--binary',default=str(ROOT/'build/vantage'));p.add_argument('--solvers',default='cpu,cuda,highs');p.add_argument('--verification-timeout',type=float,default=60);p.add_argument('--parse-timeout',type=float,default=60);p.add_argument('--runs',type=int,default=3);p.add_argument('--iterations',type=int,default=100000);p.add_argument('--time-limit',type=float,default=10);p.add_argument('--threads',type=int,default=1);p.add_argument('--tol',type=float,default=1e-6);p.add_argument('--output',default='results/latest');p.add_argument('--method',choices=['auto','simplex','dual-simplex','barrier','concurrent','pdhg','halpern','rhpdhg','r2hpdhg'],default='pdhg');p.add_argument('--scaling',choices=['ruiz','combined'],default='ruiz');p.add_argument('--branching',choices=['fractional','reliability'],default='fractional');p.add_argument('--no-scaling',action='store_true');p.add_argument('--no-restart',action='store_true');p.add_argument('--no-adaptive',action='store_true')
    p.add_argument('--primal-weight',choices=['displacement','pid'],default='displacement')
    p.add_argument('--power-iterations',type=int,default=0)
    p.add_argument('--polishing',action='store_true')
    p.add_argument('--cuts',action='store_true')
    p.add_argument('--node-selection',choices=['best-bound','depth-first','best-estimate'],default='best-bound')
    p.add_argument('--primal-heuristic',choices=['repair','pump','rins','local','all'],default='repair')
    p.add_argument("--cuda-graphs",action="store_true")
    p.add_argument("--gpu-monitor",action="store_true")
    p.add_argument("--gpu-indices",choices=["auto","32","64"],default="auto")
    p.add_argument("--matrix-precision",choices=["fp64","mixed"],default="fp64")
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
    metadata['load_average']=list(os.getloadavg()) if hasattr(os,'getloadavg') else None
    metadata['background_cpu_snapshot']=execute(['ps','-eo','comm,pcpu','--sort=-pcpu'],10)[1].splitlines()[:12]
    metadata['measurement_scope']='Local machine; consult background CPU snapshot before interpreting timing'
    metadata['cpu_info']=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),platform.processor()) if Path('/proc/cpuinfo').exists() else platform.processor()
    metadata['ram_bytes']=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES') if hasattr(os,'sysconf') else None
    metadata['compiler']=execute(['c++','--version'],10)[1]
    metadata['cuda_toolkit']=execute(['nvcc','--version'],10)[1]
    metadata['gpu_state']=execute(['nvidia-smi','--query-gpu=name,driver_version,memory.total,power.limit','--format=csv'],10)[1]
    cache=Path(a.binary).parent/'CMakeCache.txt'
    if cache.exists():metadata['cmake_cache']=cache.read_text()
    source_files=[ROOT/'CMakeLists.txt']+[f for base in ['src','include','app','benchmark','scripts'] for f in (ROOT/base).rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.pyc','.pyo')]
    metadata['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
    for f in source_files:
        archived=out/'source'/f.relative_to(ROOT);archived.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,archived)
    metadata['git_dirty']=bool(subprocess.run(['git','status','--porcelain'],cwd=ROOT,text=True,capture_output=True).stdout.strip())
    metadata['devices']=execute([a.binary,'devices'],10)[1];(out/'manifest.json').write_text(json.dumps(metadata,indent=2))
    rows=[]
    for model_index,model in enumerate(a.models):
        path=Path(model).resolve();tag=f'{model_index:03d}_{path.stem}'
        try:checksum=hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:checksum=None
        if checksum is not None:shutil.copy2(path,raw/f'{tag}.input{path.suffix}')
        inspect=execute([a.binary,'inspect',str(path)],a.parse_timeout)[0]
        mps=raw/f'{tag}.mps'
        if path.suffix.lower() in ('.mps','.qps'):mps=path
        else:execute([a.binary,'convert',str(path),str(mps)],a.parse_timeout)
        for solver in a.solvers.split(','):
            for run in range(-1 if solver=='cuda' else 0,a.runs):
                name=f'{tag}_{solver}_{"warmup" if run<0 else run}'
                if solver in ('cpu','cuda'):
                    command=[a.binary,'solve',str(path),'--device',solver,'--time-limit',str(a.time_limit),'--threads',str(a.threads),'--tol',str(a.tol),'--iterations',str(a.iterations)]
                    command+=['--method',a.method]
                    if a.scaling!='ruiz':command+=['--scaling',a.scaling]
                    if a.branching!='fractional':command+=['--branching',a.branching]
                    if a.node_selection!='best-bound':command+=['--node-selection',a.node_selection]
                    if a.primal_weight!='displacement':command+=['--primal-weight',a.primal_weight]
                    if a.power_iterations:command+=['--power-iterations',str(a.power_iterations)]
                    if a.polishing:command+=['--polishing']
                    if a.cuts:command+=['--cuts']
                    if a.primal_heuristic!='repair':command+=['--primal-heuristic',a.primal_heuristic]
                    if a.no_scaling:command+=['--scaling-passes','0']
                    if a.no_restart:command+=['--no-restart']
                    if a.no_adaptive:command+=['--no-adaptive']
                elif solver=='scip':command=[sys.executable,str(ROOT/'benchmark/adapters/scip.py'),str(mps),'--time-limit',str(a.time_limit),'--threads',str(a.threads),'--tol',str(a.tol)]
                elif solver in ('highs','highs-ipm'):command=[sys.executable,str(ROOT/'benchmark/adapters/highs.py'),str(mps),'--time-limit',str(a.time_limit),'--threads',str(a.threads),'--tol',str(a.tol)]
                else:raise ValueError(f'Unknown solver {solver}')
                if solver=='highs-ipm':command+=['--method','ipm']
                if solver=='cuda':
                    if a.cuda_graphs:command+=['--cuda-graphs']
                    if a.gpu_monitor:command+=['--gpu-monitor']
                    if a.gpu_indices!='auto':command+=['--gpu-indices',a.gpu_indices]
                    if a.matrix_precision!='fp64':command+=['--matrix-precision',a.matrix_precision]
                data,stdout,stderr,code,wall=execute(command,a.time_limit+30)
                (raw/f'{name}.stdout').write_text(stdout);(raw/f'{name}.stderr').write_text(stderr)
                accuracy=data.get('accuracy',{})
                if solver in ('cpu','cuda') and data.get('model',{}).get('fingerprint'):
                    # Re-read the serialized result in a separate verifier process.
                    sol=raw/f'{name}.solution.json';sol.write_text(stdout)
                    verification_command=[a.binary,'verify',str(path),str(sol)]
                    v,vout,verr,vcode,_=execute(verification_command,a.verification_timeout)
                    (raw/f'{name}.verify.stdout').write_text(vout)
                    (raw/f'{name}.verify.stderr').write_text(verr)
                    data['independent_verification']=v
                    data['verification_process']=dict(command=verification_command,exit_code=vcode,tolerance=1e-6)
                if solver in ('highs','highs-ipm','scip') and data.get('primal') and inspect.get('fingerprint'):
                    # Verify baseline primal using NIRYUKTI's independent original-space checker.
                    sense=1
                    if path.suffix.lower()=='.json':sense=1 if json.loads(path.read_text()).get('sense','min')=='min' else -1
                    else:
                        converted=raw/f'{tag}.json';execute([a.binary,'convert',str(path),str(converted)],a.parse_timeout)
                        if converted.exists():sense=1 if json.loads(converted.read_text())['sense']=='min' else -1
                    dual=[-sense*v for v in data.get('row_dual',[])]
                    if len(dual)!=inspect['rows']:dual=[0.0]*inspect['rows']
                    sol=raw/f'{name}.solution.json';sol.write_text(json.dumps(dict(model=dict(fingerprint=inspect['fingerprint']),status=data['status'],objective=data.get('objective'),primal=data['primal'],dual=dual)))
                    v=execute([a.binary,'verify',str(path),str(sol)],a.verification_timeout)[0];accuracy=v.get('accuracy',{});data['independent_verification']=v
                reported_status=data.get('status','UNKNOWN')
                status=reported_status
                verification_status=data.get('independent_verification',{}).get('status')
                accepted_verification=('VERIFIED_FEASIBLE',) if inspect.get('type') in ('MILP','MIQP') else ('VERIFIED_OPTIMAL',)
                if reported_status=='OPTIMAL' and verification_status not in accepted_verification:
                    status='VERIFICATION_FAILED'
                perf=data.get('performance',{})
                record=dict(instance=path.name,solver=solver,run=run,warmup=run<0,status=status,reported_status=reported_status,verification_status=verification_status,problem_type=inspect.get('type'),objective=data.get('objective'),primal_residual=accuracy.get('primal_residual'),dual_residual=accuracy.get('dual_residual'),kkt_error=accuracy.get('kkt_error'),iterations=perf.get('iterations'),end_to_end_seconds=perf.get('end_to_end_seconds'),iteration_seconds=perf.get('iteration_seconds'),process_wall_seconds=wall,rows=inspect.get('rows'),columns=inspect.get('columns'),nonzeros=inspect.get('nonzeros'),dataset_sha256=checksum,method_selected=data.get('selection',{}).get('method'),device_reason=data.get('selection',{}).get('reason'),monitor_checks=perf.get('monitor_checks'),host_candidate_checks=perf.get('host_candidate_checks'),skipped_candidate_checks=perf.get('skipped_candidate_checks'),mip_gap=data.get('mip',{}).get('relative_gap',data.get('mip_gap')),best_bound=data.get('mip',{}).get('best_bound',data.get('best_bound')),nodes=data.get('mip',{}).get('nodes',data.get('nodes')))
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
    subprocess.run([sys.executable,str(ROOT/'benchmark/profiles.py'),str(out/'runs.csv')],check=True)

if __name__=='__main__':main()
