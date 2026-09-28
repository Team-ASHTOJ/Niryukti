#!/usr/bin/env python3
"""Local dashboard and process-isolated access to the NIRYUKTI CLI. Standard library only."""
import argparse
import csv
import io
import json
import math
import mimetypes
import os
import re
from pathlib import Path
import secrets
import signal
import statistics
import sys
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'dashboard/static'
BINARY = Path(os.environ.get('VANTAGE_BINARY', str(ROOT / 'build/vantage')))
STORE = ROOT / 'results/dashboard'
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.Lock()
JOBS = {}
UPLOADED = {}
GPU_SNAPSHOT = {}
GPU_LOCK = threading.Lock()
CATALOG = [
    ('coupled_dispatch', 'Coupled power dispatch', 'QP', 'Industrial', 'examples/coupled_dispatch.json', 'Synthetic convex sparse quadratic generation costs with cross-generator coupling.'),
    ('integer_dispatch', 'Integer power dispatch', 'MIQP', 'Industrial', 'examples/integer_dispatch.json', 'Synthetic integer dispatch with a convex quadratic cost and global gap.'),
    ('production', 'Production and inventory', 'LP', 'Industrial', 'examples/production.json', 'Synthetic two-period capacity, inventory and demand balances.'),
    ('refinery', 'Crude blending', 'LP', 'Industrial', 'examples/refinery.json', 'Allocate crude flows while meeting demand, capacity and sulfur specifications.'),
    ('dispatch', 'Power dispatch', 'QP', 'Industrial', 'examples/dispatch.json', 'Balance generator output with a separable convex quadratic cost.'),
    ('supply_chain', 'Supply chain', 'MILP', 'Industrial', 'examples/supply_chain.json', 'Choose facilities and transport flows under capacity and demand limits.'),
    ('scheduling', 'Refinery scheduling', 'MILP', 'Industrial', 'examples/scheduling.json', 'Coordinate production, startup decisions and inventory across periods.'),
    ('afiro', 'AFIRO', 'LP', 'Netlib', 'datasets/afiro.mps', 'A small public LP used for independent numerical validation.'),
    ('adlittle', 'ADLITTLE', 'LP', 'Netlib', 'datasets/adlittle.mps', 'A public LP with 56 rows and 97 variables.'),
    ('israel', 'ISRAEL', 'LP', 'Netlib', 'datasets/israel.mps', 'A harder public LP used to track convergence improvements.'),
    ('e226', 'E226', 'LP', 'Netlib', 'datasets/e226.mps', 'Numerical recovery is verified in the current repeated campaign; performance remains behind HiGHS.'),
    ('planted_1m', 'Million-variable sparse LP', 'LP', 'Scalability', 'datasets/public/stress/planted_1m.mps', 'Synthetic planted cyclic LP: 1,000,000 variables; known optimum 3,437,485. Single-run screening, not MRPL data.'),
    ('dfl001', 'DFL001', 'LP', 'Netlib', 'datasets/public/netlib/dfl001.mps', 'Public convergence stress case; CPU/CUDA PDHG compared with HiGHS.'),
    ('stocfor2', 'STOCFOR2', 'LP', 'Netlib', 'datasets/public/netlib/stocfor2.mps', 'Public stochastic-programming LP, retained whether solved or limited.'),
    ('rail507_lp', 'RAIL507 · LP relaxation', 'LP', 'MIPLIB relaxations', 'datasets/public/relaxations/rail507_lp.mps', 'Continuous relaxation of the public railway MILP; not an integer solution.'),
    ('rail03_lp', 'RAIL03 · LP relaxation', 'LP', 'MIPLIB relaxations', 'datasets/public/relaxations/rail03_lp.mps', 'Approximately 759,000 variables. Convergence stress; parser/setup costs are included.'),
    ('rail507', 'RAIL507', 'MILP', 'MIPLIB', 'datasets/public/miplib/rail507.mps', 'Public railway MILP compared against HiGHS and SCIP, with incumbent and global gap.'),
    ('rail03', 'RAIL03', 'MILP', 'MIPLIB', 'datasets/stress_public/miplib/rail03.mps', 'Large public railway MILP; single-run 60-second screening is not a full MIPLIB campaign.'),
    ('ns1644855', 'NS1644855', 'MILP', 'MIPLIB', 'datasets/stress_public/miplib/ns1644855.mps', 'Public MILP with over two million matrix nonzeros.'),
    ('refinery_large', 'Large-scale blending', 'LP', 'Scalability', 'datasets/refinery_large.mps', '46,720 variables across synthetic, largely independent refinery periods.'),
]


def gpu_telemetry():
    """Device-wide driver counters, not solver-attributed utilization or peak memory."""
    with GPU_LOCK:
        now=time.time()
        if now-GPU_SNAPSHOT.get('timestamp',0)<2:return GPU_SNAPSHOT.copy()
        snapshot=dict(timestamp=now,available=False,scope='Device-wide NVIDIA counters; includes other processes',devices=[])
        try:
            p=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2)
            if p.returncode==0:
                for line in p.stdout.splitlines():
                    values=next(csv.reader([line],skipinitialspace=True))
                    if len(values)==5:
                        snapshot['devices'].append(dict(name=values[0],utilization_percent=number(values[1]),memory_used_mb=number(values[2]),memory_total_mb=number(values[3]),temperature_c=number(values[4])))
                snapshot['available']=bool(snapshot['devices'])
        except (OSError,subprocess.TimeoutExpired):pass
        GPU_SNAPSHOT.clear();GPU_SNAPSHOT.update(snapshot)
        return snapshot.copy()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    return value


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError): return None


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def benchmarks():
    cases, sources = {}, []
    base = next((v for v in ['completion_final_20260927','completion_20260927','submission-final','phase2-final'] if (ROOT/'results'/v/'summary.json').exists()), None)
    campaigns = [base] if base else ['demo','netlib','scalability-final']
    campaigns += ['stress_lp_20260927/reverified','stress_mip_corrected_20260927','stress_highs_ipm_20260927']
    for suite in campaigns:
        folder = ROOT/'results'/suite
        if not (folder/'manifest.json').exists() or not ((folder/'runs.csv').exists() or (folder/'summary.json').exists()):
            # Legacy small fixtures may contain raw records only.
            if suite not in ('demo','netlib','scalability-final'):continue
            if not (folder/'manifest.json').exists():continue
        manifest = json.loads((folder/'manifest.json').read_text()); options=manifest.get('arguments',{})
        sources.append(dict(suite=suite,timestamp=manifest.get('timestamp'),threads=options.get('threads'),iterations=options.get('iterations',100000),tolerance=options.get('tol'),time_limit=options.get('time_limit'),runs=options.get('runs',3),method=options.get('method'),exploratory=options.get('runs',3)==1,binary_sha256=manifest.get('binary_sha256'),cpu=manifest.get('cpu_info'),devices=manifest.get('devices'),verification_replayed=bool(manifest.get('verification_replay'))))
        if (folder/'runs.csv').exists():
            with (folder/'runs.csv').open() as f:records=list(csv.DictReader(f))
        else:
            records=[]
            for path in sorted((folder/'raw').glob('*.json')):
                try:
                    raw=json.loads(path.read_text()); r=raw.get('record')
                    if r and not r.get('warmup') and r.get('run',-1)>=0:records.append(r)
                except (ValueError,OSError):continue
        groups={}
        for r in records:
            groups.setdefault(Path(r['instance']).stem,{}).setdefault(r['solver'],[]).append(r)
        for key,engines in groups.items():
            entry=next((v for v in CATALOG if v[0]==key),None)
            sample=next((r for vs in engines.values() for r in vs if number(r.get('columns')) is not None),next(iter(engines.values()))[0])
            case=cases.setdefault(key,dict(id=key,name=entry[1] if entry else key,type=entry[2] if entry else sample.get('problem_type') or 'LP',category=entry[3] if entry else 'Public benchmarks',suite=suite,suites=[],rows=number(sample.get('rows')),columns=number(sample.get('columns')),nonzeros=number(sample.get('nonzeros')),engines={}))
            case['suites'].append(suite)
            for field in ('rows','columns','nonzeros'):
                if case[field] is None:case[field]=number(sample.get(field))
            for engine,rs in engines.items():
                statuses=list(dict.fromkeys(r['status'] for r in rs))
                case['engines'][engine]=dict(status=statuses[0] if len(statuses)==1 else 'MIXED',statuses=statuses,runs=len(rs),solved=sum(r['status']=='OPTIMAL' for r in rs),suite=suite,exploratory=options.get('runs',3)==1,verification_statuses=list(dict.fromkeys(r.get('verification_status') for r in rs if r.get('verification_status'))),
                    seconds=median([number(r.get('end_to_end_seconds')) for r in rs]),iteration_seconds=median([number(r.get('iteration_seconds')) for r in rs]),wall_seconds=median([number(r.get('process_wall_seconds')) for r in rs]),objective=median([number(r.get('objective')) for r in rs]),primal=max((number(r.get('primal_residual')) for r in rs if number(r.get('primal_residual')) is not None),default=None),kkt=max((number(r.get('kkt_error')) for r in rs if number(r.get('kkt_error')) is not None),default=None),gap=median([number(r.get('mip_gap')) for r in rs]),best_bound=median([number(r.get('best_bound')) for r in rs]),nodes=median([number(r.get('nodes')) for r in rs]),iterations=median([number(r.get('iterations')) for r in rs]),min_seconds=min((number(r.get('end_to_end_seconds')) for r in rs if number(r.get('end_to_end_seconds')) is not None),default=None),max_seconds=max((number(r.get('end_to_end_seconds')) for r in rs if number(r.get('end_to_end_seconds')) is not None),default=None))
    for case in cases.values():
        baseline=next((case['engines'][k] for k in ('highs','highs-ipm','scip') if case['engines'].get(k,{}).get('status')=='OPTIMAL' and case['engines'][k].get('objective') is not None),{})
        for engine in case['engines'].values():
            obj,ref=engine['objective'],baseline.get('objective')
            engine['objective_error']=abs(obj-ref)/max(1,abs(ref)) if case['type'] in ('LP','QP') and engine['status']=='OPTIMAL' and obj is not None and ref is not None else None
    ordered=sorted(cases.values(),key=lambda c:next((i for i,v in enumerate(CATALOG) if v[0]==c['id']),99))
    names=['cpu','cuda','highs','highs-ipm','scip']
    counts={e:sum(c['engines'].get(e,{}).get('status')=='OPTIMAL' for c in ordered) for e in names}
    available={e:sum(e in c['engines'] for c in ordered) for e in names}
    large=cases.get('planted_1m',cases.get('refinery_large',{})).get('engines',{})
    speedup=large['cpu']['seconds']/large['cuda']['seconds'] if all(large.get(k,{}).get('status')=='OPTIMAL' and large[k].get('seconds') for k in ('cpu','cuda')) else None
    iteration_speedup=large['cpu']['iteration_seconds']/large['cuda']['iteration_seconds'] if all(large.get(k,{}).get('status')=='OPTIMAL' and large[k].get('iteration_seconds') for k in ('cpu','cuda')) else None
    errors=[e['objective_error'] for c in ordered for k,e in c['engines'].items() if k in ('cpu','cuda') and e['objective_error'] is not None]
    return dict(cases=ordered,sources=sources,counts=counts,available=available,total=len(ordered),speedup=speedup,iteration_speedup=iteration_speedup,speedup_model='planted_1m' if 'planted_1m' in cases else 'refinery_large',max_objective_error=max(errors,default=None))


def catalog():
    data = benchmarks()
    by_id = {c['id']: c for c in data['cases']}
    return [dict(id=v[0], name=v[1], type=v[2], category=v[3], path=v[4], description=v[5], available=(ROOT/v[4]).exists(),
                 rows=by_id.get(v[0], {}).get('rows'), columns=by_id.get(v[0], {}).get('columns'), nonzeros=by_id.get(v[0], {}).get('nonzeros')) for v in CATALOG]


def public_job(job, full=False):
    item={k:v for k,v in job.items() if k not in ('process','path')}
    if not full and isinstance(item.get('result'),dict):
        result=item['result'].copy();counts={}
        for key,value in result.items():
            if isinstance(value,list) and len(value)>256:
                counts[key]=len(value);result[key]=value[:256]
        if counts:
            result['preview']=dict(limit=256,total_lengths=counts,note='Preview only. Download the original result for complete solution vectors.')
        item['result']=result
    return item



def run_arena(job, options):
    """External benchmark orchestration only; never used as our solve engine."""
    folder=STORE/('arena-'+job['id'])
    python=ROOT/'.venv/bin/python'
    command=[str(python) if python.is_file() else sys.executable,str(ROOT/'benchmark/run.py'),str(job['path']),
             '--binary',str(BINARY),'--solvers','cpu,cuda,highs','--runs','3','--time-limit',str(options['time_limit']),
             '--threads',str(options['threads']),'--tol',str(options['tol']),'--iterations',str(options['iterations']),
             '--method',options['method'],'--branching',options['branching'],'--node-selection',options['node_selection'],'--primal-heuristic',options.get('primal_heuristic','repair'),
             '--output',str(folder)]
    for key,flag in [('cuts','--cuts'),('gpu_monitor','--gpu-monitor')]:
        if options.get(key):command.append(flag)
    for key,flag in [('adaptive','--no-adaptive'),('scaling','--no-scaling'),('restart','--no-restart')]:
        if not options.get(key,True):command.append(flag)
    job['state']='running';job['command']=command
    try:
        with (STORE/(job['id']+'.log')).open('w') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=log,start_new_session=True)
            with LOCK:
                job['process']=proc
                if job.get('cancel_requested'):os.killpg(proc.pid,signal.SIGINT)
            try:proc.wait(timeout=10*(options['time_limit']+30)+120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        groups={}
        for path in sorted((folder/'raw').glob('*.json')):
            raw=json.loads(path.read_text());record=raw.get('record',{})
            if not record or record.get('warmup') or record.get('run',-1)<0:continue
            groups.setdefault(record['solver'],[]).append(record)
        comparisons=[]
        for solver in ['cpu','cuda','highs']:
            records=groups.get(solver,[]);statuses=list(dict.fromkeys(r['status'] for r in records))
            comparisons.append(dict(solver=solver,status=statuses[0] if len(statuses)==1 else 'MIXED' if statuses else 'UNAVAILABLE',
                                    runs=len(records),optimal_runs=sum(r['status']=='OPTIMAL' for r in records),
                                    seconds=median([number(r.get('end_to_end_seconds')) for r in records]),
                                    objective=median([number(r.get('objective')) for r in records]),
                                    primal=max((number(r.get('primal_residual')) for r in records if number(r.get('primal_residual')) is not None),default=None),
                                    kkt=max((number(r.get('kkt_error')) for r in records if number(r.get('kkt_error')) is not None),default=None),
                                    mip_gap=median([number(r.get('mip_gap')) for r in records])))
        job['result']=dict(status='INTERRUPTED' if job.get('cancel_requested') else 'COMPARISON_COMPLETE' if proc.returncode==0 else 'COMPARISON_PARTIAL',
                           comparisons=comparisons,report=f"/arena/{job['id']}/index.html" if (folder/'index.html').exists() else None,
                           scope='Three sequential measured repetitions, CUDA warmup, equal budgets; failures retained. Device/algorithm choices can differ.')
    except (OSError,ValueError) as error:job['result']=dict(status='PROCESS_ERROR',message=str(error),comparisons=[])
    finally:
        with LOCK:
            job['state']='finished';job['finished_at']=time.time();job.pop('process',None)
        (STORE/(job['id']+'.json')).write_text(json.dumps(clean(public_job(job,full=True)),indent=2))


def run_job(job, options):
    command = [str(BINARY), 'solve', str(job['path']), '--device', options['device'], '--tol', str(options['tol']),
               '--time-limit', str(options['time_limit']), '--threads', str(options['threads']),
               '--iterations', str(options.get('iterations',100000)), '--verbose']
    command += ['--method', options.get('method', 'pdhg'), '--branching', options.get('branching', 'fractional'), '--node-selection', options.get('node_selection', 'best-bound'), '--primal-heuristic',options.get('primal_heuristic','repair')]
    if options.get('cuts'): command += ['--cuts']
    if options.get('gpu_monitor') and options['device']=='cuda': command += ['--gpu-monitor']
    if not options.get('scaling', True): command += ['--scaling-passes', '0']
    if not options.get('restart', True): command += ['--no-restart']
    if not options.get('adaptive', True): command += ['--no-adaptive']
    try:
        job['state'] = 'running'
        job['command'] = command
        log_path, out_path = STORE / f"{job['id']}.log", STORE / f"{job['id']}.stdout"
        with out_path.open('w') as out, log_path.open('w') as log:
            proc = subprocess.Popen(command, stdout=out, stderr=log, start_new_session=True)
            with LOCK:
                job['process'] = proc
                if job.get('cancel_requested'): proc.send_signal(signal.SIGINT)
            # A local UI request cannot create an unbounded worker.
            try: proc.wait(timeout=options['time_limit'] + 45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        raw = out_path.read_text()
        try: result = json.loads(raw)
        except ValueError:
            diagnostic=log_path.read_text()[-3000:]
            try: result=json.loads(diagnostic)
            except ValueError: result=dict(status='ERROR',message=diagnostic or 'Solver did not return a result.')
        if not isinstance(result,dict):result=dict(status='ERROR',message='Solver returned an invalid result object.')
        job['result'] = clean(result)
        job['state'] = 'finished'
    except Exception as exc:
        job.update(state='finished', result=dict(status='ERROR', message=str(exc)))
    finally:
        job.pop('process',None)
        job['finished_at'] = time.time()
        (STORE/f"{job['id']}.json").write_text(json.dumps(clean(public_job(job)), indent=2))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def send(self, status, data, content_type='application/json; charset=utf-8', filename=None):
        payload = json.dumps(clean(data), allow_nan=False).encode() if isinstance(data, (dict, list)) else data if isinstance(data, bytes) else str(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store' if content_type.startswith('application/json') else 'no-cache')
        if filename: self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers(); self.wfile.write(payload)

    def local_host(self):
        host = self.headers.get('Host', '').split(':')[0].lower()
        if host not in ('127.0.0.1', 'localhost'):
            self.send(403, dict(error='Local workspace host required.'))
            return False
        return True

    def do_GET(self):
        if not self.local_host(): return
        try:
            path = unquote(urlparse(self.path).path)
            if path.startswith('/api/docs/'):
                name=path.removeprefix('/api/docs/')
                if name not in ('solver_completion_20260927.md','stress_campaign_20260927.md','development_log.md'):
                    return self.send(404,dict(error='Documentation unavailable'))
                file=ROOT/'docs'/name
                if not file.is_file():return self.send(404,dict(error='Documentation unavailable'))
                return self.send(200,file.read_bytes(),'text/markdown; charset=utf-8',filename=name)
            if path == '/api/bootstrap':
                return self.send(200, dict(token=TOKEN, models=catalog(), benchmarks=benchmarks(), binary_available=BINARY.exists()))
            if path == '/api/system': return self.send(200,gpu_telemetry())
            if path == '/api/benchmarks': return self.send(200, benchmarks())
            if path == '/api/runs':
                with LOCK: jobs = [public_job(j) for j in JOBS.values()]
                return self.send(200, sorted(jobs, key=lambda j:j['started_at'], reverse=True))
            if path.startswith('/api/runs/'):
                parts = path.split('/'); key = parts[3]
                with LOCK: job = JOBS.get(key)
                if not job: return self.send(404, dict(error='Run not found'))
                if len(parts)>4 and parts[4] in ('certificate', 'verification_report'):
                    certificate = job.get('result', {}).get('verification_certificate')
                    if certificate is None: return self.send(404, dict(error='Certificate unavailable'))
                    value = certificate if parts[4]=='certificate' else certificate.get('verification', {})
                    return self.send(200, value, filename=parts[4]+'.json')
                if len(parts)>4 and parts[4]=='download':
                    return self.send(200, job.get('result', {}), filename=f'vantage-{key}.json')
                item=public_job(job).copy(); log=STORE/f'{key}.log'
                if log.exists():
                    text=log.read_text(); item['logs']=text[-16000:]
                    item['trace']='\n'.join(re.findall(r'iter=\d+ objective=\S+ primal=\S+ dual=\S+ gap=\S+', text))
                return self.send(200, item)
            if path == '/api/export.csv':
                out=io.StringIO(); writer=csv.writer(out);writer.writerow(['instance','backend','status','runs','optimal_runs','median_seconds','objective','primal_residual','kkt_error','mip_gap','relative_objective_error'])
                for c in benchmarks()['cases']:
                    for backend,e in c['engines'].items(): writer.writerow([c['id'],backend,e['status'],e['runs'],e['solved'],e['seconds'],e['objective'],e['primal'],e['kkt'] if c['type'] not in ('MILP','MIQP') else '',e['gap'],e['objective_error']])
                return self.send(200,out.getvalue(),'text/csv; charset=utf-8','vantage-benchmarks.csv')
            if path.startswith('/arena/'):
                parts=Path(path).parts
                if len(parts)<4 or parts[2] not in JOBS or JOBS[parts[2]].get('kind')!='arena':return self.send(404,dict(error='Arena report unavailable'))
                base=(STORE/('arena-'+parts[2])).resolve();file=(base/Path(*parts[3:])).resolve()
                if not file.is_relative_to(base) or file.suffix not in ('.html','.json','.csv','.stdout','.stderr'):return self.send(404,dict(error='Arena file unavailable'))
                if not file.is_file():return self.send(404,dict(error='Arena file unavailable'))
                return self.send(200,file.read_bytes(),mimetypes.guess_type(file)[0] or 'application/octet-stream')
            if path.startswith('/reports/'):

                rest=path.removeprefix('/reports/'); parts=Path(rest).parts
                if not parts or parts[0] not in ('demo','netlib','scalability-final','phase2-final','submission-final','completion_20260927','completion_final_20260927','stress_lp_20260927','stress_mip_corrected_20260927','stress_highs_ipm_20260927'): return self.send(404, dict(error='Unknown report'))
                file=(ROOT/'results'/rest).resolve()
                if not file.is_relative_to(ROOT/'results'/parts[0]) or file.suffix not in ('.html','.json','.csv','.stdout','.stderr'): return self.send(404,dict(error='File unavailable'))
            else:
                file=(STATIC/('index.html' if path=='/' else path.lstrip('/'))).resolve()
                if not file.is_relative_to(STATIC): return self.send(404,dict(error='Not found'))
            if not file.is_file(): return self.send(404,dict(error='Not found'))
            return self.send(200,file.read_bytes(),mimetypes.guess_type(file)[0] or 'application/octet-stream')
        except (OSError, ValueError) as exc: return self.send(500,dict(error=str(exc)))

    def do_POST(self):
        if not self.local_host(): return
        if self.headers.get('X-Vantage-Token') != TOKEN: return self.send(403,dict(error='Reload the dashboard before making changes.'))
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=6_000_000: return self.send(413,dict(error='Request must be smaller than 6 MB.'))
            body=json.loads(self.rfile.read(length));path=urlparse(self.path).path
            if not isinstance(body,dict):return self.send(400,dict(error='Request must be a JSON object.'))
            if path == '/api/upload':
                name=Path(body.get('name','')).name; suffix=Path(name).suffix.lower()
                if suffix not in ('.mps','.lp','.json','.qps','.qplib'): return self.send(400,dict(error='Choose an MPS, LP, JSON, QPS or supported QPLIB model.'))
                text=body.get('content')
                if not isinstance(text,str): return self.send(400,dict(error='Text model required.'))
                key=secrets.token_hex(8); file=STORE/(key+suffix);file.write_text(text)
                proc=subprocess.run([str(BINARY),'inspect',str(file)],capture_output=True,text=True,timeout=20)
                if proc.returncode: file.unlink(); return self.send(400,dict(error=json.loads(proc.stderr).get('message','Invalid model')))
                meta=json.loads(proc.stdout);UPLOADED[key]=file
                return self.send(200,dict(id=key,name=name,category='Uploaded',type=meta['type'],columns=meta['columns'],rows=meta['rows'],nonzeros=meta['nonzeros'],available=True,description='Your local model.'))
            if path == '/api/runs':
                key=body.get('model');entry=next((v for v in CATALOG if v[0]==key),None)
                model=ROOT/entry[4] if entry else UPLOADED.get(key)
                if not model or not model.is_file(): return self.send(400,dict(error='Model is unavailable.'))
                device=body.get('device','cpu');tol=float(body.get('tol',1e-6));limit=float(body.get('time_limit',30));threads=int(body.get('threads',4));iterations=int(body.get('iterations',100000))
                if device not in ('cpu','cuda','auto') or tol not in (1e-4,1e-6,1e-8) or not 1<=limit<=120 or not 1<=threads<=32 or not 1<=iterations<=2000000:
                    return self.send(400,dict(error='Invalid solver settings.'))
                kind=body.get('kind','solve')
                if kind not in ('solve','arena') or (kind=='arena' and limit>15):return self.send(400,dict(error='Arena supports at most 15 seconds per measured run.'))
                method=body.get('method','pdhg');branching=body.get('branching','fractional');node_selection=body.get('node_selection','best-bound');primal_heuristic=body.get('primal_heuristic','repair')
                if method not in ('auto','simplex','dual-simplex','barrier','concurrent','pdhg','rhpdhg','r2hpdhg') or branching not in ('fractional','reliability') or node_selection not in ('best-bound','depth-first','best-estimate') or primal_heuristic not in ('repair','pump','rins','local','all'):
                    return self.send(400,dict(error='Invalid algorithm settings.'))
                if method in ('simplex','dual-simplex') and device=='cuda':
                    return self.send(400,dict(error='The selected factorization method currently runs on CPU.'))
                if method in ('rhpdhg','r2hpdhg') and body.get('adaptive',True):
                    return self.send(400,dict(error='Halpern methods require adaptive step sizes to be disabled.'))
                with LOCK:
                    if any(j['state']!='finished' for j in JOBS.values()): return self.send(409,dict(error='A solve is already running. Cancel or wait for it to finish.'))
                    ident=secrets.token_hex(8)
                    job=dict(id=ident,kind=kind,model=key,name=entry[1] if entry else model.name,state='queued',started_at=time.time(),path=model)
                    JOBS[ident]=job
                options=dict(method=method,branching=branching,node_selection=node_selection,primal_heuristic=primal_heuristic,cuts=bool(body.get('cuts',False)),gpu_monitor=bool(body.get('gpu_monitor',False)),device=device,tol=tol,time_limit=limit,threads=threads,iterations=iterations,adaptive=bool(body.get('adaptive',True)),scaling=bool(body.get('scaling',True)),restart=bool(body.get('restart',True)))
                job['options']=options; threading.Thread(target=run_arena if kind=='arena' else run_job,args=(job,options),daemon=True).start()
                return self.send(202,public_job(job))
            if path.startswith('/api/runs/') and path.endswith('/cancel'):
                key=path.split('/')[3]
                with LOCK:
                    job=JOBS.get(key)
                    if not job: return self.send(404,dict(error='Run not found'))
                    job['cancel_requested']=True;proc=job.get('process')
                    if proc and proc.poll() is None:
                        if job.get('kind')=='arena':os.killpg(proc.pid,signal.SIGINT)
                        else:proc.send_signal(signal.SIGINT)
                return self.send(200,dict(ok=True))
            return self.send(404,dict(error='Unknown action'))
        except (ValueError,TypeError,OSError,subprocess.TimeoutExpired) as exc: return self.send(400,dict(error=str(exc)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8080);parser.add_argument('--host',choices=['127.0.0.1','0.0.0.0'],default='127.0.0.1');args=parser.parse_args()
    STORE.mkdir(parents=True,exist_ok=True)
    for path in STORE.glob('*.json'):
        try:
            job=json.loads(path.read_text())
            if isinstance(job,dict) and job.get('state')=='finished' and job.get('id'): JOBS[job['id']]=job
        except (ValueError,OSError): pass
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    print(f'NIRYUKTI dashboard → http://127.0.0.1:{args.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__=='__main__': main()
