#!/usr/bin/env python3
"""Local dashboard and process-isolated access to the NIRYUKTI CLI. Standard library only."""
import argparse
import csv
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import secrets
import signal
import statistics
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'dashboard/static'
BINARY = ROOT / 'build/vantage'
STORE = ROOT / 'results/dashboard'
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.Lock()
JOBS = {}
UPLOADED = {}
CATALOG = [
    ('refinery', 'Crude blending', 'LP', 'Industrial', 'examples/refinery.json', 'Allocate crude flows while meeting demand, capacity and sulfur specifications.'),
    ('dispatch', 'Power dispatch', 'QP', 'Industrial', 'examples/dispatch.json', 'Balance generator output with a separable convex quadratic cost.'),
    ('supply_chain', 'Supply chain', 'MILP', 'Industrial', 'examples/supply_chain.json', 'Choose facilities and transport flows under capacity and demand limits.'),
    ('scheduling', 'Refinery scheduling', 'MILP', 'Industrial', 'examples/scheduling.json', 'Coordinate production, startup decisions and inventory across periods.'),
    ('afiro', 'AFIRO', 'LP', 'Netlib', 'datasets/afiro.mps', 'A small public LP used for independent numerical validation.'),
    ('adlittle', 'ADLITTLE', 'LP', 'Netlib', 'datasets/adlittle.mps', 'A public LP with 56 rows and 97 variables.'),
    ('israel', 'ISRAEL', 'LP', 'Netlib', 'datasets/israel.mps', 'A harder public LP used to track convergence improvements.'),
    ('e226', 'E226', 'LP', 'Netlib', 'datasets/e226.mps', 'A harder public instance retained in the report despite incomplete convergence.'),
    ('refinery_large', 'Large-scale blending', 'LP', 'Scalability', 'datasets/refinery_large.mps', '46,720 variables across synthetic, largely independent refinery periods.'),
]

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
    cases = {}
    sources = []
    # Prefer the public campaign for AFIRO; never count its demo repetitions as a new case.
    campaigns = ['phase2-final'] if (ROOT/'results/phase2-final/summary.json').exists() else ['demo', 'netlib', 'scalability-final']
    for suite in campaigns:
        folder = ROOT / 'results' / suite
        if not (folder / 'manifest.json').exists(): continue
        manifest = json.loads((folder / 'manifest.json').read_text())
        sources.append(dict(suite=suite, timestamp=manifest.get('timestamp'), threads=manifest.get('arguments', {}).get('threads'),
                            iterations=manifest.get('arguments', {}).get('iterations',100000), tolerance=manifest.get('arguments', {}).get('tol'), time_limit=manifest.get('arguments', {}).get('time_limit'),
                            binary_sha256=manifest.get('binary_sha256'), cpu=manifest.get('cpu_info'), devices=manifest.get('devices')))
        groups = {}
        for path in sorted((folder / 'raw').glob('*.json')):
            try: raw = json.loads(path.read_text())
            except (ValueError, OSError): continue
            r = raw.get('record')
            if not r or r.get('warmup') or r.get('run', -1) < 0: continue
            key = Path(r['instance']).stem
            groups.setdefault(key, {}).setdefault(r['solver'], []).append((r, raw.get('result', {})))
        for key, engines in groups.items():
            entry = next((v for v in CATALOG if v[0] == key), None)
            sample = next(iter(engines.values()))[0][0]
            case = dict(id=key, name=entry[1] if entry else key, type=entry[2] if entry else 'LP', category=entry[3] if entry else 'Other',
                        suite=suite, rows=sample.get('rows'), columns=sample.get('columns'), nonzeros=sample.get('nonzeros'), engines={})
            for engine, values in engines.items():
                records = [r for r, _ in values]
                statuses = list(dict.fromkeys(r['status'] for r in records))
                case['engines'][engine] = dict(status=statuses[0] if len(statuses) == 1 else 'MIXED', statuses=statuses,
                    runs=len(records), solved=sum(r['status']=='OPTIMAL' for r in records),
                    seconds=median([number(r.get('end_to_end_seconds')) for r in records]),
                    iteration_seconds=median([number(r.get('iteration_seconds')) for r in records]),
                    wall_seconds=median([number(r.get('process_wall_seconds')) for r in records]),
                    objective=median([number(r.get('objective')) for r in records]),
                    primal=max((number(r.get('primal_residual')) for r in records if number(r.get('primal_residual')) is not None), default=None),
                    kkt=max((number(r.get('kkt_error')) for r in records if number(r.get('kkt_error')) is not None), default=None),
                    gap=median([number(d.get('mip', {}).get('relative_gap', d.get('mip_gap'))) for _, d in values]),
                    iterations=median([number(r.get('iterations')) for r in records]),
                    min_seconds=min((number(r.get('end_to_end_seconds')) for r in records if number(r.get('end_to_end_seconds')) is not None), default=None),
                    max_seconds=max((number(r.get('end_to_end_seconds')) for r in records if number(r.get('end_to_end_seconds')) is not None), default=None))
            baseline = case['engines'].get('highs', {})
            for engine in case['engines'].values():
                obj, ref = engine['objective'], baseline.get('objective')
                engine['objective_error'] = abs(obj-ref)/max(1, abs(ref)) if obj is not None and ref is not None else None
            cases[key] = case
    ordered = sorted(cases.values(), key=lambda x: next((i for i,v in enumerate(CATALOG) if v[0]==x['id']), 99))
    counts = {engine: sum(c['engines'].get(engine, {}).get('status') == 'OPTIMAL' for c in ordered) for engine in ['cpu','cuda','highs']}
    large = cases.get('refinery_large', {}).get('engines', {})
    speedup = large['cpu']['seconds']/large['cuda']['seconds'] if all(large.get(k, {}).get('status') == 'OPTIMAL' and large[k].get('seconds') for k in ('cpu','cuda')) else None
    errors = [e['objective_error'] for c in ordered for k,e in c['engines'].items() if k!='highs' and e['status']=='OPTIMAL' and e['objective_error'] is not None]
    return dict(cases=ordered, sources=sources, counts=counts, total=len(ordered), speedup=speedup, max_objective_error=max(errors, default=None))


def catalog():
    data = benchmarks()
    by_id = {c['id']: c for c in data['cases']}
    return [dict(id=v[0], name=v[1], type=v[2], category=v[3], path=v[4], description=v[5], available=(ROOT/v[4]).exists(),
                 rows=by_id.get(v[0], {}).get('rows'), columns=by_id.get(v[0], {}).get('columns'), nonzeros=by_id.get(v[0], {}).get('nonzeros')) for v in CATALOG]


def public_job(job):
    return {k:v for k,v in job.items() if k not in ('process', 'path')}


def run_job(job, options):
    command = [str(BINARY), 'solve', str(job['path']), '--device', options['device'], '--tol', str(options['tol']),
               '--time-limit', str(options['time_limit']), '--threads', str(options['threads']),
               '--iterations', str(options.get('iterations',100000)), '--verbose']
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
        except ValueError: result = dict(status='ERROR', message=log_path.read_text()[-3000:] or 'Solver did not return a result.')
        job['result'] = clean(result)
        job['state'] = 'finished'
    except Exception as exc:
        job.update(state='finished', result=dict(status='ERROR', message=str(exc)))
    finally:
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
            if path == '/api/bootstrap':
                return self.send(200, dict(token=TOKEN, models=catalog(), benchmarks=benchmarks(), binary_available=BINARY.exists()))
            if path == '/api/benchmarks': return self.send(200, benchmarks())
            if path == '/api/runs':
                with LOCK: jobs = [public_job(j) for j in JOBS.values()]
                return self.send(200, sorted(jobs, key=lambda j:j['started_at'], reverse=True))
            if path.startswith('/api/runs/'):
                parts = path.split('/'); key = parts[3]
                with LOCK: job = JOBS.get(key)
                if not job: return self.send(404, dict(error='Run not found'))
                if len(parts)>4 and parts[4]=='download':
                    return self.send(200, job.get('result', {}), filename=f'vantage-{key}.json')
                item=public_job(job).copy(); log=STORE/f'{key}.log'
                if log.exists(): item['logs']=log.read_text()[-16000:]
                return self.send(200, item)
            if path == '/api/export.csv':
                out=io.StringIO(); writer=csv.writer(out);writer.writerow(['instance','backend','status','runs','optimal_runs','median_seconds','objective','primal_residual','kkt_error','mip_gap','relative_objective_error'])
                for c in benchmarks()['cases']:
                    for backend,e in c['engines'].items(): writer.writerow([c['id'],backend,e['status'],e['runs'],e['solved'],e['seconds'],e['objective'],e['primal'],e['kkt'] if c['type']!='MILP' else '',e['gap'],e['objective_error']])
                return self.send(200,out.getvalue(),'text/csv; charset=utf-8','vantage-benchmarks.csv')
            if path.startswith('/reports/'):
                rest=path.removeprefix('/reports/'); parts=Path(rest).parts
                if not parts or parts[0] not in ('demo','netlib','scalability-final','phase2-final'): return self.send(404, dict(error='Unknown report'))
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
            if path == '/api/upload':
                name=Path(body.get('name','')).name; suffix=Path(name).suffix.lower()
                if suffix not in ('.mps','.lp','.json'): return self.send(400,dict(error='Choose an MPS, LP or JSON model.'))
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
                with LOCK:
                    if any(j['state']!='finished' for j in JOBS.values()): return self.send(409,dict(error='A solve is already running. Cancel or wait for it to finish.'))
                    ident=secrets.token_hex(8)
                    job=dict(id=ident,model=key,name=entry[1] if entry else model.name,state='queued',started_at=time.time(),path=model)
                    JOBS[ident]=job
                options=dict(device=device,tol=tol,time_limit=limit,threads=threads,iterations=iterations,adaptive=bool(body.get('adaptive',True)),scaling=bool(body.get('scaling',True)),restart=bool(body.get('restart',True)))
                job['options']=options; threading.Thread(target=run_job,args=(job,options),daemon=True).start()
                return self.send(202,public_job(job))
            if path.startswith('/api/runs/') and path.endswith('/cancel'):
                key=path.split('/')[3]
                with LOCK:
                    job=JOBS.get(key)
                    if not job: return self.send(404,dict(error='Run not found'))
                    job['cancel_requested']=True;proc=job.get('process')
                    if proc and proc.poll() is None: proc.send_signal(signal.SIGINT)
                return self.send(200,dict(ok=True))
            return self.send(404,dict(error='Unknown action'))
        except (ValueError,TypeError,OSError,subprocess.TimeoutExpired) as exc: return self.send(400,dict(error=str(exc)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8080);args=parser.parse_args()
    STORE.mkdir(parents=True,exist_ok=True)
    for path in STORE.glob('*.json'):
        try:
            job=json.loads(path.read_text())
            if isinstance(job,dict) and job.get('state')=='finished' and job.get('id'): JOBS[job['id']]=job
        except (ValueError,OSError): pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'NIRYUKTI dashboard → http://127.0.0.1:{args.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__=='__main__': main()
