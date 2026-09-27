"""Persistent solves and explicitly authorized feasibility repair proposals."""
import copy
import json
import subprocess
import tempfile
from pathlib import Path
from . import _binary, solve


class SolverSession:
    """Sequential context-managed session; matrix structure is immutable.

    Bounds use None for infinity. Objective updates use original objective units.
    Compatible primal/dual vectors and simplex bases are reused; presolve still runs.
    """
    def __init__(self, model, *, binary=None):
        self._temp = tempfile.TemporaryDirectory(prefix='vantage-session-')
        self.binary = _binary(binary)
        self._stderr = tempfile.TemporaryFile(mode='w+t')
        path = model
        if hasattr(model, 'data') or isinstance(model, dict):
            path = Path(self._temp.name)/'model.json'
            path.write_text(json.dumps(model.data if hasattr(model, 'data') else model, allow_nan=False))
        try:
            self._process = subprocess.Popen([self.binary, 'session', str(path)], stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE, stderr=self._stderr, text=True)
            ready=self._process.stdout.readline()
            if not ready or json.loads(ready).get('status') != 'READY':
                self._stderr.seek(0)
                error=self._stderr.read() or 'Solver does not support persistent sessions'
                self.close()
                raise RuntimeError(error)
        except Exception:
            self._stderr.close(); self._temp.cleanup(); raise

    def _request(self, action, **fields):
        if self._process.poll() is not None:
            raise RuntimeError('Solver session is closed')
        self._process.stdin.write(json.dumps(dict(action=action, **fields), allow_nan=False)+'\n')
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            self._stderr.seek(0)
            raise RuntimeError(self._stderr.read() or 'Solver session terminated')
        result = json.loads(line)
        if result.get('status') == 'ERROR': raise ValueError(result.get('message'))
        return result

    def update_row_bounds(self, updates): return self._request('update', rows=updates)
    def update_variable_bounds(self, updates): return self._request('update', variables=updates)
    def update_objective(self, updates): return self._request('update', objective=updates)
    def solve(self, *, warm_start=True, device='cpu', method='auto', time_limit=60, tol=1e-6, threads=1, iterations=100000):
        return self._request('solve', warm_start=warm_start, device=device, method=method,
                             time_limit=time_limit, tol=tol, threads=threads, iterations=iterations)
    def close(self):
        if not self._process.stdin.closed: self._process.stdin.close()
        if self._process.poll() is None:
            try: self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try: self._process.wait(timeout=5)
                except subprocess.TimeoutExpired: self._process.kill(); self._process.wait()
        self._process.stdout.close()
        self._stderr.close(); self._temp.cleanup()
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def diagnose_infeasibility(model, *, binary=None, **options):
    """Return named Farkas support only when a separate verifier accepts the ray.

    The support also relies on model variable bounds. It is not a minimal IIS.
    """
    data = model.data if hasattr(model, 'data') else model
    with tempfile.TemporaryDirectory(prefix='vantage-diagnose-') as folder:
        path = Path(folder)/'model.json'; path.write_text(json.dumps(data, allow_nan=False))
        result = solve(path, binary=binary, **options)
        solution = Path(folder)/'result.json'; solution.write_text(json.dumps(result))
        proc = subprocess.run([_binary(binary),'verify',str(path),str(solution)], capture_output=True,text=True)
        try: verification = json.loads(proc.stdout)
        except ValueError: verification = dict(status='UNAVAILABLE')
        ray=result.get('certificate',{}).get('dual_ray',[])
        support=[dict(row=row['name'],multiplier=value) for row,value in zip(data['constraints'],ray) if value != 0]
        return dict(result=result, verification=verification, participating_rows=support,
                    conflict_verified=proc.returncode == 0 and verification.get('status') == 'VERIFIED_INFEASIBLE',
                    scope='Farkas evidence uses named rows and variable bounds; not a minimal IIS.')


def propose_repair(model, relaxable_rows, *, violation_limits=None, binary=None, **options):
    """Minimize weighted row-bound violation; all other rows/bounds stay hard.

    relaxable_rows maps explicitly authorized row names to positive penalties.
    Original operating cost is deliberately excluded from the repair objective.
    """
    import math
    data = copy.deepcopy(model.data if hasattr(model, 'data') else model)
    names = [r['name'] for r in data['constraints']]
    if len(set(names)) != len(names): raise ValueError('Repair requires unique row names')
    if not relaxable_rows or not set(relaxable_rows) <= set(names): raise ValueError('Specify known relaxable rows')
    for weight in relaxable_rows.values():
        if not math.isfinite(weight) or weight <= 0: raise ValueError('Penalties must be finite and positive')
    violation_limits=violation_limits or {}
    if not set(violation_limits)<=set(relaxable_rows): raise ValueError('Limits require an authorized row')
    for sides in violation_limits.values():
        for side,limit in sides.items():
            if side not in ('lb','ub') or not math.isfinite(limit) or limit<0:
                raise ValueError('Violation limits must be finite nonnegative lb/ub values')
    original_count = len(data['variables'])
    data['sense'] = 'min'; data['objective'] = dict(linear=[0.]*original_count)
    rows=[]; slacks=[]; used={v['name'] for v in data['variables']}
    for row in data['constraints']:
        if row['name'] not in relaxable_rows:
            rows.append(row); continue
        for side, sign in [('lb',1),('ub',-1)]:
            if row.get(side) is None: continue
            name=f'__repair_{len(slacks)}'
            while name in used: name+='_' 
            used.add(name)
            index=len(data['variables'])
            data['variables'].append(dict(name=name,lb=0,ub=violation_limits.get(row['name'],{}).get(side)))
            data['objective']['linear'].append(relaxable_rows[row['name']])
            part=copy.deepcopy(row);part['name']=f'{row["name"]}__repair_{side}'
            part.pop('ub' if side=='lb' else 'lb',None);part['coefficients'][name]=sign
            rows.append(part);slacks.append((row['name'],side,index))
    data['constraints']=rows
    with tempfile.TemporaryDirectory(prefix='vantage-repair-') as folder:
        path=Path(folder)/'repair.json';path.write_text(json.dumps(data,allow_nan=False))
        result=solve(path,binary=binary,**options)
        solution=Path(folder)/'solution.json';solution.write_text(json.dumps(result))
        proc=subprocess.run([_binary(binary),'verify',str(path),str(solution)],capture_output=True,text=True)
        try: verification=json.loads(proc.stdout)
        except ValueError: verification=dict(status='UNAVAILABLE')
    accepted=proc.returncode==0 and verification.get('status') in ('VERIFIED_OPTIMAL','VERIFIED_FEASIBLE')
    x=result.get('primal',[])
    changes=[dict(row=row,bound=side,violation=x[index]) for row,side,index in slacks if accepted and x[index]>1e-7]
    return dict(status='REPAIR_PROPOSAL' if accepted else 'NO_VERIFIED_REPAIR', repaired_model=data,
                changes=changes, result=result, verification=verification,
                original_primal=x[:original_count] if accepted else [],
                scope='Proposal for an explicitly relaxed model; not a feasible solution certificate for the original model.')
