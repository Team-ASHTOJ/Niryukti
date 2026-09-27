"""Small Python API to the independent C++ engine through its structured CLI.

No numerical solver packages are imported. Calls run in isolated subprocesses.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path


def _binary(binary=None):
    return str(binary or os.environ.get('VANTAGE_BINARY') or Path(__file__).resolve().parents[2] / 'build/vantage')


def solve(path, *, device='auto', tol=1e-6, time_limit=60, threads=1, warm_start=None,
          allow_model_change=False, binary=None, iterations=100000, method='pdhg',
          scaling='ruiz', adaptive=True, branching='fractional', primal_weight='displacement',
          power_iterations=0, polishing=False, cuts=False, primal_heuristic="repair", cuda_graphs=False,
          gpu_indices="auto", matrix_precision="fp64", gpu_monitor=False, node_selection="best-bound", gpu_presolve=False,
          batch_strong_branching=False, checkpoint_out=None, resume=None, checkpoint_nodes=100):
    command=[_binary(binary),'solve',str(path),'--device',device,'--tol',str(tol),
             '--time-limit',str(time_limit),'--threads',str(threads),'--iterations',str(iterations),'--node-selection',node_selection]
    if method!='pdhg':command+=['--method',method]
    if scaling!='ruiz':command+=['--scaling',scaling]
    if branching!='fractional':command+=['--branching',branching]
    if primal_weight!='displacement':command+=['--primal-weight',primal_weight]
    if power_iterations:command+=['--power-iterations',str(power_iterations)]
    if polishing:command+=['--polishing']
    if cuts:command+=['--cuts']
    if primal_heuristic!='repair':command+=['--primal-heuristic',primal_heuristic]
    if gpu_monitor:command += ["--gpu-monitor"]
    if gpu_presolve:command += ["--gpu-presolve"]
    if batch_strong_branching:command += ["--batch-strong-branching"]
    if checkpoint_out:command += ["--checkpoint-out",str(checkpoint_out),"--checkpoint-nodes",str(checkpoint_nodes)]
    if resume:command += ["--resume",str(resume)]
    if cuda_graphs:command += ["--cuda-graphs"]
    if gpu_indices!="auto":command += ["--gpu-indices",gpu_indices]
    if matrix_precision!="fp64":command += ["--matrix-precision",matrix_precision]
    if not adaptive:command+=['--no-adaptive']
    with tempfile.TemporaryDirectory(prefix='vantage-api-') as temp:
        if warm_start is not None:
            if isinstance(warm_start,dict):
                warm=Path(temp)/'warm.json';warm.write_text(json.dumps(warm_start))
            else:warm=Path(warm_start)
            command+=['--warm-start',str(warm)]
            if allow_model_change:command+=['--allow-model-change']
        p=subprocess.run(command,capture_output=True,text=True)
        if p.returncode not in (0,2):raise RuntimeError(p.stderr.strip() or 'NIRYUKTI process failed')
        return json.loads(p.stdout)


class Model:
    def __init__(self,name='model'):
        self.data=dict(name=name,sense='min',variables=[],objective=dict(linear=[]),constraints=[])

    def add_var(self,name=None,lb=0,ub=None,integer=False,binary=False):
        name=name or f'x{len(self.data["variables"])}'
        if any(v['name']==name for v in self.data['variables']):raise ValueError('Duplicate variable name')
        self.data['variables'].append(dict(name=name,lb=lb,ub=1 if binary and ub is None else ub,
                                          type='binary' if binary else 'integer' if integer else 'continuous'))
        self.data['objective']['linear'].append(0)
        return name

    def add_constraint(self,coefficients,sense,rhs,name=None):
        if sense not in ('<=','>=','='):raise ValueError('Use <=, >=, or =')
        names={v['name'] for v in self.data['variables']}
        if not set(coefficients)<=names:raise ValueError('Unknown variable')
        self.data['constraints'].append(dict(name=name or f'r{len(self.data["constraints"])}',
                                            coefficients=coefficients,lb=rhs if sense in ('>=','=') else None,
                                            ub=rhs if sense in ('<=','=') else None))

    def set_objective(self,coefficients,sense='min',quadratic_diagonal=None,offset=0):
        if sense not in ('min','max'):raise ValueError('Use min or max')
        names=[v['name'] for v in self.data['variables']]
        if not set(coefficients)<=set(names):raise ValueError('Unknown variable')
        self.data['sense']=sense
        self.data['objective']=dict(linear=[coefficients.get(n,0) for n in names],offset=offset)
        if quadratic_diagonal is not None:
            if not set(quadratic_diagonal)<=set(names):raise ValueError('Unknown quadratic variable')
            self.data['objective']['quadratic_diagonal']=[quadratic_diagonal.get(n,0) for n in names]

    def write(self,path):Path(path).write_text(json.dumps(self.data,indent=2)+'\n')

    def solve(self,**options):
        with tempfile.TemporaryDirectory(prefix='vantage-model-') as temp:
            path=Path(temp)/'model.json';self.write(path);return solve(path,**options)


from .planning import SolverSession, diagnose_infeasibility, propose_repair
from .native import NativeSession
from .sensitivity import rhs_sensitivity
from .stability import stable_plan_model, replan
