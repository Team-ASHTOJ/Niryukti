"""Post-optimal analysis, conflict diagnosis, bilinear global optimization and learned
method selection. Each call runs the C++ engine in an isolated subprocess."""
import json
import subprocess
import tempfile
from pathlib import Path


def _run(arguments, model, *, binary=None, solution=None):
    from . import _binary, Model
    with tempfile.TemporaryDirectory(prefix='niryukti-analysis-') as folder:
        folder = Path(folder)
        if isinstance(model, (Model, dict)):
            path = folder / 'model.json'
            path.write_text(json.dumps(model.data if isinstance(model, Model) else model, allow_nan=False))
        else:
            path = Path(model)
        command = [_binary(binary), arguments[0], str(path)]
        if solution is not None:
            if isinstance(solution, dict):
                saved = folder / 'solution.json'
                saved.write_text(json.dumps(solution))
                solution = saved
            command.append(str(solution))
        command += [str(a) for a in arguments[1:]]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode not in (0, 2):
            raise RuntimeError(result.stderr.strip() or 'NIRYUKTI process failed')
        return json.loads(result.stdout)


def sensitivity(model, solution=None, *, time_limit=60, binary=None):
    """Basis-exact shadow prices, reduced costs and RHS/objective ranging for an LP.

    MILPs are analysed with integer variables fixed at the incumbent. `solution` may be a
    saved result (path or dict); otherwise the model is solved first."""
    return _run(['sensitivity', '--time-limit', time_limit], model, binary=binary, solution=solution)


def find_iis(model, *, time_limit=60, binary=None):
    """Irreducible infeasible subsystem with verified Farkas evidence and repair proposals."""
    return _run(['iis', '--time-limit', time_limit], model, binary=binary)


def solve_global(model, *, time_limit=60, node_limit=100000, mip_gap=1e-4, binary=None):
    """McCormick spatial branch-and-bound for bilinear models (e.g. pooling).

    Returns a CERTIFIED_GLOBAL / BOUNDED / LOCAL_ONLY badge with the proven gap."""
    return _run(['global', '--time-limit', time_limit, '--node-limit', node_limit,
                 '--mip-gap', mip_gap], model, binary=binary)


def recommend(model, history, *, binary=None):
    """Rank methods using verified outcomes of structurally similar past solves."""
    return _run(['recommend', '--history', history], model, binary=binary)


def decompose(model, *, linking_rows=None, detect_only=False, time_limit=60, binary=None):
    """Dantzig-Wolfe decomposition; `linking_rows` names the border (trailing '*' = prefix)."""
    arguments = ['decompose', '--time-limit', time_limit]
    if linking_rows:
        arguments += ['--linking', ','.join(linking_rows)]
    if detect_only:
        arguments.append('--detect-only')
    return _run(arguments, model, binary=binary)
