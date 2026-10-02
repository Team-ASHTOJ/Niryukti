"""Differentiable LP layer.

`differentiate` returns exact basis derivatives of an LP optimum (bounds, matrix entries) and
an optional perturbed-optimizer estimate for costs. `torch_lp_layer` wraps it as a PyTorch
autograd function when PyTorch is installed; PyTorch is never required by the engine."""
import copy
import json
import math
import subprocess
import tempfile
from pathlib import Path


def _model_data(model):
    from . import Model
    if isinstance(model, Model):
        return copy.deepcopy(model.data)
    if isinstance(model, dict):
        return copy.deepcopy(model)
    return json.loads(Path(model).read_text())


def differentiate(model, upstream=None, *, samples=0, sigma=0.05, seed=1, time_limit=60, binary=None):
    """Derivatives of an LP optimum.

    `upstream` is dl/dx as a list (variable order) or {name: value}. Returns the engine report:
    `objective_gradient` always; `vjp` (row_lower/row_upper/variable_lower/variable_upper/A/c)
    when `upstream` is given. Costs use `samples` antithetic perturbed solves of relative scale
    `sigma` (0 samples returns the exact, almost-everywhere-zero cost derivative)."""
    from . import _binary
    data = _model_data(model)
    with tempfile.TemporaryDirectory(prefix='niryukti-diff-') as folder:
        folder = Path(folder)
        path = folder / 'model.json'
        path.write_text(json.dumps(data, allow_nan=False))
        command = [_binary(binary), 'differentiate', str(path), '--samples', str(samples),
                   '--sigma', str(sigma), '--seed', str(seed), '--time-limit', str(time_limit)]
        if upstream is not None:
            gradient = folder / 'upstream.json'
            gradient.write_text(json.dumps(upstream if isinstance(upstream, dict) else list(upstream)))
            command += ['--upstream', str(gradient)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode not in (0, 2):
            raise RuntimeError(result.stderr.strip() or 'NIRYUKTI process failed')
        return json.loads(result.stdout)


def _with_parameters(data, c, row_lower, row_upper):
    model = copy.deepcopy(data)
    model['objective']['linear'] = [float(v) for v in c]
    for row, lower, upper in zip(model['constraints'], row_lower, row_upper):
        row['lb'] = None if math.isinf(float(lower)) else float(lower)
        row['ub'] = None if math.isinf(float(upper)) else float(upper)
    return model


def lp_parameters(model):
    """(c, row_lower, row_upper) lists from a JSON model; infinite bounds become +/-inf."""
    data = _model_data(model)
    linear = data['objective']['linear']
    if isinstance(linear, dict):
        raise ValueError('Use an objective.linear array for differentiable models')
    lower = [-math.inf if r.get('lb') is None else float(r['lb']) for r in data['constraints']]
    upper = [math.inf if r.get('ub') is None else float(r['ub']) for r in data['constraints']]
    return list(map(float, linear)), lower, upper


def torch_lp_layer(model, *, samples=32, sigma=0.05, seed=1, binary=None):
    """Return f(c, row_lower, row_upper) -> x* as a PyTorch autograd function.

    Backward gives exact gradients for row bounds and a perturbed-optimizer estimate for c.
    For an equality row pass the same tensor as row_lower and row_upper; its gradient is
    reported under row_lower so autograd's sum is correct."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only without torch
        raise ImportError('torch_lp_layer requires PyTorch; the engine itself does not') from error
    data = _model_data(model)

    class LinearProgram(torch.autograd.Function):
        @staticmethod
        def forward(ctx, c, row_lower, row_upper):
            instance = _with_parameters(data, c.tolist(), row_lower.tolist(), row_upper.tolist())
            report = differentiate(instance, binary=binary)
            if report.get('status') != 'OPTIMAL_BASIS':
                raise RuntimeError(f"LP layer forward failed: {report.get('status')} {report.get('message', '')}")
            ctx.instance = instance
            return torch.tensor(report['x'], dtype=c.dtype)

        @staticmethod
        def backward(ctx, grad_x):
            report = differentiate(ctx.instance, grad_x.tolist(), samples=samples, sigma=sigma,
                                   seed=seed, binary=binary)
            vjp = report['vjp']
            as_tensor = lambda values: torch.tensor(values, dtype=grad_x.dtype)
            return as_tensor(vjp['c']), as_tensor(vjp['row_lower']), as_tensor(vjp['row_upper'])

    return LinearProgram.apply
