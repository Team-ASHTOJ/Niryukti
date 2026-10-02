"""Mixed-precision Newton refinement, Dantzig-Wolfe decomposition, LP differentiation,
learned (GNN) branching and the controlled-English translator."""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = os.environ.get('NIRYUKTI_BINARY', str(ROOT / 'build' / 'niryukti'))
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'examples'))


def run(*arguments):
    result = subprocess.run([BINARY, *map(str, arguments)], capture_output=True, text=True)
    if result.returncode not in (0, 2):
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


def write(folder, name, document):
    path = Path(folder) / name
    path.write_text(json.dumps(document))
    return path


class MixedPrecisionTests(unittest.TestCase):
    def test_barrier_refinement_matches_fp64(self):
        for model in ('examples/coupled_dispatch.json', 'examples/refinery.json'):
            base = run('solve', ROOT / model, '--method', 'barrier', '--tol', '1e-8')
            if base['status'] == 'UNSUPPORTED':
                self.skipTest('barrier requires the sparse LU build')
            mixed = run('solve', ROOT / model, '--method', 'barrier', '--tol', '1e-8',
                        '--newton-precision', 'mixed')
            self.assertEqual(mixed['status'], 'OPTIMAL')
            self.assertAlmostEqual(mixed['objective'], base['objective'], places=5)
            stats = mixed['mixed_precision']
            self.assertGreater(stats['refined_solves'], 0)
            self.assertEqual(stats['fp64_fallbacks'], 0)
            self.assertLessEqual(stats['worst_relative_residual'], 1e-12)

    def test_invalid_precision_rejected(self):
        result = subprocess.run([BINARY, 'solve', ROOT / 'examples/production.json',
                                 '--newton-precision', 'fp16'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)


class DecompositionTests(unittest.TestCase):
    def test_multi_refinery_matches_direct_solve(self):
        direct = run('solve', ROOT / 'examples/multi_refinery.json')
        report = run('decompose', ROOT / 'examples/multi_refinery.json')
        self.assertEqual(report['status'], 'OPTIMAL')
        self.assertEqual(report['structure']['blocks'], 4)
        self.assertEqual(sorted(report['structure']['linking_row_names']),
                         ['demand_0', 'demand_1', 'jetty_0', 'jetty_1', 'jetty_2'])
        self.assertAlmostEqual(report['objective'], direct['objective'], places=5)
        self.assertLessEqual(report['absolute_gap'], 1e-6 * abs(direct['objective']))
        self.assertTrue(report['verification']['feasible'])

    def test_forced_border_and_larger_instance(self):
        import generate_multi_refinery as generator
        with tempfile.TemporaryDirectory() as folder:
            path = write(folder, 'm.json', generator.build(12))
            direct = run('solve', path)
            report = run('decompose', path, '--linking', 'jetty_*,demand_*')
        self.assertEqual(report['structure']['blocks'], 12)
        self.assertAlmostEqual(report['objective'], direct['objective'], places=4)

    def test_no_structure(self):
        dense = {'name': 'dense', 'sense': 'min',
                 'variables': [{'name': f'x{j}', 'lb': 0, 'ub': 5} for j in range(4)],
                 'objective': {'linear': [1, 2, 3, 4]},
                 'constraints': [{'name': f'r{i}', 'lb': 1 + i,
                                  'coefficients': {f'x{j}': 1 + (i + j) % 3 for j in range(4)}}
                                 for i in range(3)]}
        with tempfile.TemporaryDirectory() as folder:
            report = run('decompose', write(folder, 'd.json', dense))
        self.assertEqual(report['status'], 'NO_STRUCTURE')


class DifferentiationTests(unittest.TestCase):
    def test_vjp_matches_hand_derivation(self):
        with tempfile.TemporaryDirectory() as folder:
            upstream = write(folder, 'g.json', {'production_2': 1.0, 'inventory_1': 0.5})
            report = run('differentiate', ROOT / 'examples/production.json', '--upstream', upstream)
        vjp = report['vjp']
        # x: production_1=75 (capacity), inventory_1=75-d1, production_2=d2-inventory_1.
        self.assertAlmostEqual(vjp['row_lower'][0], 0.5)    # d/d(demand_1) = 1 - 0.5
        self.assertAlmostEqual(vjp['row_lower'][1], 1.0)    # d/d(demand_2)
        self.assertAlmostEqual(vjp['row_upper'][2], -0.5)   # d/d(capacity_1)
        self.assertEqual(vjp['c_method'], 'exact: zero almost everywhere')
        self.assertAlmostEqual(report['objective_gradient']['row_lower'][0], 5.8)

    def test_matrix_gradient_by_finite_difference(self):
        model = json.loads((ROOT / 'examples/production.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            upstream = write(folder, 'g.json', {'production_2': 1.0})
            report = run('differentiate', write(folder, 'm.json', model), '--upstream', upstream)
            names = report['variable_names']
            gradients = {(i, j): v for i, j, v in report['vjp']['A']}
            # Perturb A[balance_2, inventory_1] and re-solve.
            row, column = 1, names.index('inventory_1')
            h = 1e-3
            model['constraints'][row]['coefficients']['inventory_1'] += h
            moved = run('solve', write(folder, 'p.json', model))
            base = report['x'][names.index('production_2')]
            fd = (moved['primal'][names.index('production_2')] - base) / h
        self.assertAlmostEqual(gradients[(row, column)], fd, places=2)

    def test_perturbed_cost_gradient_is_finite(self):
        with tempfile.TemporaryDirectory() as folder:
            upstream = write(folder, 'g.json', [0, 1, 0, 0])
            report = run('differentiate', ROOT / 'examples/production.json', '--upstream', upstream,
                         '--samples', 50, '--sigma', 0.3)
        self.assertEqual(report['vjp']['c_samples_used'], 50)
        self.assertTrue(all(math.isfinite(v) for v in report['vjp']['c']))

    def test_torch_layer_if_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch not installed')
        os.environ.setdefault('NIRYUKTI_BINARY', BINARY)
        import niryukti
        model = json.loads((ROOT / 'examples/production.json').read_text())
        layer = niryukti.torch_lp_layer(model, samples=0, binary=BINARY)
        c, lower, upper = (torch.tensor(v, dtype=torch.float64, requires_grad=True)
                           for v in niryukti.lp_parameters(model))
        x = layer(c, lower, upper)
        (x[1] + 0.5 * x[2]).backward()
        self.assertAlmostEqual(lower.grad[0].item(), 0.5)
        self.assertAlmostEqual(upper.grad[2].item(), -0.5)


class LearnedBranchingTests(unittest.TestCase):
    def test_builtin_gnn_branching_is_correct(self):
        for model in ('examples/scheduling.json', 'examples/supply_chain.json'):
            fractional = run('solve', ROOT / model)
            learned = run('solve', ROOT / model, '--branching', 'gnn')
            self.assertEqual(learned['status'], 'OPTIMAL')
            self.assertAlmostEqual(learned['objective'], fractional['objective'], places=6)

    def test_training_produces_loadable_model(self):
        import generate_milp_benchmarks as generator
        import random
        rng = random.Random(4)
        with tempfile.TemporaryDirectory() as folder:
            paths = [write(folder, f'k{k}.json', generator.knapsack(rng, items=14, dimensions=3))
                     for k in range(3)]
            weights = Path(folder) / 'w.json'
            summary = run('train-branching', weights, *paths, '--dives', 3, '--epochs', 3)
            self.assertGreaterEqual(summary['metrics']['samples'], 10)
            saved = json.loads(weights.read_text())
            self.assertEqual(saved['format'], 'niryukti-branching-gnn')
            solved = run('solve', paths[0], '--branching', 'gnn', '--branching-model', weights)
            reference = run('solve', paths[0])
        self.assertAlmostEqual(solved['objective'], reference['objective'], places=6)


class TranslatorTests(unittest.TestCase):
    def test_refinery_plan(self):
        from vantage.language import translate
        model, report = translate((ROOT / 'examples/refinery_plan.txt').read_text())
        rows = {r['name']: r for r in model['constraints']}
        spec = rows['sulfur_spec']['coefficients']
        self.assertAlmostEqual(spec['light_crude'], -0.9)
        self.assertAlmostEqual(spec['heavy_crude'], 1.4)
        self.assertEqual((rows['petrol_demand']['lb'], rows['petrol_demand']['ub']), (120, 220))
        self.assertEqual(model['sense'], 'max')
        self.assertIn('binary run_hydrotreater', report['understood'])
        with tempfile.TemporaryDirectory() as folder:
            solved = run('solve', write(folder, 'plan.json', model))
        self.assertEqual(solved['status'], 'OPTIMAL')

    def test_unparsed_lines_are_reported_not_guessed(self):
        from vantage.language import translate, TranslationError
        with self.assertRaises(TranslationError) as caught:
            translate('variables a, b\nminimize a + b\nmake it cheap please\ncap: a + b >= 2')
        self.assertEqual([n for n, _ in caught.exception.errors], [3])

    def test_implicit_variables_warn(self):
        from vantage.language import translate
        model, report = translate('maximize 3 x + 2 y\nlimit: x + y at most 4\nx <= 3')
        self.assertEqual(len(report['warnings']), 2)
        self.assertEqual([v['name'] for v in model['variables']], ['x', 'y'])


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
