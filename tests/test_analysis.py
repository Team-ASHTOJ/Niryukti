"""Sensitivity ranging, IIS diagnosis, bilinear global optimization, assurance badges and
learned method selection. Expected values are derived by hand or from the literature."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = os.environ.get('NIRYUKTI_BINARY', str(ROOT / 'build' / 'niryukti'))


def run(*arguments):
    result = subprocess.run([BINARY, *map(str, arguments)], capture_output=True, text=True)
    if result.returncode not in (0, 2):
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


def write(folder, name, document):
    path = Path(folder) / name
    path.write_text(json.dumps(document))
    return path


class SensitivityTests(unittest.TestCase):
    def test_production_prices_and_ranges(self):
        # Period-1 capacity binds: extra period-1 demand is met from period 2 (6 - 0.2).
        report = run('sensitivity', ROOT / 'examples/production.json')
        self.assertEqual(report['status'], 'OPTIMAL_BASIS')
        self.assertAlmostEqual(report['objective'], 517)
        rows = {r['name']: r for r in report['constraints']}
        self.assertAlmostEqual(rows['balance_1']['shadow_price'], 5.8)
        self.assertAlmostEqual(rows['balance_2']['shadow_price'], 6)
        self.assertAlmostEqual(rows['capacity_1']['shadow_price'], -1.8)
        self.assertEqual(rows['capacity_2']['status'], 'not_binding')
        for got, want in zip(rows['balance_1']['upper_bound_range'], [5, 75]):
            self.assertAlmostEqual(got, want)
        variables = {v['name']: v for v in report['variables']}
        self.assertAlmostEqual(variables['inventory_2']['reduced_cost'], 6.2)
        self.assertAlmostEqual(variables['production_1']['objective_coefficient_range'][1], 5.8)
        self.assertIsNone(variables['production_1']['objective_coefficient_range'][0])
        self.assertTrue(report['verification']['passed'])

    def test_shadow_price_matches_resolve(self):
        report = run('sensitivity', ROOT / 'examples/production.json')
        model = json.loads((ROOT / 'examples/production.json').read_text())
        for row in model['constraints']:
            if row['name'] == 'balance_2':
                row['lb'] = row['ub'] = row['ub'] + 1
        with tempfile.TemporaryDirectory() as folder:
            changed = run('solve', write(folder, 'm.json', model))
        self.assertAlmostEqual(changed['objective'] - report['objective'], 6, places=6)

    def test_degenerate_vertex_repairs_basis(self):
        model = {'name': 'degenerate', 'sense': 'max',
                 'variables': [{'name': 'x', 'lb': 0}, {'name': 'y', 'lb': 0}],
                 'objective': {'linear': [1, 0.1]},
                 'constraints': [
                     {'name': 'x_cap', 'coefficients': {'x': 1}, 'ub': 1},
                     {'name': 'y_cap', 'coefficients': {'y': 1}, 'ub': 1},
                     {'name': 'total', 'coefficients': {'x': 1, 'y': 1}, 'ub': 2}]}
        with tempfile.TemporaryDirectory() as folder:
            path = write(folder, 'm.json', model)
            solution = Path(folder) / 's.json'
            run('solve', path, '--json-out', solution)
            report = run('sensitivity', path, solution)
        self.assertTrue(report['basis']['dual_feasible'])
        self.assertGreaterEqual(report['basis']['degenerate_pivots'], 1)
        prices = {r['name']: r['shadow_price'] for r in report['constraints']}
        # Maximization: one more unit of x capacity earns 0.9 net of the shared total.
        self.assertAlmostEqual(prices['x_cap'], 0.9)
        self.assertAlmostEqual(prices['total'], 0.1)

    def test_milp_fixed_integer_analysis(self):
        report = run('sensitivity', ROOT / 'examples/supply_chain.json')
        self.assertTrue(report['fixed_integer_analysis'])
        self.assertEqual(report['status'], 'OPTIMAL_BASIS')

    def test_quadratic_rejected(self):
        self.assertEqual(run('sensitivity', ROOT / 'examples/dispatch.json')['status'], 'UNSUPPORTED')


class IisTests(unittest.TestCase):
    def test_blend_conflict(self):
        report = run('iis', ROOT / 'examples/infeasible_blend.json')
        self.assertEqual(report['status'], 'IIS_FOUND')
        self.assertTrue(report['irreducible'])
        self.assertEqual({r['name'] for r in report['rows']}, {'fuel_demand', 'sulfur_spec'})
        self.assertEqual([(b['variable'], b['side']) for b in report['bounds']],
                         [('light_crude', 'upper')])
        self.assertGreater(report['certificate']['verified_margin'], 0)
        repairs = {r['name']: r['repair'] for r in report['rows']}
        self.assertAlmostEqual(repairs['fuel_demand']['new_limit'], 62.5)

    def test_free_columns_after_bound_removal(self):
        model = {'name': 'conflict', 'sense': 'min',
                 'variables': [{'name': 'x', 'lb': 0, 'ub': 10}, {'name': 'y', 'lb': 0, 'ub': 10}],
                 'objective': {'linear': [1, 1]},
                 'constraints': [
                     {'name': 'c1', 'coefficients': {'x': 1, 'y': 1}, 'lb': 15},
                     {'name': 'c2', 'coefficients': {'x': 1}, 'ub': 3},
                     {'name': 'c3', 'coefficients': {'y': 1}, 'ub': 4}]}
        with tempfile.TemporaryDirectory() as folder:
            report = run('iis', write(folder, 'm.json', model))
        self.assertTrue(report['irreducible'])
        # Removing both bounds of y leaves a free column; the exact-sign margin still certifies.
        self.assertIn('Bound: x <= 10', report['members'])
        members = len(report['rows']) + len(report['bounds'])
        self.assertEqual(members, 3)

    def test_feasible_model(self):
        self.assertEqual(run('iis', ROOT / 'examples/production.json')['status'], 'FEASIBLE')


class GlobalTests(unittest.TestCase):
    def test_haverly_certified_global(self):
        report = run('global', ROOT / 'examples/haverly_pooling.json')
        self.assertEqual(report['badge']['state'], 'CERTIFIED_GLOBAL')
        self.assertAlmostEqual(report['objective'], 400, places=4)
        self.assertLessEqual(report['upper_bound'], 400 * (1 + 1e-4))
        self.assertAlmostEqual(report['root_relaxation_bound'], 500, places=4)
        self.assertTrue(report['verification']['feasible'])

    def test_haverly_variants(self):
        base = json.loads((ROOT / 'examples/haverly_pooling.json').read_text())
        for variable in base['variables']:
            if variable['name'] != 'pool_sulfur':
                variable['ub'] = 1000
        second = json.loads(json.dumps(base))
        next(r for r in second['constraints'] if r['name'] == 'demand_x')['ub'] = 600
        third = json.loads(json.dumps(base))
        third['objective']['linear']['crude_b'] = -13
        with tempfile.TemporaryDirectory() as folder:
            for name, model, expected in [('h2', second, 600), ('h3', third, 750)]:
                report = run('global', write(folder, name + '.json', model))
                self.assertEqual(report['status'], 'GLOBAL_OPTIMAL', name)
                self.assertAlmostEqual(report['objective'], expected, places=4)

    def test_node_limit_reports_bounded_gap(self):
        base = json.loads((ROOT / 'examples/haverly_pooling.json').read_text())
        base['objective']['linear']['crude_b'] = -13
        for variable in base['variables']:
            if variable['name'] != 'pool_sulfur':
                variable['ub'] = 1000
        with tempfile.TemporaryDirectory() as folder:
            report = run('global', write(folder, 'm.json', base), '--node-limit', 1)
        self.assertEqual(report['badge']['state'], 'BOUNDED')
        self.assertGreater(report['badge']['unverified_gap'], 0)
        self.assertGreaterEqual(report['upper_bound'], 750 - 1e-6)

    def test_infeasible_pooling(self):
        base = json.loads((ROOT / 'examples/haverly_pooling.json').read_text())
        base['constraints'].append({'name': 'min_y', 'lb': 250,
                                    'coefficients': {'pool_to_y': 1, 'crude_c_to_y': 1}})
        with tempfile.TemporaryDirectory() as folder:
            report = run('global', write(folder, 'm.json', base))
        self.assertEqual(report['badge']['state'], 'CERTIFIED_INFEASIBLE')


class AssuranceAndHistoryTests(unittest.TestCase):
    def test_badges(self):
        self.assertEqual(run('solve', ROOT / 'examples/production.json')['assurance']['state'],
                         'CERTIFIED_OPTIMAL')
        self.assertEqual(run('solve', ROOT / 'examples/scheduling.json')['assurance']['state'],
                         'BOUNDED')
        self.assertEqual(run('solve', ROOT / 'examples/infeasible_blend.json')['assurance']['state'],
                         'CERTIFIED_INFEASIBLE')

    def test_learned_method(self):
        with tempfile.TemporaryDirectory() as folder:
            history = Path(folder) / 'history.jsonl'
            empty = run('solve', ROOT / 'datasets/afiro.mps', '--method', 'learned',
                        '--history', history)
            self.assertIn('Insufficient', empty['selection']['learned']['reason'])
            for _ in range(2):
                for method in ('simplex', 'pdhg'):
                    run('solve', ROOT / 'datasets/afiro.mps', '--method', method,
                        '--history', history)
            records = [json.loads(line) for line in history.read_text().splitlines()]
            self.assertEqual(len(records), 5)
            ranking = run('recommend', ROOT / 'datasets/afiro.mps', '--history', history)
            self.assertEqual(ranking['source'], 'history')
            self.assertIn(ranking['recommended_method'], ('simplex', 'pdhg'))
            learned = run('solve', ROOT / 'datasets/afiro.mps', '--method', 'learned',
                          '--history', history)
            self.assertEqual(learned['selection']['learned']['method'],
                             ranking['recommended_method'])


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
