"""Independent enumeration and checkpoint evidence for learned bound clauses."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

BINARY = os.environ.get('NIRYUKTI_BINARY', 'build/niryukti')

class ConflictLearning(unittest.TestCase):
    def test_propagation_proved_minimal_clauses(self):
        # An odd cycle of exact binary XOR relations is LP-feasible but integer-infeasible.
        model = {'variables': [{'name': n, 'lb': 0, 'ub': 1, 'type': 'binary'} for n in 'xyzw'],
                 'objective': {'linear': [0, 0, 0, 0]},
                 'constraints': [{'name': f'r{i}', 'coefficients': dict.fromkeys(pair, 1), 'lb': 1, 'ub': 1}
                                 for i, pair in enumerate(('xy', 'yz', 'zx'))]}
        feasible = []
        for mask in range(16):
            point = {name: (mask >> i) & 1 for i, name in enumerate('xyzw')}
            if all(sum(point[n] for n in pair) == 1 for pair in ('xy', 'yz', 'zx')):
                feasible.append(point)
        self.assertEqual(feasible, [])
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'model.json'; checkpoint = Path(temp)/'tree.json'
            source.write_text(json.dumps(model))
            run = subprocess.run([BINARY, 'solve', str(source), '--device', 'cpu', '--method', 'simplex',
                                  '--checkpoint-out', str(checkpoint), '--time-limit', '20'],
                                 capture_output=True, text=True, timeout=30)
            self.assertIn(run.returncode, (0, 2), run.stderr)
            result = json.loads(run.stdout)
            self.assertEqual(result['status'], 'INFEASIBLE')
            clauses = json.loads(checkpoint.read_text())['conflicts']
            self.assertGreaterEqual(len(clauses), 1)
            self.assertTrue(all(len(clause) == 1 for clause in clauses), clauses)
            # Each saved one-literal conjunction is independently impossible.
            for clause in clauses:
                for mask in range(16):
                    if all(((mask >> j) & 1) == value for j, value in clause):
                        self.assertTrue(any(sum((mask >> 'xyzw'.index(n)) & 1 for n in pair) != 1
                                            for pair in ('xy', 'yz', 'zx')))

    def test_general_integer_bound_clauses_and_resume(self):
        names = 'xyz'
        pairs = ('xy', 'yz', 'zx')
        model = {'variables': [{'name': n, 'lb': 2, 'ub': 3, 'type': 'integer'} for n in names],
                 'objective': {'linear': [0, 0, 0]},
                 'constraints': [{'name': f'r{i}', 'coefficients': dict.fromkeys(pair, 1), 'lb': 5, 'ub': 5}
                                 for i, pair in enumerate(pairs)]}
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'integer.json'; checkpoint = Path(temp)/'tree.json'
            source.write_text(json.dumps(model))
            command = [BINARY, 'solve', str(source), '--device', 'cpu', '--method', 'simplex',
                       '--checkpoint-out', str(checkpoint), '--time-limit', '20']
            run = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertIn(run.returncode, (0, 2), run.stderr)
            self.assertEqual(json.loads(run.stdout)['status'], 'INFEASIBLE')
            saved = json.loads(checkpoint.read_text())
            clauses = saved['bound_conflicts']
            self.assertGreater(len(clauses), 0)
            self.assertGreater(saved['statistics']['bound_conflicts_learned'], 0)
            for clause in clauses:
                for mask in range(8):
                    point = [2 + ((mask >> j) & 1) for j in range(3)]
                    forbidden = all(point[l['variable']] >= l['value'] if l['lower'] else
                                    point[l['variable']] <= l['value'] for l in clause)
                    if forbidden:
                        self.assertTrue(any(sum(point[names.index(n)] for n in pair) != 5 for pair in pairs))
            resumed = subprocess.run(command + ['--resume', str(checkpoint)], capture_output=True,
                                     text=True, timeout=30)
            self.assertIn(resumed.returncode, (0, 2), resumed.stderr)
            self.assertEqual(json.loads(resumed.stdout)['status'], 'INFEASIBLE')

if __name__ == '__main__':
    unittest.main()
