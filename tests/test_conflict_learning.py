"""Independent enumeration and checkpoint evidence for learned binary clauses."""
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

if __name__ == '__main__':
    unittest.main()
