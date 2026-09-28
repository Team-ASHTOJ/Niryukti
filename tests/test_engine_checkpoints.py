"""Resume actual phase/basis and Newton state, and reject corrupted checkpoints."""
import json
import os
from pathlib import Path
import subprocess
import signal
import time
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = os.environ.get('NIRYUKTI_BINARY', str(ROOT/'build/niryukti'))

class EngineCheckpointTests(unittest.TestCase):
    def solve(self, model, method, *extra):
        p = subprocess.run([BINARY, 'solve', str(model), '--method', method,
                            '--device', 'cpu', '--no-presolve', '--time-limit', '30',
                            *extra], capture_output=True, text=True, timeout=40)
        self.assertIn(p.returncode, (0, 2), p.stderr)
        return json.loads(p.stdout)

    def test_gpu_presolve_checkpoint_dual_reconstruction(self):
        devices = subprocess.check_output([BINARY, 'devices'], text=True)
        if 'CUDA: NVIDIA' not in devices:
            self.skipTest('CUDA hardware/backend unavailable')
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)/'model.json'; checkpoint = Path(folder)/'state.json'
            model.write_text(json.dumps({
                'variables': [{'name': n, 'lb': 0, 'ub': 10} for n in 'xzw'],
                'objective': {'linear': [1, -.1, 2]},
                'constraints': [
                    {'name': 'blend', 'coefficients': {'x': 1, 'z': 1}, 'lb': 6},
                    {'name': 'limit', 'coefficients': {'z': 1}, 'ub': 3},
                    {'name': 'demand', 'coefficients': {'x': 1, 'w': 1}, 'lb': 8}]}))
            command = [BINARY, 'solve', str(model), '--device', 'cuda', '--method', 'pdhg',
                       '--gpu-presolve', '--gpu-monitor', '--cuda-graphs', '--check-every', '1',
                       '--time-limit', '20', '--checkpoint-out', str(checkpoint), '--checkpoint-nodes', '1']
            first = subprocess.run(command + ['--iterations', '1'], capture_output=True, text=True, timeout=30)
            self.assertIn(first.returncode, (0, 2), first.stderr)
            self.assertTrue(checkpoint.is_file())
            resumed = subprocess.run(command + ['--iterations', '100000', '--resume', str(checkpoint)],
                                     capture_output=True, text=True, timeout=30)
            self.assertIn(resumed.returncode, (0, 2), resumed.stderr)
            result = json.loads(resumed.stdout)
            self.assertEqual(result['status'], 'OPTIMAL', result.get('message'))
            self.assertGreater(result['presolve']['bounds_tightened'], 0)
            self.assertLessEqual(result['accuracy']['kkt_error'], 1e-6)
            self.assertAlmostEqual(result['objective'], 7.7, places=4)

    def test_actual_solver_sigint(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder)/'iterations.log'
            with log.open('w') as stderr:
                worker = subprocess.Popen([BINARY,'solve',str(ROOT/'datasets/e226.mps'),
                    '--method','pdhg','--device','cpu','--no-presolve','--tol','1e-12',
                    '--iterations','100000000','--time-limit','60','--verbose'],
                    stdout=subprocess.PIPE,stderr=stderr,text=True)
                try:
                    deadline = time.monotonic()+10
                    while worker.poll() is None and time.monotonic()<deadline:
                        if 'iter=' in log.read_text(): break
                        time.sleep(.01)
                    self.assertIsNone(worker.poll(), 'solver finished before cancellation test')
                    self.assertIn('iter=',log.read_text(), 'solver did not enter iterations')
                    worker.send_signal(signal.SIGINT)
                    output,_ = worker.communicate(timeout=10)
                    result = json.loads(output)
                    self.assertEqual(result['status'],'INTERRUPTED')
                    self.assertGreater(result['performance']['iterations'],0)
                finally:
                    if worker.poll() is None:
                        worker.kill(); worker.communicate()

    def test_zero_budget_stops_preprocessing(self):
        for method in ('pdhg', 'simplex', 'barrier'):
            result = self.solve(ROOT/'examples/coupled_dispatch.json' if method == 'barrier' else ROOT/'datasets/adlittle.mps',
                                method, '--time-limit', '0')
            self.assertEqual(result['status'], 'TIME_LIMIT')
            self.assertEqual(result['performance']['iterations'], 0)

    def test_portfolio_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/'portfolio.json'
            model = ROOT/'datasets/adlittle.mps'
            self.solve(model, 'concurrent', '--threads', '3', '--iterations', '2',
                       '--checkpoint-out', str(checkpoint), '--checkpoint-nodes', '1')
            saved = json.loads(checkpoint.read_text())
            self.assertEqual(saved['schema'], 'niryukti-portfolio-1')
            self.assertEqual(len(saved['children']), 3)
            self.assertTrue(all((checkpoint.parent/p).is_file() for p in saved['children']))
            resumed = self.solve(model, 'auto', '--threads', '3', '--iterations', '10000', '--resume', str(checkpoint))
            self.assertEqual(resumed['status'], 'OPTIMAL', resumed.get('message'))
            self.assertLessEqual(resumed['accuracy']['kkt_error'], 1e-6)

    def test_barrier_and_simplex_resume(self):
        for method, model in [('barrier', ROOT/'examples/coupled_dispatch.json'),
                              ('simplex', ROOT/'datasets/adlittle.mps')]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'state.json'
                limited = self.solve(model, method, '--iterations', '2', '--checkpoint-out', str(path),
                                     '--checkpoint-nodes', '1')
                state = json.loads(path.read_text())
                self.assertEqual(state['iterations'], limited['performance']['iterations'])
                self.assertEqual(state['iterations'], 2)
                resumed = self.solve(model, 'auto', '--iterations', '10000', '--resume', str(path))
                cold = self.solve(model, method, '--iterations', '10000')
                self.assertEqual(resumed['status'], 'OPTIMAL', resumed.get('message'))
                self.assertAlmostEqual(resumed['objective'], cold['objective'], delta=1e-4)
                self.assertLessEqual(resumed['accuracy']['kkt_error'], 1e-6)
                # Model binding and internal state validation are independent.
                state['fingerprint'] = 'corrupted'
                path.write_text(json.dumps(state))
                bad = subprocess.run([BINARY,'solve',str(model),'--method',method,'--device','cpu',
                                      '--no-presolve','--resume',str(path)],capture_output=True,text=True)
                self.assertNotEqual(bad.returncode, 0)
                self.assertIn('mismatch', bad.stderr + bad.stdout)
                state['fingerprint'] = model_fingerprint(model)
                if method == 'barrier': state['slack'][0] = -1
                else: state['basis'][0] = -1
                path.write_text(json.dumps(state))
                bad = subprocess.run([BINARY,'solve',str(model),'--method',method,'--device','cpu',
                                      '--no-presolve','--resume',str(path)],capture_output=True,text=True)
                self.assertNotEqual(bad.returncode, 0)

def model_fingerprint(model):
    return json.loads(subprocess.check_output([BINARY,'inspect',str(model)],text=True))['fingerprint']

if __name__ == '__main__': unittest.main()
