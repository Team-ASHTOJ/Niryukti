"""Exercise evidence generation using actual solver and verifier processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class BenchmarkEvidence(unittest.TestCase):
    def test_independent_verification_and_offline_artifacts(self):
        binary=Path(os.environ.get('VANTAGE_TEST_BINARY',ROOT/'build/vantage')).resolve()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'campaign'
            subprocess.run([sys.executable,str(ROOT/'benchmark/run.py'),str(ROOT/'examples/toy.lp'),
                            str(ROOT/'examples/integer_dispatch.json'),'--binary',str(binary),
                            '--solvers','cpu','--method','auto','--runs','1','--output',str(out)],check=True,capture_output=True,text=True)
            continuous=json.loads((out/'raw/000_toy_cpu_0.json').read_text())
            integer=json.loads((out/'raw/001_integer_dispatch_cpu_0.json').read_text())
            self.assertEqual(continuous['record']['verification_status'],'VERIFIED_OPTIMAL')
            self.assertEqual(integer['record']['verification_status'],'VERIFIED_FEASIBLE')
            self.assertEqual(continuous['result']['verification_process']['exit_code'],0)
            self.assertTrue((out/'raw/000_toy.input.lp').is_file())
            self.assertTrue((out/'raw/000_toy_cpu_0.verify.stdout').is_file())
            self.assertTrue((out/'source/benchmark/run.py').is_file())
            page=(out/'index.html').read_text()
            self.assertIn('VERIFIED_OPTIMAL',page)
            self.assertIn('Revision:',page)
            self.assertIn('not the search tree',page)

if __name__=='__main__':unittest.main()
