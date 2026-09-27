import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from vantage import Model,NativeSession,rhs_sensitivity
from vantage.evidence import create_bundle,verify_bundle

def linear():
    m=Model();m.add_var('x',ub=10);m.add_constraint({'x':1},'>=',2,name='demand');m.set_objective({'x':3});return m

class Extensions(unittest.TestCase):
    def test_native_in_process_updates_rollback_and_close(self):
        with patch('subprocess.Popen',side_effect=AssertionError('No subprocess allowed')):
            with NativeSession(linear()) as session:
                self.assertAlmostEqual(session.solve()['objective'],6)
                session.update_row_bounds({'demand':{'lb':4}})
                self.assertAlmostEqual(session.solve()['objective'],12)
                with self.assertRaises(ValueError):session.update_variable_bounds({'x':{'lb':11}})
                self.assertAlmostEqual(session.solve()['objective'],12)
                with self.assertRaises(ValueError):session._request('solve',unknown_option=True)
                self.assertAlmostEqual(session.solve()['objective'],12)
            with self.assertRaises(RuntimeError):session.solve()
            session.close()

    def test_native_invalid_inputs(self):
        with self.assertRaises(ValueError):NativeSession('/missing/vantage/model.json')
        with NativeSession(linear()) as session:
            self.assertIsNone(session._lib.vantage_request(None,b'{}'))
            self.assertIsNone(session._lib.vantage_request(session._handle,b'not json'))
            self.assertAlmostEqual(session.solve()['objective'],6)

    def test_measured_sensitivity_analytical_and_kink(self):
        result=rhs_sensitivity(linear(),'demand',.1,session_type=NativeSession)
        self.assertEqual(result['status'],'MEASURED')
        self.assertAlmostEqual(result['slopes']['left'],3)
        self.assertAlmostEqual(result['slopes']['right'],3)
        m=linear();m.data['constraints'][0]['lb']=0
        result=rhs_sensitivity(m,'demand',.1,session_type=NativeSession)
        self.assertAlmostEqual(result['slopes']['left'],0)
        self.assertAlmostEqual(result['slopes']['right'],3)
        m.data['variables'][0]['type']='integer'
        with self.assertRaises(ValueError):rhs_sensitivity(m,'demand',.1)

    def test_bundle_integrity_and_fresh_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model.json';linear().write(model)
            archive=root/'solve.zip';created=create_bundle(model,archive,method='auto',device='cpu')
            result=verify_bundle(archive,expected_sha256=created['sha256'])
            self.assertTrue(result['numerically_verified'])
            with self.assertRaises(ValueError):verify_bundle(archive,expected_sha256='0'*64)
            with zipfile.ZipFile(archive) as z:files={n:z.read(n) for n in z.namelist()}
            solution=json.loads(files['solution.json']);solution['objective']=12345
            files['solution.json']=json.dumps(solution).encode()
            corrupt=root/'corrupt.zip'
            with zipfile.ZipFile(corrupt,'w') as z:
                for n,b in files.items():z.writestr(n,b)
            with self.assertRaises(ValueError):verify_bundle(corrupt)
            # Even rehashing a forged solution cannot bypass numerical verification.
            manifest=json.loads(files['manifest.json']);manifest['sha256']['solution.json']=hashlib.sha256(files['solution.json']).hexdigest()
            files['manifest.json']=json.dumps(manifest).encode()
            forged=root/'forged.zip'
            with zipfile.ZipFile(forged,'w') as z:
                for n,b in files.items():z.writestr(n,b)
            self.assertFalse(verify_bundle(forged)['numerically_verified'])
            with self.assertRaises(FileExistsError):create_bundle(model,archive)

    def test_bundle_rejects_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'unsafe.zip'
            with zipfile.ZipFile(path,'w') as z:z.writestr('../outside','payload')
            with self.assertRaises(ValueError):verify_bundle(path)

if __name__=='__main__':unittest.main()
