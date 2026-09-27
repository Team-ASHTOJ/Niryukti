import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from vantage import Model,replan,stable_plan_model
from vantage.evidence import create_bundle,sign_bundle,verify_signed_bundle

def model(sense='min'):
    m=Model();m.add_var('x',ub=10);m.add_constraint({'x':1},'>=',5,name='demand');m.set_objective({'x':1},sense=sense);return m

class Stability(unittest.TestCase):
    def test_cost_and_disruption_tradeoff(self):
        m=model()
        for weight,expected in [(.5,5),(2,8)]:
            r=replan(m,{'x':8},{'x':weight},device='cpu',method='auto')
            self.assertEqual(r['status'],'VERIFIED_PLAN')
            self.assertAlmostEqual(r['plan']['x'],expected)
            self.assertAlmostEqual(r['operating_objective'],expected)
            self.assertAlmostEqual(r['disruption_penalty'],weight*abs(expected-8))
        self.assertEqual(len(m.data['variables']),1)

    def test_locks_and_maximization(self):
        m=model('max')
        r=replan(m,{'x':6},{'x':2},method='auto',device='cpu')
        self.assertAlmostEqual(r['plan']['x'],6)
        r=replan(m,{'x':7},{},locked=['x'],method='auto',device='cpu')
        self.assertAlmostEqual(r['plan']['x'],7)
        with self.assertRaises(ValueError):stable_plan_model(m,{'x':11},{},locked=['x'])
        r=replan(m,{'x':2},{},locked=['x'],method='auto',device='cpu')
        self.assertEqual(r['status'],'NO_VERIFIED_PLAN')

@unittest.skipUnless(shutil.which('openssl'),'OpenSSL required for optional signature tests')
class Signatures(unittest.TestCase):
    def test_signature_trust_tampering_and_mathematical_check(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for name in ('team','other'):
                subprocess.run(['openssl','genrsa','-out',str(folder/f'{name}.key'), '2048'],check=True,capture_output=True)
                subprocess.run(['openssl','pkey','-in',str(folder/f'{name}.key'),'-pubout','-out',str(folder/f'{name}.pub')],check=True,capture_output=True)
            path=folder/'model.json';model().write(path)
            bundle=folder/'model.zip';signature=folder/'model.sig'
            create_bundle(path,bundle,method='auto',device='cpu')
            sign_bundle(bundle,folder/'team.key',signature)
            result=verify_signed_bundle(bundle,signature,folder/'team.pub')
            self.assertTrue(result['signature_verified']);self.assertTrue(result['numerically_verified'])
            with self.assertRaises(ValueError):verify_signed_bundle(bundle,signature,folder/'other.pub')
            with zipfile.ZipFile(bundle) as z:files={name:z.read(name) for name in z.namelist()}
            data=json.loads(files['solution.json']);data['objective']=999
            files['solution.json']=json.dumps(data).encode()
            manifest=json.loads(files['manifest.json']);manifest['sha256']['solution.json']=hashlib.sha256(files['solution.json']).hexdigest()
            files['manifest.json']=json.dumps(manifest).encode()
            tampered=folder/'tampered.zip'
            with zipfile.ZipFile(tampered,'w') as z:
                for name,data in files.items():z.writestr(name,data)
            with self.assertRaises(ValueError):verify_signed_bundle(tampered,signature,folder/'team.pub')
            signed_bad=folder/'bad.sig';sign_bundle(tampered,folder/'team.key',signed_bad)
            result=verify_signed_bundle(tampered,signed_bad,folder/'team.pub')
            self.assertTrue(result['signature_verified']);self.assertFalse(result['numerically_verified'])

if __name__=='__main__':unittest.main()
