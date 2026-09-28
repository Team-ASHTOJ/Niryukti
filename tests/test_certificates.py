"""Independent certificate replay and corruption regression tests (offline)."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from vantage import Model, solve, verify_certificate
BINARY=os.environ.get('VANTAGE_BINARY',str(ROOT/'build/vantage'))

class Certificates(unittest.TestCase):
    def setUp(self):
        self.m=Model('certificate test');self.m.add_var('x',ub=10)
        self.m.add_constraint({'x':1},'>=',2,'demand');self.m.set_objective({'x':1})
    def result(self,model=None,**kwargs):return (model or self.m).solve(binary=BINARY,method='auto',device='cpu',**kwargs)
    def check(self,c,model=None):return verify_certificate(model or self.m,c,binary=BINARY)
    def test_lp_and_tamper(self):
        c=self.result()['verification_certificate'];v=self.check(c)
        self.assertTrue(v['valid']);self.assertEqual(v['status'],'OPTIMAL')
        for key,value in [('objective',100),('model_fingerprint','0'*64),('primal',{'x':3}),('dual',{'demand':100}),('verification',{'valid':True,'status':'FEASIBLE'}),('termination_status','INFEASIBLE'),('primal',{'x':None}),('primal',{'x':'NaN'}),('primal',{'x':'Infinity'}),('primal',{'x':float('nan')}),('primal',{'x':float('inf')})]:
            bad=copy.deepcopy(c);bad[key]=value;self.assertFalse(self.check(bad)['valid'],key)
        for key in c:
            bad=copy.deepcopy(c);del bad[key];self.assertFalse(self.check(bad)['valid'],key)
        for field in ['lb','ub']:
            wrong=copy.deepcopy(self.m.data);wrong['variables'][0][field]=1 if field=='lb' else 9
            self.assertFalse(self.check(c,wrong)['valid'])
        wrong=copy.deepcopy(self.m.data);wrong['constraints'][0]['lb']=3
        self.assertFalse(self.check(c,wrong)['valid'])
    def test_qp(self):
        self.m.set_objective({'x':-4},quadratic_diagonal={'x':2})
        c=self.result()['verification_certificate'];self.assertTrue(self.check(c)['valid']);self.assertEqual(c['verification']['status'],'OPTIMAL')
    def test_milp(self):
        self.m.data['variables'][0]['type']='integer'
        c=self.result()['verification_certificate'];self.assertTrue(self.check(c)['valid'])
        self.assertEqual(c['verification']['status'],'OPTIMAL_WITHIN_VERIFIED_GAP_TOLERANCE')
        c['primal']['x']=2.5;self.assertFalse(self.check(c)['valid'])
    def test_general_qp(self):
        r=solve(ROOT/'examples/coupled_dispatch.json',binary=BINARY,device='cpu',time_limit=10)
        self.assertTrue(verify_certificate(ROOT/'examples/coupled_dispatch.json',r['verification_certificate'],binary=BINARY)['valid'])
    def test_order(self):
        self.m.add_var('z',ub=1);self.m.set_objective({'x':1,'z':2})
        c=self.result()['verification_certificate'];m=copy.deepcopy(self.m.data)
        m['variables'].reverse();m['objective']['linear'].reverse()
        self.assertTrue(self.check(c,m)['valid'])
    def test_malformed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text('{broken')
            self.assertFalse(self.check(p)['valid'])
    def test_native(self):
        from vantage import NativeSession
        with NativeSession(self.m) as session:
            r=session.solve(device='cpu')
            self.assertTrue(session.verify_certificate(r['verification_certificate'])['valid'])

    def test_cuda(self):
        p=subprocess.run([BINARY,'devices'],capture_output=True,text=True)
        if 'not compiled' in p.stdout.lower() or 'unavailable' in p.stdout.lower():self.skipTest('CUDA unavailable')
        r=self.m.solve(binary=BINARY,device='cuda')
        if r.get('status')=='UNSUPPORTED':self.skipTest('CUDA unavailable')
        self.assertTrue(self.check(r['verification_certificate'])['valid'])

if __name__=='__main__':unittest.main()
