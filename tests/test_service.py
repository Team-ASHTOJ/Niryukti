import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.request
import urllib.error
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from niryukti.service import make_server
from niryukti.report import render_report,write_report

class ServiceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.server=make_server(port=0,token='local-test-token')
  cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()
  cls.url='http://127.0.0.1:'+str(cls.server.server_port)
 @classmethod
 def tearDownClass(cls):
  cls.server.shutdown();cls.server.server_close();cls.thread.join()
 def request(self,path,data=None,auth=True):
  headers={'Authorization':'Bearer local-test-token'} if auth else {}
  if data is not None:headers['Content-Type']='application/json'
  req=urllib.request.Request(self.url+path,data=json.dumps(data).encode() if data is not None else None,headers=headers)
  return urllib.request.urlopen(req,timeout=30)
 def test_authentication_and_health(self):
  with self.assertRaises(urllib.error.HTTPError) as error:self.request('/v1/health',auth=False)
  self.assertEqual(error.exception.code,401)
  self.assertEqual(json.load(self.request('/v1/health'))['service'],'NIRYUKTI')
 def test_actual_solve_and_report(self):
  model={'name':'api-lp','variables':[{'name':'x','lb':0,'ub':10}], 'objective':{'linear':[3]},'constraints':[{'name':'demand','lb':2,'coefficients':{'x':1}}]}
  result=json.load(self.request('/v1/solve',{'model':model,'options':{'method':'auto','device':'cpu'}}))
  self.assertEqual(result['status'],'OPTIMAL');self.assertAlmostEqual(result['objective'],6)
  report=self.request('/v1/report',{'result':result,'title':'<script>bad</script>'}).read().decode()
  self.assertNotIn('<script>bad</script>',report);self.assertIn('&lt;script&gt;',report)
  self.assertIn('not a new independent verification',report)
 def test_disallows_binary_override_and_excessive_budget(self):
  for options in ({'binary':'/bin/sh'},{'time_limit':99999}):
   with self.assertRaises(urllib.error.HTTPError) as error:self.request('/v1/solve',{'model':{},'options':options})
   self.assertEqual(error.exception.code,400)
 def test_malformed_headers_and_numeric_overflow(self):
  req=urllib.request.Request(self.url+'/v1/health',headers={'Authorization':'Bearer é'})
  with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(req)
  self.assertEqual(error.exception.code,401)
  req=urllib.request.Request(self.url+'/v1/solve',data=b'{"model":{},"options":{"threads":1e999}}',
                            headers={'Authorization':'Bearer local-test-token','Content-Type':'application/json'})
  with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(req)
  self.assertEqual(error.exception.code,400)
  self.assertEqual(json.load(self.request('/v1/health'))['service'],'NIRYUKTI')
  self.assertIn('TIME_LIMIT',render_report({'status':'TIME_LIMIT','accuracy':None}))
 def test_report_file_and_invalid_input(self):
  with self.assertRaises(ValueError):render_report({})
  with tempfile.TemporaryDirectory() as d:
   source=Path(d)/'result.json';source.write_text(json.dumps({'status':'TIME_LIMIT','accuracy':{'kkt':.1}}))
   target=Path(d)/'report.html';write_report(source,target);self.assertIn('TIME_LIMIT',target.read_text())
if __name__=='__main__':unittest.main()
