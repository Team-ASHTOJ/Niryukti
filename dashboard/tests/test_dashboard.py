import importlib.util
import json
from pathlib import Path
import threading
import unittest
import urllib.request
import urllib.error

spec=importlib.util.spec_from_file_location('dashboard_server',Path(__file__).resolve().parents[1]/'server.py')
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)

class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        cls.url=f'http://127.0.0.1:{cls.http.server_port}'
        threading.Thread(target=cls.http.serve_forever,daemon=True).start()
    @classmethod
    def tearDownClass(cls): cls.http.shutdown();cls.http.server_close()
    def get(self,path):
        with urllib.request.urlopen(self.url+path) as r: return json.load(r)
    def test_data_preserves_limits_and_deduplicates(self):
        data=self.get('/api/benchmarks');ids=[c['id'] for c in data['cases']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(ids.count('afiro'),1)
        for c in data['cases']:
            for engine in c['engines'].values():
                self.assertEqual(engine['runs'],3)
                if engine['status']!='OPTIMAL':self.assertLess(engine['solved'],3)
        self.assertTrue(any(c['engines']['cuda']['status']!='OPTIMAL' for c in data['cases']))
    def test_mutation_requires_session_token(self):
        request=urllib.request.Request(self.url+'/api/runs',data=b'{}',headers={'Content-Type':'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as caught:urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code,403)
    def test_untrusted_host_rejected(self):
        request=urllib.request.Request(self.url+'/api/bootstrap',headers={'Host':'example.invalid'})
        with self.assertRaises(urllib.error.HTTPError) as caught:urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code,403)
    def test_bad_options_rejected(self):
        token=self.get('/api/bootstrap')['token']
        payload=json.dumps(dict(model='dispatch',time_limit=999999)).encode()
        request=urllib.request.Request(self.url+'/api/runs',data=payload,headers={'Content-Type':'application/json','X-Vantage-Token':token})
        with self.assertRaises(urllib.error.HTTPError) as caught:urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code,400)
    def test_excessive_iteration_budget_rejected(self):
        token=self.get('/api/bootstrap')['token']
        payload=json.dumps(dict(model='dispatch',iterations=2000001)).encode()
        request=urllib.request.Request(self.url+'/api/runs',data=payload,headers={'Content-Type':'application/json','X-Vantage-Token':token})
        with self.assertRaises(urllib.error.HTTPError) as caught:urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code,400)
    def test_no_filesystem_escape(self):
        with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(self.url+'/%2e%2e/server.py')
        with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(self.url+'/reports/demo/%2e%2e/dashboard/x.json')
    def test_csv_export(self):
        with urllib.request.urlopen(self.url+'/api/export.csv') as r:
            self.assertIn('attachment',r.headers['Content-Disposition'])
            text=r.read().decode();self.assertIn('ITERATION_LIMIT',text);self.assertIn('mip_gap',text)

if __name__=='__main__':unittest.main()
