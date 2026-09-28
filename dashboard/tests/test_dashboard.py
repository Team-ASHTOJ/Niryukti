import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import sys
import unittest
from unittest.mock import patch
import urllib.request
import urllib.error

spec=importlib.util.spec_from_file_location('dashboard_server',Path(__file__).resolve().parents[1]/'server.py')
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)

class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.original_root,cls.original_store=server.ROOT,server.STORE
        server.ROOT=Path(cls.temp.name)
        server.STORE=server.ROOT/'results/dashboard';server.STORE.mkdir(parents=True)
        server.BINARY=Path(os.environ.get('VANTAGE_TEST_BINARY',cls.original_root/'build/vantage')).resolve()
        (server.ROOT/'examples').symlink_to(cls.original_root/'examples',target_is_directory=True)
        # The API contract must not depend on ignored local benchmark campaigns.
        folder=server.ROOT/'results/demo';(folder/'raw').mkdir(parents=True)
        (folder/'manifest.json').write_text('{}')
        for backend in ('cpu','cuda','highs'):
            for run in range(3):
                record=dict(instance='afiro.mps',solver=backend,run=run,status='ITERATION_LIMIT' if backend=='cuda' else 'OPTIMAL')
                (folder/'raw'/f'{backend}_{run}.json').write_text(json.dumps(dict(record=record)))
        cls.http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        cls.url=f'http://127.0.0.1:{cls.http.server_port}'
        threading.Thread(target=cls.http.serve_forever,daemon=True).start()
    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown();cls.http.server_close()
        server.ROOT,server.STORE=cls.original_root,cls.original_store
        cls.temp.cleanup()
    def setUp(self):
        server.JOBS.clear();server.UPLOADED.clear()
    def post(self,path,body):
        request=urllib.request.Request(self.url+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Vantage-Token':server.TOKEN})
        with urllib.request.urlopen(request) as response:return json.load(response)
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
    def test_csv_evidence_merges_baselines_and_keeps_single_run_scope(self):
        folder=server.ROOT/'results/stress_highs_ipm_20260927';folder.mkdir(parents=True)
        (folder/'manifest.json').write_text(json.dumps(dict(arguments=dict(runs=1,threads=4,time_limit=60))))
        (folder/'runs.csv').write_text('instance,solver,run,status,reported_status,verification_status,objective,end_to_end_seconds,problem_type,rows,columns,nonzeros\nafiro.mps,highs-ipm,0,OPTIMAL,OPTIMAL,VERIFIED_OPTIMAL,1,2,LP,27,32,83\n')
        try:
            data=self.get('/api/benchmarks');case=next(c for c in data['cases'] if c['id']=='afiro')
            self.assertEqual(set(case['engines']),{'cpu','cuda','highs','highs-ipm'})
            self.assertEqual(case['engines']['cpu']['runs'],3)
            self.assertEqual(case['engines']['highs-ipm']['runs'],1)
            self.assertTrue(case['engines']['highs-ipm']['exploratory'])
            self.assertEqual(case['engines']['highs-ipm']['objective_error'],0)
            self.assertIsNone(case['engines']['cpu']['objective_error'])
            self.assertEqual(data['available']['highs-ipm'],1)
        finally:
            for f in folder.iterdir():f.unlink()
            folder.rmdir()

    def test_large_result_preview_preserves_full_download_data(self):
        values=list(range(1000));job=dict(id='large',result=dict(primal=values,status='OPTIMAL'))
        preview=server.public_job(job)
        self.assertEqual(len(preview['result']['primal']),256)
        self.assertEqual(preview['result']['preview']['total_lengths']['primal'],1000)
        self.assertEqual(len(job['result']['primal']),1000)
        self.assertEqual(server.public_job(job,full=True)['result']['primal'],values)

    def test_certificate_downloads(self):
        import subprocess
        result=subprocess.run([str(server.BINARY),'solve',str(self.original_root/'examples/toy.lp'),'--device','cpu'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        payload=json.loads(result.stdout)
        server.JOBS['certificate-test']=dict(id='certificate-test',result=payload)
        certificate=self.get('/api/runs/certificate-test/certificate')
        self.assertTrue(certificate['verification']['valid'])
        self.assertEqual(self.get('/api/runs/certificate-test/verification_report'),certificate['verification'])

    def test_documentation_download_is_allowlisted(self):
        folder=server.ROOT/'docs';folder.mkdir(exist_ok=True)
        file=folder/'solver_completion_20260927.md';file.write_text('Current implementation scope')
        try:
            with urllib.request.urlopen(self.url+'/api/docs/solver_completion_20260927.md') as response:
                self.assertIn('attachment',response.headers['Content-Disposition'])
                self.assertEqual(response.read(),b'Current implementation scope')
            with self.assertRaises(urllib.error.HTTPError):
                urllib.request.urlopen(self.url+'/api/docs/%2e%2e/server.py')
        finally:file.unlink();folder.rmdir()

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

    def test_import_and_invalid_model(self):
        model=(self.original_root/'examples/refinery.json').read_text()
        result=self.post('/api/upload',dict(name='refinery.json',content=model))
        self.assertEqual(result['type'],'LP')
        self.assertTrue(server.UPLOADED[result['id']].is_file())
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post('/api/upload',dict(name='broken.json',content='{'))
        self.assertEqual(caught.exception.code,400)

    def test_arena_report_and_traversal(self):
        server.JOBS['fixture']=dict(id='fixture',kind='arena',state='finished',result=dict(status='COMPARISON_PARTIAL'))
        folder=server.STORE/'arena-fixture';folder.mkdir(exist_ok=True)
        (folder/'index.html').write_text('retained failures')
        (server.STORE/'secret.json').write_text('{}')
        with urllib.request.urlopen(self.url+'/arena/fixture/index.html') as response:
            self.assertEqual(response.read(),b'retained failures')
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(self.url+'/arena/fixture/%2e%2e/secret.json')
        with urllib.request.urlopen(self.url+'/api/runs/fixture/download') as response:
            self.assertIn('attachment',response.headers['Content-Disposition'])
            self.assertEqual(json.load(response)['status'],'COMPARISON_PARTIAL')

    def test_queued_arena_cancellation_and_concurrency(self):
        # Hold the worker at the queue boundary to deterministically test the race.
        with patch.object(server,'run_arena'):
            job=self.post('/api/runs',dict(model='dispatch',kind='arena',time_limit=1))
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post('/api/runs',dict(model='dispatch',time_limit=1))
        self.assertEqual(caught.exception.code,409)
        self.post('/api/runs/'+job['id']+'/cancel',{})
        self.assertTrue(server.JOBS[job['id']]['cancel_requested'])

    def test_non_object_request_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:self.post('/api/runs',[])
        self.assertEqual(caught.exception.code,400)

    def test_arena_worker_retains_all_engines(self):
        job=dict(id='worker',kind='arena',path=self.original_root/'examples/toy.lp',state='queued')
        options=dict(time_limit=1,threads=1,tol=1e-6,iterations=1000,method='auto',branching='fractional',node_selection='best-bound')
        with patch.object(server,'ROOT',self.original_root):server.run_arena(job,options)
        self.assertEqual(job['state'],'finished')
        self.assertEqual(job['result']['status'],'COMPARISON_COMPLETE')
        engines={item['solver']:item for item in job['result']['comparisons']}
        self.assertEqual(set(engines),{'cpu','cuda','highs'})
        self.assertEqual(engines['cpu']['optimal_runs'],3)
        self.assertTrue(all(item['runs']==3 for item in engines.values()))
        self.assertTrue((server.STORE/'arena-worker/index.html').is_file())

    def test_running_process_cancellation(self):
        # A real child process with deterministic readiness and SIGINT behavior.
        executable=server.ROOT/'interruptible-worker'
        executable.write_text('#!'+sys.executable+'\nimport signal,time,json,sys\n'
            'def stop(*args):\n print(json.dumps({"status":"INTERRUPTED"}),flush=True);sys.exit(0)\n'
            'signal.signal(signal.SIGINT,stop)\nprint("READY",file=sys.stderr,flush=True)\n'
            'while True: time.sleep(.1)\n')
        executable.chmod(0o700)
        job=dict(id='running-cancel',path=self.original_root/'examples/toy.lp',state='queued')
        server.JOBS[job['id']]=job
        options=dict(device='cpu',time_limit=10,threads=1,tol=1e-6,iterations=10000)
        with patch.object(server,'BINARY',executable):
            worker=threading.Thread(target=server.run_job,args=(job,options));worker.start()
            deadline=time.monotonic()+5
            logfile=server.STORE/'running-cancel.log'
            while time.monotonic()<deadline:
                if job.get('process') and logfile.exists() and 'READY' in logfile.read_text():break
                time.sleep(.01)
            self.assertTrue(job.get('process'),'worker never started')
            proc=job['process']
            self.post('/api/runs/running-cancel/cancel',{})
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive(),'worker did not terminate')
            self.assertIsNotNone(proc.poll(),'child process still alive')
        self.assertEqual(job['result']['status'],'INTERRUPTED')
        self.assertNotIn('process',job)

    def test_solve_worker_current_options(self):
        job=dict(id='solve-worker',path=self.original_root/'examples/dispatch.json',state='queued')
        options=dict(device='cpu',time_limit=2,threads=1,tol=1e-6,iterations=100000,method='pdhg',branching='fractional',node_selection='best-bound')
        server.run_job(job,options)
        self.assertEqual(job['state'],'finished')
        self.assertEqual(job['result']['status'],'OPTIMAL')
        self.assertEqual(job['result']['hardware']['backend'],'cpu')
        self.assertNotIn('process',job)

if __name__=='__main__':unittest.main()
