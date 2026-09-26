import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('profiles',Path(__file__).resolve().parents[1]/'benchmark/profiles.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class Profiles(unittest.TestCase):
    def test_failures_ties_missing_and_repetitions(self):
        def row(instance,solver,time,status='OPTIMAL',**kwargs):
            return dict(instance=instance,solver=solver,end_to_end_seconds=time,status=status,**kwargs)
        rows=[row('a','one',1),row('a','two',2),row('b','one',2),row('b','two',2),
              row('c','one',.1,'TIME_LIMIT'),row('c','two',3),row('d','one',4),
              row('d','one',4,'ITERATION_LIMIT'),row('e','two',0),
              row('warm','one',1,warmup=True)]
        data=p.profiles(rows)
        self.assertEqual(len(data['instances']),5)
        self.assertEqual(data['ratios']['two']['a'],2)
        self.assertIsNone(data['ratios']['one']['d'])
        self.assertIsNone(data['ratios']['two']['e'])
        self.assertEqual(data['curves']['one'],[[1.,.4],[2.,.4]])
        self.assertEqual(data['curves']['two'],[[1.,.4],[2.,.6]])
        with tempfile.TemporaryDirectory() as folder:
            p.write_report(data,Path(folder))
            self.assertTrue((Path(folder)/'profile.svg').exists())
    def test_nonfinite_and_empty(self):
        self.assertEqual(p.profiles([])['curves'],{})
        for value in ('nan','inf',-1,None):
            data=p.profiles([dict(instance='a',solver='s',status='OPTIMAL',end_to_end_seconds=value)])
            self.assertIsNone(data['ratios']['s']['a'])

if __name__=='__main__':unittest.main()
