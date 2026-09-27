import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from vantage import Model,SolverSession,propose_repair,diagnose_infeasibility

class Planning(unittest.TestCase):
    def test_coupled_refinery_analytical_cost_and_disruptions(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
        from refinery_replanning import refinery_model
        m=refinery_model()
        with SolverSession(m) as session:
            baseline=session.solve()
            self.assertEqual(baseline['status'],'OPTIMAL')
            # 420 tonnes demand: sulfur requires 280 clean / 140 sour.
            # Initial stock contributes 40 clean / 20 sour at sunk cost.
            self.assertAlmostEqual(baseline['objective'],240*70+120*45,places=5)
            session.update_row_bounds({'capacity_3':{'ub':0}})
            revised=session.solve()
            self.assertEqual(revised['status'],'OPTIMAL')
            x=dict(zip([v['name'] for v in m.data['variables']],revised['primal']))
            self.assertAlmostEqual(x['feed_clean_3']+x['feed_sour_3'],0,places=6)
            self.assertGreaterEqual(x['product_2'],60-1e-6)
            for t in range(7):
                self.assertLessEqual(.005*x[f'feed_clean_{t}']+.02*x[f'feed_sour_{t}'],.01*(x[f'feed_clean_{t}']+x[f'feed_sour_{t}'])+1e-6)
                previous=x[f'product_{t-1}'] if t else 0
                self.assertAlmostEqual(previous+x[f'feed_clean_{t}']+x[f'feed_sour_{t}']-x[f'product_{t}'],60,places=5)

    def test_session_updates_and_transaction_rollback(self):
        m=Model();m.add_var('x',ub=10);m.add_constraint({'x':1},'>=',2,name='demand');m.set_objective({'x':3})
        with SolverSession(m) as session:
            pid=session._process.pid
            self.assertAlmostEqual(session.solve()['objective'],6)
            session.update_row_bounds({'demand':{'lb':4}})
            self.assertAlmostEqual(session.solve()['objective'],12)
            session.update_objective({'x':2})
            self.assertAlmostEqual(session.solve()['objective'],8)
            with self.assertRaises(ValueError):session.update_variable_bounds({'x':{'lb':20}})
            self.assertAlmostEqual(session.solve(warm_start=False)['objective'],8)
            self.assertEqual(pid,session._process.pid)
        self.assertIsNotNone(session._process.poll())

    def test_repair_preserves_hard_bounds_and_is_minimal(self):
        m=Model();m.add_var('x',ub=3);m.add_constraint({'x':1},'>=',5,name='demand');m.set_objective({'x':-100})
        repair=propose_repair(m,{'demand':1},device='cpu',method='auto')
        self.assertEqual(repair['status'],'REPAIR_PROPOSAL')
        self.assertAlmostEqual(repair['result']['objective'],2)
        self.assertAlmostEqual(repair['original_primal'][0],3)
        self.assertEqual(repair['changes'][0]['row'],'demand')
        self.assertEqual(m.data['constraints'][0]['lb'],5)
        with self.assertRaises(ValueError):propose_repair(m,{'demand':-1})
        capped=propose_repair(m,{'demand':1},violation_limits={'demand':{'lb':1}},device='cpu',method='auto')
        self.assertEqual(capped['status'],'NO_VERIFIED_REPAIR')

    def test_hard_conflict_is_not_reported_as_repair(self):
        m=Model();m.add_var('x',ub=3);m.add_constraint({'x':1},'>=',5,name='hard');m.add_constraint({'x':1},'>=',6,name='soft')
        repair=propose_repair(m,{'soft':1},device='cpu',method='auto')
        self.assertEqual(repair['status'],'NO_VERIFIED_REPAIR')
        diagnosis=diagnose_infeasibility(m,device='cpu',method='auto')
        if diagnosis['conflict_verified']:
            self.assertEqual(diagnosis['verification']['status'],'VERIFIED_INFEASIBLE')
        self.assertEqual(diagnosis['result']['status'],'INFEASIBLE')

if __name__=='__main__':unittest.main()
