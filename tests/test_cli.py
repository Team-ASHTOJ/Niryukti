#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--binary',default='build/vantage');a=p.parse_args();binary=str(Path(a.binary).resolve())
sys.path.insert(0,str(root/'python'));from vantage import Model

def run(*args):return subprocess.run([binary,*map(str,args)],text=True,capture_output=True)

with tempfile.TemporaryDirectory(prefix='vantage-cli-tests-') as tmp:
    tmp=Path(tmp);sol=tmp/'solution.json'
    qplib=tmp/'parsed.json'
    # Advisory output must reject the same invalid option domains as real solves.
    for args in [('analyze', '--tol', 'nan'), ('analyze', '--time-limit', '-1'),
                 ('analyze', '--scaling', 'bogus'), ('solve', '--method', 'bogus'),
                 ('solve', '--threads', '0')]:
        cmd, *flags = args
        if cmd == 'solve': flags.append('--dry-run')
        bad = run(cmd, root/'examples/toy.lp', *flags)
        assert bad.returncode != 0, (args, bad.stdout)
    # Auto dry-run and analyze expose deterministic, machine-readable choices.
    analysis=run('analyze',root/'examples/toy.lp','--device','cpu')
    assert analysis.returncode==0,analysis.stderr
    analyzed=json.loads(analysis.stdout)['analysis']
    assert analyzed['problem_class']=='LP' and analyzed['selection']['device']=='cpu'
    assert analyzed['model']['variables']==2 and analyzed['model']['nonzeros']==2
    dry=run('solve',root/'examples/toy.lp','--auto','--device','cpu','--dry-run')
    assert dry.returncode==0,dry.stderr
    dry_json=json.loads(dry.stdout)
    assert dry_json['mode']=='auto' and dry_json['selection']['device']=='cpu'
    auto_result=run('solve',root/'examples/toy.lp','--auto','--device','cpu')
    assert auto_result.returncode==0,auto_result.stderr
    auto_json=json.loads(auto_result.stdout)
    assert auto_json['selection']['automatic'] and auto_json['selection']['device']=='cpu'
    assert auto_json['selection']['method']=='revised-primal-simplex'
    assert auto_json['verification_certificate']['verification']['valid']
    qp=run('analyze',root/'examples/coupled_dispatch.json','--device','cpu')
    assert qp.returncode==0,qp.stderr
    qp_data=json.loads(qp.stdout)['analysis']
    assert qp_data['problem_class']=='QP'
    assert qp_data['model']['quadratic_off_diagonal_terms']>0
    milp=run('analyze',root/'examples/integer_dispatch.json','--device','cpu')
    assert json.loads(milp.stdout)['analysis']['problem_class']=='MIQP'
    actual_milp=run('solve',root/'examples/supply_chain.json','--auto','--device','cpu')
    assert actual_milp.returncode==0,actual_milp.stderr
    actual_mip=json.loads(actual_milp.stdout)
    assert actual_mip['selection']['method']=='branch-and-bound'
    assert actual_mip['selection']['relaxation_method']=='revised-primal-simplex'
    assert actual_mip['selection']['configuration']['branching']=='fractional'
    assert actual_mip['selection']['configuration']['cuts_enabled'] is False
    miqp_path=tmp/'miqp-analysis.json'
    miqp_data=json.loads((root/'examples/coupled_dispatch.json').read_text())
    miqp_data['variables'][0].update(type='binary',lb=0,ub=1)
    for variable in miqp_data['variables'][1:]:variable['type']='integer'
    miqp_path.write_text(json.dumps(miqp_data))
    miqp=run('analyze',miqp_path,'--device','cpu')
    assert miqp.returncode==0,miqp.stderr
    miqp_result=json.loads(miqp.stdout)['analysis']
    assert miqp_result['problem_class']=='MIQP' and miqp_result['model']['binary_variables']==1
    r=run('convert',root/'examples/toy.qplib',qplib);assert r.returncode==0,r.stderr
    r=run('solve',qplib);assert r.returncode==0,r.stderr
    qdata=json.loads(r.stdout);assert qdata['status']=='OPTIMAL'
    assert abs(qdata['objective']+1)<1e-5
    bad_qplib=tmp/'bad.qplib';bad_qplib.write_text((root/'examples/toy.qplib').read_text().replace('\n2\n3\n','\n2junk\n3\n'))
    assert run('inspect',bad_qplib).returncode==1
    r=run('solve',root/'examples/toy.lp','--json-out',sol);assert r.returncode==0,r.stderr
    data=json.loads(sol.read_text());assert data['status']=='OPTIMAL'
    assert run('verify',root/'examples/toy.lp',sol).returncode==0
    pristine=dict(data)
    data['objective']+=10;sol.write_text(json.dumps(data));assert run('verify',root/'examples/toy.lp',sol).returncode==2
    data=dict(pristine); data['dual']=[10.0]*len(data['dual']);sol.write_text(json.dumps(data));assert run('verify',root/'examples/toy.lp',sol).returncode==2
    for corrupt in ('fingerprint','nan','inf'):
        data=json.loads(json.dumps(pristine))
        if corrupt=='fingerprint':data['model']['fingerprint']='wrong-model'
        else:data['primal'][0]=float(corrupt)
        sol.write_text(json.dumps(data));assert run('verify',root/'examples/toy.lp',sol).returncode!=0
    certificate=tmp/'certificate.json'
    r=run('solve',root/'examples/toy.lp','--certificate-out',certificate);assert r.returncode==0,r.stderr
    assert run('verify',root/'examples/toy.lp',certificate).returncode==0
    assert json.loads(certificate.read_text())['verification_evidence']['tree_optimality_replayed'] is False
    data=dict(pristine)
    data['primal']=[0,0];sol.write_text(json.dumps(data));assert run('verify',root/'examples/toy.lp',sol).returncode==2
    assert run('solve',root/'examples/toy.lp','--device','imaginary').returncode==1
    assert run('solve',root/'examples/toy.lp','--tol','nan').returncode==1
    assert run('solve',root/'examples/toy.lp','--integer-tol','nan').returncode==1
    for option in ('--method', '--scaling', '--branching', '--primal-weight', '--primal-heuristic'):
        assert run('solve',root/'examples/toy.lp',option,'invalid').returncode==1
    assert run('solve',root/'examples/toy.lp','--method','halpern').returncode==1
    assert run('solve',root/'examples/dispatch.json','--method','halpern','--no-adaptive').returncode==1
    for scaling in ('ruiz', 'combined'):
        r=run('solve',root/'examples/toy.lp','--method','halpern','--no-adaptive','--scaling',scaling)
        assert r.returncode==0,r.stderr
        data=json.loads(r.stdout)
        assert data['options']['method']=='halpern' and data['options']['scaling']==scaling
        assert data['accuracy']['kkt_error']<=1e-6
    for method in ('rhpdhg','r2hpdhg'):
        r=run('solve',root/'examples/toy.lp','--method',method,'--no-adaptive','--primal-weight','pid')
        assert r.returncode==0,r.stderr
        data=json.loads(r.stdout)
        assert data['options']['method']==method and data['accuracy']['kkt_error']<=1e-6
    r=run('solve',root/'examples/toy.lp','--polishing','--power-iterations','20')
    assert r.returncode==0,r.stderr
    assert json.loads(r.stdout)['performance']['operator_norm_estimate']>0
    assert run('solve',root/'examples/toy.lp','--power-iterations','-1').returncode==1
    assert run('solve',root/'examples/dispatch.json','--polishing').returncode==1
    # Fixed MPS: embedded blanks in names and an initially blank bound-set field.
    fixed=tmp/'fixed.mps'
    row=lambda kind,name: f' {kind}  {name:<8}\n'
    column=lambda name,row,value: f'    {name:<8}  {row:<8}  {value:>12}\n'
    fixed.write_text('NAME FIXED\nROWS\n'+row('N','OBJ')+row('G','DEM AND')+'COLUMNS\n'+column('FLOW A','OBJ','1')+column('FLOW A','DEM AND','1')+'RHS\n'+column('RHS','DEM AND','2')+'BOUNDS\n'+f' UP {"":8}  {"FLOW A":8}  {"3":>12}\n'+'ENDATA\n')
    r=run('solve',fixed,'--method','simplex');assert r.returncode==0,r.stderr
    assert abs(json.loads(r.stdout)['objective']-2)<1e-8
    # Malformed sparse JSON and malformed MPS may not crash or enter the engine.
    malformed=[{'variables':[],'objective':{'linear':[1]},'constraints':[]},
               {'variables':[{'name':'x'}],'objective':{'linear':[1]},'constraints':[{'coefficients':{'missing':1},'lb':2}]}]
    for i,data in enumerate(malformed):
        path=tmp/f'bad{i}.json';path.write_text(json.dumps(data));r=run('solve',path);assert r.returncode==1,(r.returncode,r.stderr)
    path=tmp/'bad.mps';path.write_text("NAME BAD\nROWS\n N OBJ\nCOLUMNS\n M 'MARKER' 'INTEND'\nENDATA\n");assert run('solve',path).returncode==1
    # Certificate verification recomputes the proof and rejects a tampered ray.
    path=tmp/'infeasible.lp'
    path.write_text('Minimize\n x\nSubject To\n a: x + y >= 3\n b: x + y <= 1\nBounds\n 0 <= x <= 10\n 0 <= y <= 10\nEnd\n')
    r=run('solve',path,'--json-out',sol)
    data=json.loads(sol.read_text());assert data['status']=='INFEASIBLE',data['status']
    checked=run('verify',path,sol);assert checked.returncode==0,checked.stdout
    assert json.loads(checked.stdout)['status']=='VERIFIED_INFEASIBLE'
    data['certificate']['dual_ray']=[0,0];sol.write_text(json.dumps(data))
    assert run('verify',path,sol).returncode==2
    # A changed model is rejected unless parametric warm-start use is explicit.
    model=Model();x=model.add_var('x',ub=10);y=model.add_var('y',ub=10)
    model.add_constraint({x:1,y:2},'>=',4);model.set_objective({x:1,y:1})
    first=model.solve(binary=binary,device='cpu');assert first['status']=='OPTIMAL'
    experiment=model.solve(binary=binary,method='halpern',adaptive=False,scaling='combined')
    assert experiment['status']=='OPTIMAL'
    assert abs(experiment['objective']-first['objective'])<1e-5
    mip=Model();x=mip.add_var('x',binary=True);y=mip.add_var('y',binary=True)
    mip.add_constraint({x:2,y:2},'<=',3);mip.set_objective({x:-1,y:-1})
    reliable=mip.solve(binary=binary,branching='reliability')
    assert reliable['status']=='OPTIMAL' and abs(reliable['objective']+1)<1e-6
    enhanced=mip.solve(binary=binary,branching='reliability',cuts=True,primal_heuristic='all')
    assert enhanced['status']=='OPTIMAL' and abs(enhanced['objective']+1)<1e-6
    assert enhanced['mip']['cuts_added']>0
    assert enhanced['options']['primal_heuristic']=='all'
    model.data['constraints'][0]['lb']=4.2
    try:model.solve(binary=binary,warm_start=first);raise AssertionError('mismatched warm start accepted')
    except RuntimeError:pass
    second=model.solve(binary=binary,warm_start=first,allow_model_change=True);assert second['status']=='OPTIMAL'
    assert abs(second['objective']-2.1)<1e-5
print('CLI, malformed-input rejection, verification, Python API and parametric warm-start tests passed')
