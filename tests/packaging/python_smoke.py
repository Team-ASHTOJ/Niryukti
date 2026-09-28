from niryukti import Model, NativeSession
m=Model();m.add_var('x',ub=10);m.add_constraint({'x':1},'>=',2);m.set_objective({'x':3})
r=m.solve(device='cpu',method='auto');assert r['status']=='OPTIMAL' and abs(r['objective']-6)<1e-6,r
with NativeSession(m) as s:
 r=s.solve();assert r['status']=='OPTIMAL' and abs(r['objective']-6)<1e-6,r
print('Installed Python subprocess and native APIs: PASS')

from niryukti.report import render_report
from niryukti.service import make_server
assert '&lt;script&gt;' in render_report(r,title='<script>bad</script>')
server=make_server(port=0,token='package-smoke-only');server.server_close()
print('Installed API service and report modules: PASS')
