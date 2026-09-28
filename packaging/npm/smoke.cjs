const assert = require('node:assert/strict');
const {solve,renderReport} = require('niryukti');
(async()=>{
 const model={name:'installed-node',sense:'min',variables:[{name:'x',lb:0,ub:10,type:'continuous'}],objective:{linear:[3]},constraints:[{name:'demand',coefficients:{x:1},lb:2}]};
 const r=await solve(model);assert.equal(r.status,'OPTIMAL');assert.ok(Math.abs(r.objective-6)<1e-6);
 await assert.rejects(solve('/missing/model.mps'));
 const controller=new AbortController();controller.abort();await assert.rejects(solve(model,{signal:controller.signal}));
 assert.ok(renderReport(r,{title:'<script>bad</script>'}).includes('&lt;script&gt;'));
 assert.throws(()=>renderReport({}));
 console.log('Installed Node solve, invalid input and cancellation: PASS');
})().catch(e=>{console.error(e);process.exitCode=1});
