const {chromium}=require('@playwright/test');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||'/opt/google/chrome/chrome',args:['--no-sandbox']});
 try{
 const page=await browser.newPage({hasTouch:true});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.DASHBOARD_URL||'http://127.0.0.1:8080');await page.waitForSelector('.stats-grid');
 for(const status of ['OPTIMAL','ITERATION_LIMIT','TIME_LIMIT','MIXED','UNKNOWN']){
  await page.evaluate(status=>{
   const c={id:'hover-test',name:'Hover test',type:'LP',category:'Test',engines:{cpu:{status,statuses:[status],iterations:1000,primal:0.01,kkt:0.02,seconds:1,runs:3,solved:status==='OPTIMAL'?3:0}}};
   state.data.cases=[c];document.querySelector('#main').innerHTML=chart(c.id);
  },status);
  await page.locator('.bar-row').hover();const tip=page.locator('#graph-tooltip');await tip.waitFor({state:'visible'});
  const text=await tip.innerText();assert.equal(await tip.locator('dl > div').count(),4);assert(text.includes('Iterations (median)'));assert(text.includes('KKT error'));
  if(status==='ITERATION_LIMIT')assert(text.includes('Iteration budget exhausted'));
  if(status==='OPTIMAL')assert(text.includes('Target reached (solver reported)'));
  await page.keyboard.press('Escape');assert(await tip.isHidden());
 }
 await page.evaluate(()=>{const c=state.data.cases[0];c.nonzeros=10;c.engines.cpu.status='OPTIMAL';c.engines.cpu.statuses=['OPTIMAL'];c.engines.cpu.solved=3;c.engines.cuda={...c.engines.cpu,status:'OPTIMAL',seconds:0.5,objective_error:0};document.querySelector('#main').innerHTML=benchmarkPlots()+evidencePlots();});
 await page.locator('.research-plot [data-graph-detail]').first().hover();assert((await page.locator('#graph-tooltip').innerText()).includes('Outcome'));
 await page.locator('.evidence-grid circle[data-graph-detail]').first().focus();assert((await page.locator('#graph-tooltip').innerText()).includes('KKT error'));
 await page.evaluate(()=>{document.querySelector('#main').innerHTML=convergence({name:'Limited sample',state:'finished',options:{tol:1e-6,iterations:100},result:{status:'ITERATION_LIMIT',accuracy:{kkt_error:0.01}},trace:'iter=100 objective=1 primal=1e-8 dual=1e-8 gap=1e-8'});});
 await page.locator('.graph-sample').focus();let text=await page.locator('#graph-tooltip').innerText();
 assert(text.includes('Iteration budget exhausted'));assert(text.includes('Sample residual / target'));assert(!text.includes('Target reached'));assert.equal(await page.locator('#graph-tooltip dl > div').count(),4);
 await page.setViewportSize({width:390,height:844});await page.locator('.graph-sample').scrollIntoViewIfNeeded();await page.waitForTimeout(100);await page.locator('.graph-sample').tap({force:true});
 const box=await page.locator('#graph-tooltip').boundingBox();assert(box&&box.x>=0&&box.x+box.width<=391);
 assert.deepEqual(errors,[]);console.log('PASS: graph hover outcomes, metrics, keyboard dismissal, below-target limited sample, mobile bounds');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
