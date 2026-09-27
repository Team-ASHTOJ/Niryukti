const {chromium}=require('@playwright/test');
(async()=>{
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||'/opt/google/chrome/chrome',args:['--no-sandbox']});
const page=await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});
const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto((process.env.DASHBOARD_URL||'http://127.0.0.1:8080'));await page.waitForSelector('.stats-grid').catch(async error=>{console.error(errors,await page.locator('main').innerText());throw error;});await page.evaluate(()=>document.fonts.ready);
if(await page.locator('.brand use[href="#nk-mark"]').count()!==1||await page.locator('.brand use[href="#nk-word"]').count()!==1)throw Error('NIRYUKTI brand mark is not installed');
const favicon=await page.request.get(new URL('/favicon.svg',page.url()).href);if(!favicon.ok()||!(await favicon.text()).includes('prefers-color-scheme'))throw Error('Theme-aware favicon unavailable');
// Theme toggle flips light/dark, persists, and changes the surface colour.
const bgOf=()=>page.evaluate(()=>getComputedStyle(document.body).backgroundColor);
const theme0=await page.evaluate(()=>document.documentElement.dataset.theme),bg0=await bgOf();
await page.locator('#theme-toggle').click();
if(await page.evaluate(()=>document.documentElement.dataset.theme)===theme0||await bgOf()===bg0)throw Error('Theme toggle failed');
if(await page.evaluate(()=>localStorage.getItem('niryukti-theme'))===theme0)throw Error('Theme choice not persisted');
await page.locator('#theme-toggle').click();
await page.screenshot({animations:'disabled',path:'/tmp/niryukti-overview.png',fullPage:true});
if(await page.locator('#pixel-snow').evaluate(c=>c.hidden||c.width>480||c.height>600))throw Error('PixelSnow shader unavailable or exceeded render budget');
const wave=page.locator('.wave-grid');
await wave.scrollIntoViewIfNeeded();await page.waitForTimeout(100);
const waveBefore=await wave.evaluate(c=>c.toDataURL());await page.waitForTimeout(150);
if(await wave.evaluate(c=>c.toDataURL())===waveBefore)throw Error('Wave field did not animate');
await page.locator('#effects-toggle').click();await page.waitForTimeout(80);
const pausedWave=await wave.evaluate(c=>c.toDataURL());await page.waitForTimeout(150);
if(await wave.evaluate(c=>c.toDataURL())===pausedWave)throw Error('Wave field stopped with Effects off');
await page.locator('#effects-toggle').click();
const animatedNav=page.locator('nav a[data-nav="benchmarks"]');
const navWidth=await animatedNav.evaluate(e=>e.getBoundingClientRect().width);
await animatedNav.hover();await page.waitForTimeout(75);
if(await animatedNav.locator('.scramble-visual').textContent()==='Benchmarks')throw Error('Navbar scramble did not start');
if(Math.abs(await animatedNav.evaluate(e=>e.getBoundingClientRect().width)-navWidth)>1)throw Error('Scramble shifted navbar layout');
await page.waitForTimeout(350);
if(await animatedNav.locator('.scramble-visual').textContent()!=='Benchmarks')throw Error('Navbar scramble did not settle');
await page.locator('nav a[data-nav="benchmarks"]').click();await page.waitForSelector('#case-search');
await page.locator('#case-search').fill('israel');await page.waitForTimeout(100);if(await page.locator('tbody tr[data-case]').count()!==1)throw Error('Search failed');
await page.locator('tr[data-case="israel"]').click();await page.waitForSelector('#drawer:not([hidden])');if(!await page.locator('#drawer').innerText().then(t=>t.includes('Original-space primal residual')))throw Error('Missing numerical evidence');await page.keyboard.press('Escape');
await page.locator('nav a[data-nav="solve"]').click();await page.waitForSelector('#solve-form');
await page.locator('#model-select').selectOption('dispatch');await page.locator('input[name=iterations]').fill('1000');const submitted=page.waitForResponse(r=>r.url().endsWith('/api/runs')&&r.request().method()==='POST');await page.locator('#run-button').click();const job=await (await submitted).json();if(job.options.iterations!==1000)throw Error('Iteration budget was not accepted');
await page.waitForSelector('#solve-output .pill.green',{timeout:30000});await page.screenshot({animations:'disabled',path:'/tmp/niryukti-solve.png',fullPage:true});
await page.locator('#platform-toggle').focus();await page.keyboard.press('ArrowDown');
if(!await page.locator('.mega-links a').first().evaluate(e=>e===document.activeElement))throw Error('Mega menu keyboard entry failed');
await page.screenshot({animations:'disabled',path:'/tmp/niryukti-navbar.png'});
await page.keyboard.press('Escape');
if(!await page.locator('#platform-toggle').evaluate(e=>e===document.activeElement))throw Error('Mega menu focus restore failed');
if(!await page.locator('#platform-panel').evaluate(e=>e.hidden))throw Error('Mega menu Escape failed');
const download=page.waitForEvent('download');await page.getByRole('link',{name:'Download solution'}).click();await download;
await page.locator('#model-file').setInputFiles({name:'test.lp',mimeType:'text/plain',buffer:Buffer.from('Minimize\n x\nSubject To\n c: x >= 3\nBounds\n 0 <= x <= 10\nEnd\n')});
await page.waitForFunction(()=>document.querySelector('#model-select')?.selectedOptions[0]?.textContent.includes('test.lp'));
await page.locator('#run-button').click();await page.waitForSelector('#solve-output .pill.green',{timeout:30000});
await page.locator('#platform-toggle').click();await page.locator('nav a[data-nav="models"]').click();await page.waitForSelector('.model-grid');
await page.setViewportSize({width:390,height:844});await page.goto((process.env.DASHBOARD_URL||'http://127.0.0.1:8080')+'/#overview');await page.waitForSelector('.stats-grid');await page.waitForTimeout(350);await page.screenshot({animations:'disabled',path:'/tmp/niryukti-mobile.png',fullPage:true});
const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);if(overflow)throw Error('Mobile horizontal overflow');
await page.locator('#menu-button').click();await page.locator('#platform-toggle').click();await page.screenshot({animations:'disabled',path:'/tmp/niryukti-navbar-mobile.png'});await page.locator('nav a[data-nav="verification"]').click();if(await page.locator('#menu-button').getAttribute('aria-expanded')!=='false')throw Error('Mobile menu did not close on navigation');await page.locator('#menu-button').click();await page.locator('nav a[data-nav="solve"]').click();await page.waitForSelector('#solve-form');
// Audit every route at desktop and mobile sizes, including newly added consoles.
for(const width of [1440,1920,2560,390]){
 await page.setViewportSize({width,height:900});
 for(const route of ['overview','benchmarks','models','solve','import','runs','verification','hardware','diagnostics','docs']){
  await page.goto((process.env.DASHBOARD_URL||'http://127.0.0.1:8080')+'/#'+route);
  await page.waitForSelector('main h1');
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error(`Overflow: ${route} at ${width}`);
  if(width===1440)await page.screenshot({animations:'disabled',path:`/tmp/niryukti-${route}-audit.png`,fullPage:true});
 }
}
await page.locator('#effects-toggle').click();
await page.reload();await page.waitForSelector('main h1');
if(!await page.locator('body').evaluate(b=>b.classList.contains('effects-off')))throw Error('Effects setting did not persist');
await page.emulateMedia({reducedMotion:'reduce'});
await page.setViewportSize({width:1440,height:900});
await animatedNav.hover();await page.waitForTimeout(75);
if(await animatedNav.locator('.scramble-visual').textContent()!=='Benchmarks')throw Error('Reduced motion did not disable scrambling');
if(await page.locator('.field-nodes').evaluate(e=>getComputedStyle(e).animationName)!=='none')throw Error('Reduced motion ignored');
await page.locator('#effects-toggle').click();
await page.setViewportSize({width:1440,height:900});
for(const model of ['refinery','supply_chain']){
 await page.goto((process.env.DASHBOARD_URL||'http://127.0.0.1:8080')+'/#solve');
 await page.waitForSelector('#solve-form');await page.locator('#model-select').selectOption(model);
 await page.locator('#run-button').click();
 await page.waitForSelector('a[href$="/download"]',{timeout:30000});
 await page.waitForSelector(model==='refinery'?'.refinery-schematic':'.tree-schematic');
 await page.screenshot({animations:'disabled',path:`/tmp/niryukti-${model}-audit.png`,fullPage:true});
}
if(errors.length)throw Error(errors.join('\n'));
console.log('PASS: desktop/mobile, navigation, search, details, live solve, model import, result download; no browser errors.');await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
