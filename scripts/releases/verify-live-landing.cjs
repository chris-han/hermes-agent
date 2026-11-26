const {chromium}=require(process.argv[2]+'/@playwright/test');
const fs=require('node:fs');
(async()=>{
  const expected='https://assets.semantier.cn/releases/installers/semantier-setup-ch1.exe';
  const browser=await chromium.launch({headless:true});
  const observations=[];
  try {
    for(const host of ['https://semantier.cn/','https://www.semantier.cn/']) {
      const page=await browser.newPage();
      const catalogRequests=[];
      page.on('request',r=>{if(/\/(?:manual\/)?catalog\.json(?:\?|$)/i.test(r.url())) catalogRequests.push(r.url());});
      const response=await page.goto(host,{waitUntil:'domcontentloaded',timeout:30000});
      if(!response || !response.ok()) throw Error('Landing HTTP failure: '+host);
      const link=page.locator('a[href="'+expected+'"]');
      await link.first().waitFor({state:'visible',timeout:15000});
      const href=await link.first().getAttribute('href');
      if(new URL(href,page.url()).href!==expected) throw Error('Landing installer link mismatch: '+host);
      if(catalogRequests.length) throw Error('Landing still calls catalog: '+host);
      observations.push({host,url:page.url(),setupHref:expected,visibleFixedLinks:await link.count(),catalogRequests:0});
      await page.close();
    }
  } finally {await browser.close();}
  fs.writeFileSync(process.argv[3],JSON.stringify({result:'LIVE_LANDING_FIXED_SETUP_PASS',observations},null,2));
})().catch(error=>{console.error(error.stack||String(error));process.exitCode=1;});
