// Real Chromium/rendered production component; native IPC is an explicit external boundary fixture.
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';

const server=await createServer({server:{host:'127.0.0.1',port:0},plugins:[{
 name:'update-source-browser-fixture',
 configureServer(server) {
  server.middlewares.use('/__update-source-fixture',async(_req,res)=>{
   const html=await server.transformIndexHtml('/__update-source-fixture','<html><head></head><body><div id="root"></div><script type="module" src="/e2e/update-delivery-source.fixture.tsx"></script></body></html>');
   res.setHeader('Content-Type','text/html');res.end(html);
  });
 }
}]});
let browser;
try {
 await server.listen();browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1100,height:800}});
 const port=server.httpServer.address().port;
 await page.goto('http://127.0.0.1:'+port+'/__update-source-fixture');
 const selector=page.getByRole('combobox',{name:'Update source'});
 await selector.click();
 assert.equal(await page.getByRole('option').count(),3);
 await page.getByRole('option',{name:'Channel 2 · Cloudflare'}).click();
 await page.getByText('https://update2.semantier.com',{exact:true}).waitFor();
 await page.reload();await page.getByText('https://update2.semantier.com',{exact:true}).waitFor();
 for(const [width,theme] of [[1100,'light'],[420,'dark']]) {
  await page.setViewportSize({width,height:800});
  await page.evaluate(theme=>document.documentElement.classList.toggle('dark',theme==='dark'),theme);
  await selector.focus();await page.keyboard.press('ArrowDown');
  assert.equal(await page.getByRole('option').count(),3);
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.keyboard.press('Escape');
 }
 console.log(JSON.stringify({browser:await browser.version(),platform:'Linux isolated Chromium',mode:'headless',viewports:[1100,420],options:3,selection:'channel2',reload:'PASS',keyboard:'PASS',overflow:'PASS',nativeIPC:'FIXTURE; installed Windows not tested'}));
} finally {if(browser)await browser.close();await server.close();}
