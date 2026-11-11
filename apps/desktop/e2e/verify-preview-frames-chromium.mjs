/** Real owned headless Chromium: exercise the production in-page engine in
 * same-origin and cross-origin frames with trusted Playwright input. */
import assert from 'node:assert/strict'
import { createServer as httpServer } from 'node:http'
import { chromium } from 'playwright'
import { createServer as viteServer } from 'vite'

const server = httpServer((req, res) => {
  res.setHeader('Content-Type', 'text/html')
  if (req.url === '/same') return res.end('<p>Same text</p><button>Same action</button>')
  if (req.url === '/cross') return res.end('<p>Cross text</p><button onclick="document.querySelector(\'output\').textContent=event.isTrusted">Cross action</button><label>Cross field<input oninput="document.querySelector(\'#typed\').textContent=event.isTrusted+\':\'+this.value"></label><output></output><p id="typed"></p>')
  return res.end(`<p>Outer text</p><button>Outer action</button><iframe src="/same"></iframe><iframe src="http://localhost:${server.address().port}/cross"></iframe>`)
})
let vite, browser
try {
  await new Promise(resolve => server.listen(0, '::', resolve))
  vite = await viteServer({ configFile: 'vite.config.ts', server: { host: '127.0.0.1', port: 0 } })
  const { actEngineSource } = await vite.ssrLoadModule('/src/lib/preview-act/act-in-page.ts')
  const source = actEngineSource()
  browser = await chromium.launch({ channel: 'chromium', headless: true, args: ['--site-per-process'] })
  const context = await browser.newContext({ viewport: { width: 1100, height: 800 } })
  const page = await context.newPage()
  await page.goto(`http://127.0.0.1:${server.address().port}/`)
  const frames = page.frames()
  assert.equal(frames.length, 3)
  for (const [index, name] of ['Outer action', 'Same action', 'Cross action'].entries()) {
    const inventory = await frames[index].evaluate(source => {
      const act = Function('return (' + source + ')')()
      return act(document, {}, { kind: 'elements', full: true })
    }, source)
    assert.ok(inventory.elements.some(element => element.label === name), name)
  }
  const cross = frames.find(frame => frame.url().includes('/cross'))
  await cross.getByRole('button', { name: 'Cross action' }).click()
  await cross.getByLabel('Cross field').fill('synthetic')
  assert.equal(await cross.locator('output').textContent(), 'true')
  assert.equal(await cross.locator('#typed').textContent(), 'true:synthetic')
  await context.close()
  console.log('PASS: owned Linux Chromium/Playwright engine inventory and trusted same/cross-origin fixture input')
} finally {
  await browser?.close()
  await vite?.close()
  await new Promise(resolve => server.close(resolve))
}
