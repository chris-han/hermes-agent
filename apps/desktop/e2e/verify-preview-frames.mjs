/** Synthetic real-Electron/Playwright coverage for the embedded preview frame path. */
import assert from 'node:assert/strict'
import { createServer as httpServer } from 'node:http'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { createRequire } from 'node:module'
import { build } from 'esbuild'
import { _electron } from 'playwright'
import { createServer as viteServer } from 'vite'

const require = createRequire(import.meta.url)
const site = httpServer((req, res) => {
  const url = new URL(req.url, 'http://fixture')
  res.setHeader('Content-Type', 'text/html')
  if (url.pathname === '/login') { res.writeHead(302, { 'Set-Cookie': 'fixture_auth=synthetic; HttpOnly; SameSite=Lax; Path=/', Location: '/app' }); return res.end() }
  if (url.pathname === '/app') {
    if (!req.headers.cookie?.includes('fixture_auth=synthetic')) { res.statusCode = 403; return res.end('Not authenticated') }
    return res.end(`<h1>Outer authenticated</h1><button>Outer button</button><iframe id="same" src="/same" style="width:400px;height:120px"></iframe><iframe id="cross" src="http://localhost:${site.address().port}/cross" style="width:450px;height:290px"></iframe><iframe style="display:none" src="/hidden"></iframe><div style="height:0;overflow:hidden"><iframe src="/clipped"></iframe></div>`)
  }
  if (url.pathname === '/same') return res.end(req.headers.cookie?.includes('fixture_auth=synthetic') ? '<p>Same authenticated</p><button>Same button</button>' : '<p>Not authenticated</p>')
  if (url.pathname === '/cross' || url.pathname === '/replacement') return res.end(`<p>${url.pathname === '/cross' ? 'Cross content' : 'Replacement content'}</p><button onclick="document.querySelector('output').textContent='Clicked trusted: '+event.isTrusted">Cross action</button><label>Cross field<input id="field" value="previous" oninput="document.querySelector('#typed').textContent='Typed trusted: '+event.isTrusted+' '+this.value"></label><output></output><p id="typed"></p><iframe src="http://127.0.0.1:${site.address().port}/nested" style="width:350px;height:90px"></iframe>`)
  if (url.pathname === '/nested') return res.end('<p>Nested cross content</p><button>Nested action</button>')
  if (url.pathname === '/hidden') return res.end('<p>Hidden private frame</p>')
  if (url.pathname === '/clipped') return res.end('<p>Clipped private frame</p>')
  return res.end(`<h1>Ordinary page</h1><button onclick="document.querySelector('output').textContent='Ordinary trusted: '+event.isTrusted">Ordinary action</button><output></output>`)
})
const listen = server => new Promise(resolve => server.listen(0, '::', resolve))
const temp = await mkdtemp(path.join(os.tmpdir(), 'preview-frames-e2e-'))
let vite, app
try {
  await listen(site)
  vite = await viteServer({ configFile: path.resolve('vite.config.ts'), server: { host: '127.0.0.1', port: 0, strictPort: false } })
  await vite.listen()
  const host = `http://127.0.0.1:${vite.httpServer.address().port}/e2e/preview-frames.fixture.html`
  const bundle = path.join(temp, 'main.cjs')
  await build({ entryPoints: ['e2e/preview-frames.electron.mjs'], outfile: bundle, bundle: true, platform: 'node', format: 'cjs', external: ['electron'] })
  app = await _electron.launch({ executablePath: require('electron'), args: ['--no-sandbox', bundle], env: { ...process.env, PREVIEW_FIXTURE_HOST: host, PREVIEW_FIXTURE_PRELOAD: path.resolve('e2e/preview-frames.preload.cjs'), XDG_CONFIG_HOME: temp } })
  const page = await app.firstWindow()
  await page.waitForFunction(() => window.fixture)
  const guestId = await page.evaluate(url => window.fixture.open(url), `http://127.0.0.1:${site.address().port}/login`)
  await page.waitForTimeout(750)
  const read = () => page.evaluate(() => window.fixture.read())
  const act = action => page.evaluate(value => window.fixture.act(value), action)
  let result = await read()
  for (const text of ['Outer authenticated', 'Same authenticated', 'Cross content', 'Nested cross content']) assert.ok(result.text.includes(text), text)
  assert.ok(!result.text.includes('Hidden private frame'))
  assert.ok(!result.text.includes('Clipped private frame'))
  const processIds = await app.evaluate(({ webContents }, id) => webContents.fromId(id).mainFrame.framesInSubtree.map(frame => frame.processId), guestId)
  assert.ok(new Set(processIds).size >= 2, 'cross-origin frame must be out of process')
  result = await act({ kind: 'elements', full: true })
  assert.equal(result.success, true, JSON.stringify(result))
  for (const label of ['Outer button', 'Same button', 'Cross action', 'Cross field', 'Nested action']) assert.ok(result.elements.some(element => element.label.includes(label)), label)
  const sameRef = result.elements.find(element => element.label.includes('Same button')).ref
  const nestedRef = result.elements.find(element => element.label.includes('Nested action')).ref
  assert.equal((await act({ kind: 'click', ref: sameRef })).success, true, 'same-origin frame click')
  assert.equal((await act({ kind: 'click', ref: nestedRef })).success, true, 'nested OOPIF click')
  const old = result.elements.find(element => element.label.includes('Cross action')).ref
  assert.ok(old.startsWith('frame:'))
  const field = result.elements.find(element => element.label.includes('Cross field')).ref
  result = await act({ kind: 'click', ref: old })
  assert.equal(result.success, true, JSON.stringify(result))
  await page.evaluate(() => window.fixture.zoom(1.25))
  await page.waitForTimeout(200)
  result = await act({ kind: 'click', ref: old })
  assert.equal(result.success, true, 'frame click at non-default zoom: ' + JSON.stringify(result))
  result = await act({ kind: 'type', ref: field, text: 'A汉😊' })
  assert.equal(result.success, true, JSON.stringify(result))
  result = await read()
  assert.ok(result.text.includes('Clicked trusted: true'), result.text)
  assert.ok(result.text.includes('Typed trusted: true A汉😊'), result.text)
  await page.evaluate(() => window.fixture.guestScript("document.querySelector('#cross').src='http://localhost:' + location.port + '/replacement'"))
  await page.waitForFunction(async () => { try { return (await window.fixture.read()).text.includes('Replacement content') } catch { return false } })
  result = await act({ kind: 'click', ref: old })
  assert.equal(result.success, false, 'old frame ref must fail')
  result = await act({ kind: 'elements', full: true })
  const replacementRef = result.elements.find(element => element.label.includes('Cross action')).ref
  assert.notEqual(replacementRef, old)
  await page.evaluate(() => window.fixture.guestScript("let f=document.querySelector('#cross');let n=f.cloneNode();f.replaceWith(n)"))
  await page.waitForTimeout(500)
  result = await act({ kind: 'click', ref: replacementRef })
  assert.equal(result.success, false, 'replaced frame ref must fail')
  assert.equal(await page.evaluate(() => window.fixture.guestId()), guestId, 'frame navigation preserves guest identity')
  // Another renderer and another partition cannot address this guest.
  const secondWindowId = await app.evaluate(async ({ BrowserWindow }, host) => {
    const second = new BrowserWindow({ show: false, webPreferences: { contextIsolation: true, preload: process.env.PREVIEW_FIXTURE_PRELOAD } })
    await second.loadURL(host)
    return second.id
  }, host)
  let second
  for (const candidate of app.windows()) {
    const win = await app.browserWindow(candidate).catch(() => null)
    if (win && await win.evaluate(value => value.id) === secondWindowId) second = candidate
  }
  assert.ok(second, 'second renderer exists')
  await second.waitForFunction(() => window.hermesDesktop?.previewFrames)
  await assert.rejects(second.evaluate(id => window.hermesDesktop.previewFrames({ webContentsId: id, operation: 'list' }), guestId), /WRONG_OWNER/)
  const otherId = await page.evaluate(url => window.fixture.otherPartition(url), `http://127.0.0.1:${site.address().port}/ordinary`)
  await assert.rejects(page.evaluate(id => window.hermesDesktop.previewFrames({ webContentsId: id, operation: 'list' }), otherId), /WRONG_OWNER/)
  await app.evaluate(({ BrowserWindow }, id) => BrowserWindow.fromId(id).close(), secondWindowId)
  await page.evaluate(id => window.hermesDesktop.setPreviewGuestHidden(id, true), guestId)
  await assert.rejects(read(), /PANE_UNAVAILABLE/)
  await page.evaluate(id => window.hermesDesktop.setPreviewGuestHidden(id, false), guestId)
  result = await read()
  assert.ok(result.text.includes('Same authenticated'), 'same guest authentication remains available')
  await page.evaluate(url => window.fixture.guestLoad(url), `http://127.0.0.1:${site.address().port}/ordinary`)
  await page.waitForFunction(async () => { try { return (await window.fixture.read()).text.includes('Ordinary page') } catch { return false } })
  result = await act({ kind: 'elements', full: true })
  const ordinary = result.elements.find(element => element.label === 'Ordinary action')
  assert.ok(ordinary && !ordinary.ref.startsWith('frame:'))
  result = await act({ kind: 'click', ref: ordinary.ref })
  assert.equal(result.success, true, JSON.stringify(result))
  assert.ok((await read()).text.includes('Ordinary trusted: true'))
  await page.evaluate(() => window.fixture.dispose())
  assert.equal(await app.evaluate(({ webContents }, id) => webContents.fromId(id).debugger.isAttached(), guestId), false)
  console.log('PASS: real Electron frame reads, OOPIF/nested inventory, trusted click/type, navigation, replacement, hidden frame, ownership isolation, ordinary page, debugger cleanup')
} finally {
  await app?.close()
  await vite?.close()
  await new Promise(resolve => site.close(resolve))
  await rm(temp, { recursive: true, force: true })
}
