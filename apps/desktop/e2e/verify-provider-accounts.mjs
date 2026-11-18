import assert from 'node:assert/strict'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { chromium } from 'playwright'
import { createServer } from 'vite'

const evidence = process.env.ACCOUNTS_EVIDENCE
if (!evidence) throw new Error('Set ACCOUNTS_EVIDENCE to the selected release output')
await mkdir(evidence, { recursive: true })
const server = await createServer({
  server: { host: '127.0.0.1', port: 0 },
  plugins: [{ name: 'provider-accounts-fixture', configureServer(server) {
    server.middlewares.use('/__provider-accounts', async (_req, res) => {
      const html = (await readFile('index.html', 'utf8')).replace('src="/src/main.tsx"', 'src="/e2e/provider-accounts.fixture.tsx"')
      res.setHeader('Content-Type', 'text/html')
      res.end(await server.transformIndexHtml('/__provider-accounts', html))
    })
  } }]
})
let browser
const checks = []
try {
  await server.listen()
  browser = await chromium.launch({ channel: 'chromium', headless: true })
  for (const locale of ['en', 'zh']) {
    for (const scenario of ['connected', 'unconnected', 'connected-only', 'nous', 'alternative']) {
      const context = await browser.newContext({ viewport: { width: 1050, height: 850 } })
      try {
        await context.route('**/*', route => new URL(route.request().url()).hostname === '127.0.0.1' ? route.continue() : route.abort())
        const page = await context.newPage()
        const errors = []
        page.on('pageerror', e => errors.push(e.message))
        await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__provider-accounts?locale=${locale}&scenario=${scenario}`)
        const title = locale === 'zh' ? 'Semantier 模型服务' : 'Semantier Model Provider'
        const other = locale === 'zh' ? '其他提供方' : 'Other providers'
        const recommended = locale === 'zh' ? '推荐' : 'Recommended'
        await page.getByText(scenario === 'nous' ? 'Nous Portal' : scenario === 'alternative' ? other : title, { exact: true }).waitFor()
        assert.equal(await page.getByText(recommended, { exact: true }).count(), scenario === 'unconnected' ? 1 : 0)
        if (scenario === 'connected') {
          assert.equal(await page.getByText('Nous Portal', { exact: true }).count(), 0)
          await page.getByRole('button', { name: other, exact: true }).click()
          await page.getByText('Nous Portal', { exact: true }).waitFor()
        }
        if (scenario === 'alternative') {
          await page.getByRole('button', { name: other, exact: true }).click()
          await page.getByText('MiniMax', { exact: true }).waitFor()
        }
        if (scenario === 'nous') {
          assert.equal(await page.getByText(title, { exact: true }).count(), 0)
          assert.equal(await page.getByText(/300\+|推荐使用 Semantier|recommended way to run/).count(), 0)
        }
        assert.deepEqual(errors, [])
        await page.screenshot({ path: path.join(evidence, `accounts-${locale}-${scenario}.png`), fullPage: true })
        checks.push({ locale, scenario, result: 'PASS' })
      } finally { await context.close() }
    }
  }
  await writeFile(path.join(evidence, 'accounts-rendering.json'), JSON.stringify({ result: 'PASS', browser: browser.version(), scope: 'REAL_SETTINGS_COMPONENT_SYNTHETIC_API_NO_INSTALL', checks }, null, 2))
  console.log(JSON.stringify({ result: 'PASS', checks }))
} finally {
  await browser?.close()
  await server.close()
}
