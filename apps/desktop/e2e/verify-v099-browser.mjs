import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { chromium } from 'playwright'
import { createServer } from 'vite'

const server = await createServer({
  server: { host: '127.0.0.1', port: 0 },
  plugins: [
    {
      name: 'v099-browser-fixture',
      configureServer(server) {
        server.middlewares.use('/__v099-browser', async (_req, res) => {
          const html = (await readFile('index.html', 'utf8')).replace(
            'src="/src/main.tsx"',
            'src="/e2e/v099-browser.fixture.tsx"'
          )
          res.setHeader('Content-Type', 'text/html')
          res.end(await server.transformIndexHtml('/__v099-browser', html))
        })
      }
    }
  ]
})
let browser
try {
  await server.listen()
  browser = await chromium.launch({ channel: 'chromium', headless: true })
  for (const locale of ['en', 'zh']) {
    const context = await browser.newContext({ viewport: { width: 1000, height: 850 } })
    try {
      await context.route('**/*', route =>
        new URL(route.request().url()).hostname === '127.0.0.1' ? route.continue() : route.abort()
      )
      const page = await context.newPage()
      const errors = []
      page.on('pageerror', error => errors.push(error.message))
      await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__v099-browser?locale=${locale}`)
      const builtin = locale === 'zh' ? '内置 Chromium（默认）' : 'Built-in Chromium (default)'
      const system = locale === 'zh' ? 'Edge（系统浏览器）' : 'Edge (system)'
      await page.getByText(builtin, { exact: true }).waitFor()
      assert.equal(await page.getByRole('switch').getAttribute('aria-checked'), 'false')
      assert.equal(await page.evaluate(() => localStorage.getItem('v099-browser-choice')), null)
      await page.getByRole('button', { name: new RegExp(system.replace(/[()]/g, '\\$&')) }).click()
      await page.getByRole('button', { name: locale === 'zh' ? '使用此后端' : 'Use this backend' }).click()
      await page.waitForFunction(() => localStorage.getItem('v099-browser-choice') === 'Edge (system)')
      await page.reload()
      await page.getByText(system, { exact: true }).waitFor()
      assert.equal(await page.getByRole('switch').getAttribute('aria-checked'), 'false')
      assert.equal(
        await page
          .getByRole('button', { name: new RegExp(system.replace(/[()]/g, '\\$&')) })
          .getAttribute('aria-expanded'),
        'true'
      )
      assert.deepEqual(errors, [])
      console.log(
        JSON.stringify({
          locale,
          browser: browser.version(),
          builtinDefault: true,
          explicitChoiceAfterReload: true,
          realProfileOptIn: false
        })
      )
    } finally {
      await context.close()
    }
  }
} finally {
  await browser?.close()
  await server.close()
}
