// Presentation-only fixture: no native IPC, backend execution, or real account.
/* global document, innerWidth, window */
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { chromium } from 'playwright'
import { createServer } from 'vite'

const server = await createServer({ server: { host: '127.0.0.1', port: 0 }, plugins: [{
  name: 'account-capabilities-fixture', configureServer(vite) {
    vite.middlewares.use('/__account-capabilities', async (_request, response) => {
      const entry = (await readFile('index.html', 'utf8')).replace('src="/src/main.tsx"',
        'src="/e2e/account-capabilities.fixture.tsx"')
      response.setHeader('Content-Type', 'text/html')
      response.end(await vite.transformIndexHtml('/__account-capabilities', entry))
    })
  }
}] })
let browser
try {
  await server.listen()
  browser = await chromium.launch({ headless: true })
  const port = server.httpServer.address().port
  for (const [locale, labels] of Object.entries({
    en: { choose: 'Choose and enable', pause: 'Pause', enable: 'Request enablement', confirm: 'Confirm request',
      disable: 'Turn preference off', noFolder: 'No folder selected', requested: 'Enablement requested.',
      inFlight: 'Uploads already sent may still finish after pausing or switching accounts.',
      unconfirmed: 'File Sync could not confirm that previously sent work has stopped' },
    zh: { choose: '选择并启用', pause: '暂停', enable: '请求启用', confirm: '确认请求',
      disable: '关闭设置', noFolder: '尚未选择文件夹', requested: '已请求启用。',
      inFlight: '暂停或切换账户后，已发送的上传仍可能完成。', unconfirmed: '无法确认先前发送的任务已停止' }
  })) {
    for (const viewport of [{ width: 1280, height: 800 }, { width: 390, height: 844 }]) {
      const context = await browser.newContext({ viewport, reducedMotion: 'reduce' })
      try {
        const page = await context.newPage()
        const errors = []
        page.on('pageerror', error => errors.push(error.message))
        await page.goto(`http://127.0.0.1:${port}/__account-capabilities?locale=${locale}`)
        await page.getByText(labels.noFolder, { exact: true }).waitFor()
        await page.getByText(labels.inFlight, { exact: true }).waitFor({ state: 'visible' })
        const enable = page.getByRole('button', { name: labels.enable, exact: true })
        await enable.focus()
        await enable.press('Enter')
        await page.getByRole('button', { name: labels.confirm, exact: true }).click()
        await page.getByText(labels.requested, { exact: false }).waitFor()
        assert.equal(await page.getByText(labels.noFolder, { exact: true }).count(), 1)
        await page.getByRole('button', { name: labels.choose, exact: true }).click()
        await page.getByText('fixture-folder', { exact: true }).waitFor()
        await page.getByText('Synced', { exact: true }).waitFor()
        await page.getByRole('button', { name: labels.disable, exact: true }).click()
        await page.getByRole('button', { name: labels.enable, exact: true }).waitFor()
        assert.equal(await page.getByText('fixture-folder', { exact: true }).count(), 1)
        await page.getByRole('button', { name: labels.pause, exact: true }).click()
        await page.getByText('Paused', { exact: true }).waitFor()
        assert.deepEqual(await page.evaluate(() => window.capabilityFixtureEvents),
          ['winman:true', 'file:choose', 'winman:false', 'file:pause'])
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
        await page.goto(`http://127.0.0.1:${port}/__account-capabilities?locale=${locale}&unconfirmed=1`)
        await page.getByRole('alert').getByText(labels.unconfirmed, { exact: false }).waitFor()
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
        assert.deepEqual(errors, [])
      } finally {await context.close()}
    }
  }
  console.log(`PASS Chromium ${browser.version()} headless; 1280x800 and 390x844; en and zh; fixture-only`)
} finally {
  await browser?.close()
  await server.close()
}
