import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import { chromium } from 'playwright'
import { expect } from '@playwright/test'
const evidence = process.env.JOURNEY_EVIDENCE || '/tmp/windows-picker-evidence'
await mkdir(evidence, { recursive: true })
const browser = await chromium.launch({ headless: true })
const receipt = { browser: browser.version(), mode: 'isolated Linux Chromium headless; exact viewports; Windows shared CDP untouched', runtime: 'http://127.0.0.1:5188/provider-model-journey.html', cases: [], errors: [] }
try {
  for (const width of [1440, 390]) for (const locale of ['en', 'zh']) for (const colorScheme of ['light', 'dark']) {
    const context = await browser.newContext({ viewport: { width, height: 1000 }, colorScheme })
    const page = await context.newPage()
    page.setDefaultTimeout(15000)
    page.on('pageerror', e => receipt.errors.push(e.message))
    const row = { width, locale, colorScheme, screenshots: [], result: 'FAIL' }
    receipt.cases.push(row)
    const capture = async name => {
      const file = `${name}-${locale}-${colorScheme}-${width}.png`
      await page.screenshot({ path: `${evidence}/${file}` })
      row.screenshots.push(file)
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `${name} overflow ${width}`)
    }
    try {
      await page.goto(`${receipt.runtime}?locale=${locale}`)
      const add = page.getByRole('button', { name: locale === 'zh' ? '添加服务商' : 'Add Provider', exact: true })
      await expect(add).toBeVisible()
      await page.getByText('OpenAI API', { exact: true }).waitFor()
      const toolbar = page.locator('[data-provider-toolbar]')
      const search = toolbar.getByRole('textbox')
      await expect(search).toBeVisible()
      await capture('keys')
      await search.fill('no-such-provider')
      await expect(add).toBeVisible()
      await add.click()
      for (const name of ['kaiwen', 'LiteLLM Local', 'Semantier Cloud']) await expect(page.getByText(name, { exact: true })).toBeVisible()
      const key = page.getByLabel(locale === 'zh' ? 'API 密钥' : 'API Key', { exact: true })
      if (await key.count()) assert.equal(await key.inputValue(), '')
      await capture('custom')
      await page.getByRole('button', { name: 'Backup Models', exact: true }).click()
      const combos = page.getByRole('combobox')
      await combos.first().click()
      const providerSearch = page.locator('[cmdk-input]')
      await providerSearch.fill('LiteLLM')
      await expect(page.getByRole('option', { name: 'LiteLLM Local', exact: true })).toBeVisible()
      await capture('backup-provider-search')
      await page.keyboard.press('Escape')
      await combos.nth(1).click()
      await page.locator('[cmdk-input]').fill('shared')
      await expect(page.getByRole('option', { name: 'fixture/shared', exact: true })).toBeVisible()
      await capture('backup-model-search')
      await page.keyboard.press('Escape')
      await page.getByRole('button', { name: 'Auxiliary Models', exact: true }).click()
      await page.getByRole('button', { name: locale === 'zh' ? '更改' : 'Change', exact: true }).first().click()
      await page.getByRole('combobox').first().click()
      await page.locator('[cmdk-input]').fill('LiteLLM')
      await capture('aux-provider-search')
      await page.keyboard.press('Escape')
      await page.getByRole('combobox').nth(1).click()
      await page.locator('[cmdk-input]').fill('shared')
      await capture('aux-model-search')
      await page.keyboard.press('Escape')
      row.result = 'PASS'
      console.log(`PASS ${width} ${locale} ${colorScheme}`)
    } finally { await context.close() }
  }
  assert.equal(receipt.errors.length, 0)
} finally {
  await writeFile(`${evidence}/visual-receipt.json`, JSON.stringify(receipt, null, 2))
  await browser.close()
}
