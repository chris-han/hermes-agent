import assert from 'node:assert/strict'
import { mkdir, readFile, rename, rmdir, writeFile } from 'node:fs/promises'
import { chromium } from 'playwright'
import { expect } from '@playwright/test'
const evidence = process.env.FALLBACK_FOCUS_EVIDENCE || '/tmp/fallback-apply-evidence'
await mkdir(evidence, { recursive: true })
const api = async request => {
  const r = await fetch('http://127.0.0.1:55475/api', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source: 0, request: { profile: 'default', ...request } }) })
  const d = await r.json(); assert(r.ok, JSON.stringify(d)); return d
}
const fixture = await (await fetch('http://127.0.0.1:55475/fixture')).json()
for (const id of ['first', 'second']) await api({ path: '/api/providers/custom-endpoints', method: 'POST', body: { id, name: `Synthetic ${id}`, base_url: fixture.provider_origin + '/v1', model_list_url: fixture.provider_origin + '/catalog', api_key: 'fixture-key', model: '', make_default: false } })
const chain = [{ provider: 'custom:first', model: 'fixture/alpha', base_url: fixture.provider_origin + '/v1', key_env: 'FIXTURE_ROUTE_KEY', api_mode: 'chat_completions', reasoning_effort: 'none' }, { provider: 'custom:second', model: 'fixture/shared' }]
await api({ path: '/api/config', method: 'PUT', body: { config: { fallback_providers: chain } } })
const browser = await chromium.launch({ headless: true })
const receipt = { browser: browser.version(), checks: [], visual: [], errors: [] }
const pass = name => { receipt.checks.push(name); console.log('PASS', name) }
const home = '/tmp/fallback-focus-native-fixture/source-a/config.yaml'
let moved = false
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  page.setDefaultTimeout(30000)
  page.on('pageerror', e => receipt.errors.push(e.message))
  await page.goto('http://127.0.0.1:5188/provider-model-journey.html?locale=en')
  await page.getByRole('button', { name: 'Backup Models', exact: true }).click()
  await expect(page.getByLabel('Remove', { exact: true })).toHaveCount(2)
  const choose = async (index, label) => {
    await page.getByRole('combobox', { name: `Reasoning ${index}` }).click()
    await page.mouse.move(1, 1)
    await page.getByRole('option', { name: label, exact: true }).focus()
    await page.keyboard.press('Enter')
  }
  await choose(1, 'High')
  assert.deepEqual((await api({ path: '/api/config' })).fallback_providers, chain)
  await page.getByRole('button', { name: 'Cancel', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Reasoning 1' })).toContainText('Off')
  await page.getByRole('button', { name: 'Add fallback', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Apply', exact: true })).toBeDisabled()
  await page.getByRole('button', { name: 'Cancel', exact: true }).click()
  await page.getByLabel('Remove', { exact: true }).last().click()
  assert.deepEqual((await api({ path: '/api/config' })).fallback_providers, chain)
  await page.getByRole('button', { name: 'Cancel', exact: true }).click()
  pass('reasoning/add/remove remain drafts; incomplete chain cannot Apply; Cancel restores saved chain')
  await choose(1, 'High'); await choose(2, 'Low')
  const expected = [{ ...chain[0], reasoning_effort: 'high' }, { ...chain[1], reasoning_effort: 'low' }]
  // Actual isolated filesystem persistence failure, not an intercepted success/failure response.
  await rename(home, home + '.before-apply'); moved = true; await mkdir(home)
  await page.getByRole('button', { name: 'Apply', exact: true }).click()
  await page.getByRole('alert').waitFor()
  assert((await readFile(home + '.before-apply', 'utf8')).includes('fixture/alpha'))
  await rmdir(home); await rename(home + '.before-apply', home); moved = false
  await page.getByRole('button', { name: 'Apply', exact: true }).click()
  await expect.poll(async () => (await api({ path: '/api/config' })).fallback_providers).toEqual(expected)
  await expect(page.getByRole('button', { name: 'Apply', exact: true })).toBeDisabled()
  pass('real failed disk persistence retains draft and original data; Retry acknowledges exact ordered routes/metadata')
  await page.reload()
  await page.getByRole('button', { name: 'Backup Models', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Reasoning 1' })).toContainText('High')
  await expect(page.getByRole('combobox', { name: 'Reasoning 2' })).toContainText('Low')
  await page.evaluate(() => window.__providerJourney.connect(1))
  await expect(page.getByLabel('Remove', { exact: true })).toHaveCount(0)
  await page.evaluate(() => window.__providerJourney.connect(0))
  await expect(page.getByRole('combobox', { name: 'Reasoning 1' })).toContainText('High')
  pass('reload and source switch preserve the acknowledged chain without crossing owners')
  await page.close()
  for (const width of [1440, 390]) for (const locale of ['en', 'zh']) for (const colorScheme of ['light', 'dark']) {
    const context = await browser.newContext({ viewport: { width, height: 1000 }, colorScheme })
    const p = await context.newPage(); p.setDefaultTimeout(30000)
    try {
      await p.goto(`http://127.0.0.1:5188/provider-model-journey.html?locale=${locale}`)
      await p.getByRole('button', { name: 'Backup Models', exact: true }).click()
      const select = p.getByRole('combobox', { name: /Reasoning|推理/ }).first()
      await select.click()
      const file = `backup-reasoning-${locale}-${colorScheme}-${width}.png`
      await p.screenshot({ path: `${evidence}/${file}` })
      assert(await p.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1))
      receipt.visual.push({ width, locale, colorScheme, screenshot: file, result: 'PASS' })
    } finally { await context.close() }
  }
} finally {
  if (moved) { await rmdir(home).catch(() => {}); await rename(home + '.before-apply', home) }
  await writeFile(`${evidence}/apply-receipt.json`, JSON.stringify(receipt, null, 2))
  await browser.close()
}
