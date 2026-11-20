/** Linux renderer fixtures only. Run with local Vite on port 5196. */
import assert from 'node:assert/strict'
import { chromium } from 'playwright'
import { expect } from '@playwright/test'
const { openChromium } = await import('../../../../scripts/browser/chromium.mjs')
const browser = await openChromium(chromium)
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } })
  const errors = []
  page.on('pageerror', e => errors.push(e.message))
  await page.goto('http://127.0.0.1:5196/marketplace-journey.html')
  const selector = page.getByRole('combobox')
  await expect(selector).toContainText('Semantier Official')
  await expect(page.getByRole('alert')).toContainText('Fixture registry unavailable')
  await selector.click()
  const expected = ['Semantier Official', 'Hermes Index', 'Nous Official', 'browse.sh', 'Direct URL', 'All Sources']
  assert.deepEqual(await page.getByRole('option').allTextContents(), expected)
  await page.getByRole('option', { name: 'Nous Official', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Fixture third-party skill' })).toBeVisible()
  await page.reload()
  await expect(selector).toContainText('Nous Official')
  await page.getByRole('button', { name: 'Switch fixture profile' }).click()
  await expect(selector).toContainText('Semantier Official')
  await page.getByRole('button', { name: 'Switch fixture profile' }).click()
  await expect(selector).toContainText('Nous Official')
  await selector.focus()
  await page.keyboard.press('Enter')
  await page.getByRole('option', { name: 'Direct URL', exact: true }).click()
  await page.getByRole('textbox', { name: 'SKILL.md URL' }).fill('https://example.org/SKILL.md')
  await page.getByRole('button', { name: 'Preview', exact: true }).click()
  await expect(page.getByText('# Fixture preview', { exact: true })).toBeVisible()
  await page.setViewportSize({ width: 480, height: 760 })
  await expect(selector).toBeVisible()
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
  await page.screenshot({ path: '/tmp/semantier-marketplace-compact.png' })
  await page.evaluate(() => localStorage.clear())
  await page.goto('http://127.0.0.1:5196/marketplace-journey.html?integrity')
  await expect(page.getByRole('heading', { name: 'Unsigned fixture skill' })).toBeVisible()
  await expect(page.getByText('Package integrity verified', { exact: true })).toBeVisible()
  await expect(page.getByText('Community source · publisher unverified', { exact: true })).toBeVisible()
  await expect(page.getByText('Publisher verified: Semantier', { exact: true })).toHaveCount(0)
  assert.deepEqual(errors, [])
  console.log(JSON.stringify({ result: 'PASS', browser: browser.version(), host: 'WSL/Linux owned headless Chromium', checks: ['source order', 'truthful unavailable', 'saved source after reload', 'profile isolation', 'keyboard selector', 'URL preview', 'compact 480px', 'integrity distinct from publisher authentication'], fixture_only: true }))
} finally { await browser.close() }
