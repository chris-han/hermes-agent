import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import { chromium } from 'playwright'
import { expect } from '@playwright/test'
const evidence = process.env.FALLBACK_FOCUS_EVIDENCE || '/tmp/fallback-focus-evidence'
await mkdir(evidence, { recursive: true })
const browser = await chromium.launch({ headless: true })
const receipt = { browser: browser.version(), cases: [], errors: [] }
try {
  for (const width of [1440, 390]) for (const locale of ['en', 'zh']) for (const colorScheme of ['light', 'dark']) {
    const context = await browser.newContext({ viewport: { width, height: 1000 }, colorScheme })
    const page = await context.newPage()
    page.setDefaultTimeout(60000)
    page.on('pageerror', e => receipt.errors.push(e.message))
    try {
      await page.goto(`http://127.0.0.1:5188/provider-model-journey.html?locale=${locale}`)
      await page.getByText('OpenAI API', { exact: true }).waitFor()
      const search = page.locator('[data-provider-toolbar] input')
      await search.click()
      await page.keyboard.press('Tab')
      await expect(page.getByRole('button', { name: locale === 'zh' ? '添加服务商' : 'Add Provider', exact: true })).toBeFocused()
      await page.keyboard.press('Shift+Tab')
      await expect(search).toBeFocused()
      const geometry = await search.evaluate(input => {
        const style = getComputedStyle(input), r = input.getBoundingClientRect()
        const extent = parseFloat(style.outlineWidth) + parseFloat(style.outlineOffset)
        const clips = []
        for (let el = input.parentElement; el; el = el.parentElement) {
          const s = getComputedStyle(el), b = el.getBoundingClientRect()
          if (/auto|hidden|scroll|clip/.test(s.overflowY)) clips.push({ top: b.top, bottom: b.bottom, left: b.left, right: b.right, overflow: s.overflowY, scrollTop: el.scrollTop })
        }
        return { outline: style.outlineStyle, extent, top: r.top - extent, bottom: r.bottom + extent, left: r.left - extent, right: r.right + extent, clips }
      })
      const full = geometry.outline !== 'none' && geometry.clips.every(c => geometry.top >= c.top - 0.5 && geometry.bottom <= c.bottom + 0.5 && geometry.left >= c.left - 0.5 && geometry.right <= c.right + 0.5)
      const stem = `${locale}-${colorScheme}-${width}`
      await page.screenshot({ path: `${evidence}/keys-focus-${stem}.png` })
      receipt.cases.push({ width, locale, colorScheme, geometry, full })
      if (process.env.EXPECT_CLIPPED === '1') {
        assert.equal(full, false, 'Baseline should reproduce clipping')
        console.log('RED reproduced missing focus edge', JSON.stringify(geometry))
        break
      }
      assert(full, `Clipped focus outline: ${stem} ${JSON.stringify(geometry)}`)
      await search.fill('openai')
      await expect(page.getByText('OpenAI API', { exact: true })).toBeVisible()
      await expect(page.getByRole('button', { name: locale === 'zh' ? '添加服务商' : 'Add Provider', exact: true })).toBeVisible()
      console.log(`PASS focus ${stem}`)
    } finally { await context.close() }
    if (process.env.EXPECT_CLIPPED === '1') break
  }
} finally {
  await writeFile(`${evidence}/focus-receipt.json`, JSON.stringify(receipt, null, 2))
  await browser.close()
}
