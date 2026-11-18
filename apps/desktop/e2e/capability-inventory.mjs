import { chromium } from 'playwright'
import { expect } from '@playwright/test'
import { mkdir } from 'node:fs/promises'
const out = '/tmp/desktop-capabilities-visuals'
await mkdir(out, { recursive: true })
const browser = await chromium.launch({ channel: 'chromium', headless: true })
console.log('Browser', browser.version(), 'isolated Linux headless Chromium; fixture backend, production renderer')
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
const errors = []
page.on('pageerror', e => errors.push(e.message))
try {
  if (!process.env.CAPABILITIES_PARITY_ONLY) {
    await page.goto('http://127.0.0.1:5189/capability-inventory.html?count=100&locale=en')
    await page.getByRole('heading', { name: 'Learned Skills', exact: true }).waitFor()
    console.log('PASS entry learned grouping')
    await page.getByRole('button', { name: 'Cards', exact: true }).click()
    await expect(page.locator('[data-capability-inventory]')).toHaveAttribute('data-view', 'cards')
    await page.getByRole('button', { name: 'default-skill-099', exact: true }).click()
    await expect(page.getByRole('heading', { name: 'default-skill-099', exact: true })).toBeVisible()
    await page.getByRole('button', { name: 'List', exact: true }).click()
    await expect(page.getByRole('heading', { name: 'default-skill-099', exact: true })).toBeVisible()
    console.log('PASS selection across views')
    const toggle = page.getByRole('switch', { name: 'default-skill-099', exact: true })
    await page.evaluate(() => globalThis.window.fixture.rejectToggle(true))
    await toggle.click()
    await expect(toggle).toHaveAttribute('aria-checked', 'true')
    console.log('PASS rejected mutation rollback')
    await page.evaluate(() => globalThis.window.fixture.rejectToggle(false))
    await page.getByRole('button', { name: 'Edit', exact: true }).click()
    await page.locator('.cm-content[contenteditable="true"]').fill('# Updated learned content')
    await page.getByRole('button', { name: 'Save', exact: true }).click()
    await expect
      .poll(() =>
        page.evaluate(
          () =>
            globalThis.window.fixture.calls.filter(c => c.path.startsWith('/api/learning/node') && c.method === 'PUT').at(-1)?.body
              ?.content
        )
      )
      .toBe('# Updated learned content')
    await page.getByRole('button', { name: 'Archive', exact: true }).click()
    await page.getByRole('dialog').getByRole('button', { name: 'Archive', exact: true }).click()
    await expect
      .poll(() => page.evaluate(() => globalThis.window.fixture.calls.filter(c => c.method === 'DELETE').at(-1)?.profile))
      .toBe('default')
    console.log('PASS learned edit/save/archive through existing scoped APIs')
    await page.getByRole('button', { name: 'Switch fixture profile', exact: true }).click()
    await page.getByRole('heading', { name: 'other-skill-099', exact: true }).waitFor()
    await expect(page.getByText('Exact owner: default', { exact: false })).toHaveCount(0)
    await page.getByRole('switch', { name: 'other-skill-099', exact: true }).click()
    await expect
      .poll(() =>
        page.evaluate(() => globalThis.window.fixture.calls.filter(c => c.path === '/api/skills/toggle').at(-1)?.profile)
      )
      .toBe('other')
    await page.getByRole('button', { name: 'Switch fixture connection', exact: true }).click()
    await page.getByRole('heading', { name: 'other-skill-099', exact: true }).waitFor()
    await expect
      .poll(() => page.evaluate(() => globalThis.window.fixture.calls.filter(c => c.path === '/api/skills').at(-1)?.connectionId))
      .toBe('remote-fixture')
    console.log('PASS profile and connection routing')
    await page.getByRole('button', { name: 'Skill Marketplace', exact: true }).click()
    await expect(page.getByText('Catalog research', { exact: true }).first()).toBeVisible()
    await page.evaluate(() => globalThis.window.fixture.offline(true))
    await page.reload()
    await page.getByRole('heading', { name: 'Learned Skills', exact: true }).waitFor()
    await page.getByRole('button', { name: 'Skill Marketplace', exact: true }).click()
    await expect(page.getByRole('alert')).toContainText('Catalog unavailable')
    await page.evaluate(() => globalThis.window.fixture.offline(false))
    await page.getByRole('button', { name: 'Retry', exact: true }).click()
    await expect(page.getByText('Catalog research', { exact: true }).first()).toBeVisible()
    await expect(page.locator('iframe')).toHaveCount(0)
    console.log('PASS native catalog failure and retry')
    for (const count of [0, 1, 14, 100, 500]) {
      const start = performance.now()
      await page.goto(`http://127.0.0.1:5189/capability-inventory.html?count=${count}&locale=en`)
      if (count) await page.getByRole('heading', { name: 'Learned Skills', exact: true }).waitFor()
      else
        await page
          .getByText(/No skills/i)
          .first()
          .waitFor()
      for (const view of ['Cards', 'List']) {
        await page.getByRole('button', { name: view, exact: true }).click()
        await expect(page.locator('[data-capability-inventory]')).toHaveAttribute('data-view', view.toLowerCase())
      }
      if (count === 500) {
        const filterStart = performance.now()
        await page.getByRole('textbox').fill('skill-499')
        await expect(page.getByRole('heading', { name: 'default-skill-499', exact: true })).toBeVisible()
        console.log(`PASS 500-entry search/filter-to-detail ${Math.round(performance.now() - filterStart)}ms`)
        await page.getByRole('textbox').fill('')
      }
      console.log(`PASS inventory ${count} entries; navigation-to-ready ${Math.round(performance.now() - start)}ms`)
    }
    for (const locale of ['en', 'zh'])
      for (const dark of [false, true])
        for (const width of [1440, 640]) {
          await page.setViewportSize({ width, height: 1000 })
          await page.addInitScript(
            dark => localStorage.setItem('hermes-desktop-mode-v1', dark ? 'dark' : 'light'),
            dark
          )
          await page.goto(`http://127.0.0.1:5189/capability-inventory.html?count=14&locale=${locale}`)
          await page
            .getByRole('heading', { name: locale === 'en' ? 'Learned Skills' : '习得技能', exact: true })
            .waitFor()
          for (const cards of [false, true]) {
            await page
              .getByRole('button', {
                name: locale === 'en' ? (cards ? 'Cards' : 'List') : cards ? '卡片' : '列表',
                exact: true
              })
              .click()
            await expect(page.locator('[data-capability-inventory]')).toHaveAttribute(
              'data-view',
              cards ? 'cards' : 'list'
            )
            await page.screenshot({
              path: `${out}/${locale}-${dark ? 'dark' : 'light'}-${width}-${cards ? 'cards' : 'list'}.png`
            })
            const overflow = await page.evaluate(() => globalThis.document.documentElement.scrollWidth > globalThis.innerWidth)
            expect(overflow).toBe(false)
          }
          if (width === 640) {
            await page.getByRole('button', { name: 'default-skill-012', exact: true }).click()
            const back = page.getByRole('button', {
              name: locale === 'en' ? 'Back to inventory' : '返回列表',
              exact: true
            })
            await expect(back).toBeVisible()
            await back.click()
          }
          console.log(`PASS visuals ${locale} ${dark ? 'dark' : 'light'} ${width} List/Cards and compact return`)
        }
  }
  await page.setViewportSize({ width: 1440, height: 1000 })
  await page.goto('http://127.0.0.1:5189/capability-inventory.html?count=500&locale=en')
  await page.getByRole('heading', { name: 'Learned Skills', exact: true }).waitFor()
  await page.getByRole('button', { name: 'Tools', exact: false }).click()
  await expect(page.getByRole('switch', { name: 'Turn Web Search toolset off' })).toBeVisible()
  await page.getByRole('button', { name: 'Cards', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Web Search', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Connectors fixture', exact: true }).click()
  for (const view of ['List', 'Cards']) {
    await page.getByRole('button', { name: view, exact: true }).click()
    await expect(page.locator('[data-slot="connector-row-card"]')).toHaveCount(500)
    await page.screenshot({ path: `${out}/connectors-500-${view}.png` })
  }
  await page.getByRole('textbox').fill('connector-499')
  await expect(page.locator('[data-slot="connector-row-card"]')).toHaveCount(1)
  await page.locator('[data-connector="connector-499"] button').first().click()
  await expect.poll(() => page.evaluate(() => globalThis.window.fixture.calls.at(-1)?.openConnector)).toBe('connector-499')
  console.log('PASS Tools views and 500-connector List/Cards, search and detail dispatch')
  await page.getByRole('button', { name: 'Connectors fixture', exact: true }).click()
  await page.getByRole('button', { name: 'Plugin catalog fixture', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'fixture-plugin-499', exact: true })).toBeAttached()
  await page.getByRole('textbox').fill('fixture-plugin-499')
  await expect(page.getByRole('heading', { name: /fixture-plugin-/ })).toHaveCount(1)
  await page.getByRole('button', { name: 'Install', exact: true }).click()
  console.log('PASS 500 native plugin catalog entries, search and install-dialog handoff')
  const dpi = await browser.newPage({ viewport: { width: 720, height: 600 }, deviceScaleFactor: 2 })
  try {
    await dpi.goto('http://127.0.0.1:5189/capability-inventory.html?count=14&locale=zh')
    await dpi.getByRole('heading', { name: '习得技能', exact: true }).waitFor()
    await dpi.getByRole('button', { name: '卡片', exact: true }).click()
    await dpi.keyboard.press('Tab')
    expect(await dpi.evaluate(() => globalThis.document.activeElement?.tagName)).not.toBe('BODY')
    await dpi.screenshot({ path: `${out}/zh-200-percent-dpi.png` })
    expect(await dpi.evaluate(() => globalThis.document.documentElement.scrollWidth > globalThis.innerWidth)).toBe(false)
    console.log('PASS 200% device scale, compact layout and keyboard focus')
  } finally {
    await dpi.close()
  }
  expect(errors).toEqual([])
} finally {
  await browser.close()
}
