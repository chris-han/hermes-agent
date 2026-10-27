// Render production ThemeProvider, AppearanceSettings and shared primitives.
// Native IPC/backend/model dispatch is outside this presentation fixture.
import assert from 'node:assert/strict'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { chromium } from 'playwright'
import { createServer } from 'vite'

const endpoint = process.env.SEMANTIER_THEME_CDP
const localRenderer = process.env.SEMANTIER_THEME_RENDERER === 'chromium'
assert(
  endpoint || localRenderer,
  'Supply an owned Windows Chrome endpoint or explicitly select the portable Chromium lane'
)
const output = path.resolve(process.env.SEMANTIER_THEME_OUTPUT || 'test-results/semantier-theme')
await mkdir(output, { recursive: true })
const server = await createServer({
  server: { host: '127.0.0.1', port: 0 },
  plugins: [
    {
      name: 'semantier-theme-verification',
      configureServer(server) {
        server.middlewares.use('/__semantier-theme', async (_request, response) => {
          const entry = (await readFile('index.html', 'utf8')).replace(
            'src="/src/main.tsx"',
            'src="/e2e/semantier-theme.fixture.tsx"'
          )
          const html = await server.transformIndexHtml('/__semantier-theme', entry)
          response.setHeader('Content-Type', 'text/html')
          response.end(html)
        })
      }
    }
  ]
})
let browser, context
const results = []
try {
  await server.listen()
  browser = localRenderer ? await chromium.launch({ headless: true }) : await chromium.connectOverCDP(endpoint)
  context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    reducedMotion: 'reduce',
    colorScheme: 'light'
  })
  const page = await context.newPage()
  const errors = [],
    externalFonts = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('request', request => {
    if (/fonts\.(googleapis|gstatic)\.com/.test(request.url())) externalFonts.push(request.url())
  })
  const port = server.httpServer.address().port
  await page.goto('http://127.0.0.1:' + port + '/__semantier-theme')
  await page.getByRole('heading', { name: 'Your workspace, in focus.' }).waitFor()
  if (!localRenderer)
    assert.match(
      await page.evaluate(() => navigator.userAgent),
      /Windows/,
      'Rendering authority must be Windows Chrome'
    )
  assert.equal(await page.locator('html').getAttribute('data-hermes-theme'), 'semantier')
  await page.evaluate(async () => {
    await document.fonts.load('400 14px "Manrope"', 'Workspace')
    await document.fonts.load('500 14px "Manrope"', 'Continue')
    await document.fonts.load('600 24px "Manrope"', 'Semantier')
    await document.fonts.load('400 14px "Noto Sans SC"', '司序科技')
    await document.fonts.ready
  })
  for (const width of [1440, 760]) {
    await page.setViewportSize({ width, height: 1000 })
    for (const mode of ['light', 'dark']) {
      await page
        .getByRole('button', { name: mode === 'light' ? 'Light appearance' : 'Dark appearance', exact: true })
        .click()
      await page.waitForFunction(
        mode =>
          getComputedStyle(document.querySelector('section')).backgroundColor ===
          (mode === 'dark' ? 'rgb(0, 0, 0)' : 'rgb(235, 235, 236)'),
        mode,
        { timeout: 5000 }
      )
      await page.getByRole('textbox', { name: 'Message', exact: true }).focus()
      const values = await page.evaluate(() => {
        const rgb = css => {
          const canvas = document.createElement('canvas')
          canvas.width = canvas.height = 1
          const c = canvas.getContext('2d')
          c.fillStyle = css
          c.fillRect(0, 0, 1, 1)
          return [...c.getImageData(0, 0, 1, 1).data].slice(0, 3)
        }
        const lum = rgb =>
          rgb
            .map(v => v / 255)
            .map(v => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4))
            .reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i], 0)
        const ratio = (a, b) => {
          const x = lum(rgb(a)),
            y = lum(rgb(b))
          return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05)
        }
        const style = selector => getComputedStyle(document.querySelector(selector))
        const panel = style('[data-testid="content-panel"]')
        const quiet = style('[data-testid="quiet"]')
        const primary = style('[data-testid="primary"]')
        const danger = style('[data-testid="danger"]')
        const input = style('input[aria-label="Message"]')
        return {
          quiet: ratio(quiet.color, panel.backgroundColor),
          primary: ratio(primary.color, primary.backgroundColor),
          danger: ratio(danger.color, danger.backgroundColor),
          focus: ratio(input.outlineColor, panel.backgroundColor),
          outline: input.outlineWidth,
          radius: primary.borderRadius,
          overflow: document.documentElement.scrollWidth > innerWidth,
          font: panel.fontFamily,
          stage: style('section').backgroundColor,
          icon: document.querySelector('[data-testid="primary"] svg').getAttribute('viewBox'),
          fontsLoaded:
            document.fonts.check('400 14px "Manrope"', 'Workspace') &&
            document.fonts.check('400 14px "Noto Sans SC"', '司序科技')
        }
      })
      for (const key of ['quiet', 'primary', 'danger']) assert(values[key] >= 4.5, key + ' contrast: ' + values[key])
      assert(values.focus >= 3, 'Focus contrast')
      assert.equal(values.outline, '2px')
      assert.equal(values.radius, '3px')
      assert.equal(values.icon, '0 0 256 256')
      assert(values.fontsLoaded)
      assert(!values.overflow)
      if (mode === 'dark') assert.equal(values.stage, 'rgb(0, 0, 0)')
      await page.screenshot({ path: path.join(output, 'chat-' + mode + '-' + width + '.png') })
      await page.getByRole('button', { name: 'Open dialog', exact: true }).click()
      const dialog = page.getByRole('dialog')
      await dialog.waitFor()
      assert.deepEqual(
        await dialog.evaluate(element => {
          const css = getComputedStyle(element)
          return { shadow: css.boxShadow, radius: css.borderRadius, blur: css.backdropFilter }
        }),
        { shadow: 'none', radius: '8px', blur: 'none' }
      )
      await page.keyboard.press('Escape')
      results.push({ width, mode, ...values })
    }
  }
  await page.getByRole('button', { name: 'Appearance settings', exact: true }).click()
  const semantier = page.getByRole('button', { name: /^Semantier Asimov/ })
  await semantier.waitFor()
  assert.deepEqual(
    await page.locator('[data-slot="search-field"]').evaluate(element => ({
      opacity: getComputedStyle(element).opacity,
      hintOpacity: getComputedStyle(element.querySelector('input'), '::placeholder').opacity
    })),
    { opacity: '1', hintOpacity: '1' }
  )
  assert.equal(await page.getByRole('button', { name: /^Nous Alt/ }).count(), 0)
  await page.getByRole('button', { name: /^Classic Hermes Gold/ }).click()
  assert.equal(await page.locator('html').getAttribute('data-hermes-design'), null)
  await semantier.focus()
  await page.keyboard.press('Enter')
  assert.equal(await semantier.getAttribute('aria-pressed'), 'true')
  for (const mode of ['light', 'dark']) {
    await page
      .getByRole('button', { name: mode === 'light' ? 'Light appearance' : 'Dark appearance', exact: true })
      .click()
    await page.screenshot({ path: path.join(output, 'appearance-' + mode + '.png') })
  }
  await page.reload()
  await page.getByRole('heading', { name: 'Your workspace, in focus.' }).waitFor()
  assert.equal(await page.locator('html').getAttribute('data-hermes-theme'), 'semantier')
  await page.evaluate(() => {
    localStorage.setItem('hermes-desktop-theme-v2', 'nous-alt')
    localStorage.setItem('hermes-boot-background', '#08172b')
  })
  await page.reload()
  await page.getByRole('heading', { name: 'Your workspace, in focus.' }).waitFor()
  assert.equal(await page.evaluate(() => localStorage.getItem('hermes-desktop-theme-v2')), 'semantier')
  assert.equal(externalFonts.length, 0, 'Semantier fonts must work offline')
  assert.deepEqual(errors, [])
  const summary = {
    browser: await browser.version(),
    platform: localRenderer ? 'Linux isolated Chromium; Windows qualification NOT_RUN' : 'Windows Chrome CDP',
    results,
    settings: 'PASS',
    keyboard: 'PASS',
    reload: 'PASS',
    migration: 'PASS',
    offlineFonts: 'PASS',
    nativeInstaller: 'NOT_RUN'
  }
  await writeFile(path.join(output, 'results.json'), JSON.stringify(summary, null, 2))
  console.log(JSON.stringify(summary))
} finally {
  await context?.close()
  await browser?.close()
  await server.close()
}
