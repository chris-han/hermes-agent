/** One native journey driver: explicit source lane or exact extracted-package lane. */
import { _electron, chromium, expect } from '../../node_modules/@playwright/test/index.mjs'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createHash } from 'node:crypto'
import { spawnSync } from 'node:child_process'

assert.equal(process.platform, 'win32')
const source = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..')
const root = path.resolve(process.argv[2])
const executable = process.argv[3]
const packagedMode = process.argv[7] === 'package'
let python = process.argv[4]
const payload = packagedMode ? path.join(path.dirname(executable), 'resources/agent-payload') : null
const payloadManifest = packagedMode ? JSON.parse(fs.readFileSync(path.join(payload, 'manifest.json'), 'utf8')) : null
const runtimeSource = packagedMode ? path.join(payload, payloadManifest.runtime.repoDir) : source
if (packagedMode) {
  python = path.join(payload, payloadManifest.runtime.storePython)
  assert.ok(fs.existsSync(python), 'actual shipped base interpreter is required')
  assert.ok(process.argv[8] && fs.existsSync(process.argv[8]), 'exact signed MSIX binding required')
}
assert.ok(root && executable && python, 'explicit disposable state, Electron and interpreter required')
const marker = path.join(root, '.owner.json')
if (fs.existsSync(root)) assert.equal(JSON.parse(fs.readFileSync(marker, 'utf8')).owner, 'windows-cloud-client-r1')
else {
  fs.mkdirSync(root, { recursive: true })
  fs.writeFileSync(marker, JSON.stringify({ owner: 'windows-cloud-client-r1' }))
}
const home = path.join(root, 'hermes-home'),
  profile = path.join(root, 'profile')
for (const p of [
  home,
  profile,
  path.join(profile, 'AppData/Roaming'),
  path.join(profile, 'AppData/Local'),
  path.join(root, 'tmp')
])
  fs.mkdirSync(p, { recursive: true })
const config = path.join(home, 'config.yaml')
if (!fs.existsSync(config))
  fs.writeFileSync(
    config,
    JSON.stringify({
      model: { provider: 'semantier-cloud', base_url: 'http://127.0.0.1:55474/v1', default: '' },
      providers: {
        'semantier-cloud': { base_url: 'http://127.0.0.1:55474/v1', portal_origin: 'http://127.0.0.1:3358' }
      },
      agent: { max_turns: 1 },
      memory: { enabled: false }
    })
  )
const env = Object.fromEntries(
  Object.entries(process.env).filter(
    ([k]) =>
      !k.startsWith('HERMES_') &&
      !/(?:_API_KEY|_TOKEN|_SECRET|_PASSWORD|_CREDENTIALS|_ACCESS_KEY|_PRIVATE_KEY)$/.test(k)
  )
)
Object.assign(env, {
  PYTHONPATH: source,
  HERMES_HOME: home,
  HOME: profile,
  USERPROFILE: profile,
  APPDATA: path.join(profile, 'AppData/Roaming'),
  LOCALAPPDATA: path.join(profile, 'AppData/Local'),
  TEMP: path.join(root, 'tmp'),
  TMP: path.join(root, 'tmp'),
  HERMES_DESKTOP_USER_DATA_DIR: path.join(root, process.argv[6] || 'electron-state'),
  HERMES_DESKTOP_IGNORE_EXISTING: '1',
  HERMES_DESKTOP_ISOLATED_BACKEND: '1',
  HERMES_DESKTOP_HERMES_ROOT: source,
  HERMES_DESKTOP_PYTHON: python,
  HERMES_DESKTOP_APP_NAME: 'WindowsCloudSourceR1',
  HERMES_DESKTOP_SKIP_QUIT_CONFIRM: '1',
  HERMES_DESKTOP_SKIP_PROTOCOL_REGISTRATION: '1',
  HERMES_DESKTOP_DEV_SERVER: ''
})
if (packagedMode) {
  for (const key of ['PYTHONPATH', 'HERMES_DESKTOP_HERMES_ROOT', 'HERMES_DESKTOP_PYTHON']) delete env[key]
  env.USERPROFILE = process.env.USERPROFILE
}
if (process.argv[5] === 'stop') {
  const cli = packagedMode
    ? [
        '-I',
        '-c',
        "import sys,runpy;sys.path[:0]=[sys.argv.pop(1),sys.argv.pop(1)];sys.argv[0]='hermes';runpy.run_module('hermes_cli.main',run_name='__main__')",
        runtimeSource,
        path.join(payload, payloadManifest.runtime.sitePackages),
        'gateway',
        'stop'
      ]
    : ['-m', 'hermes_cli.main', 'gateway', 'stop']
  const result = spawnSync(python, cli, {
    env,
    cwd: source,
    encoding: 'utf8',
    timeout: 45000
  })
  console.log(
    JSON.stringify({
      owned_gateway_stop_exit: result.status,
      stdout: result.stdout?.slice(-1000),
      stderr: result.stderr?.slice(-1000)
    })
  )
  process.exit(result.status ?? 1)
}
const freshMode = process.argv[5] === 'fresh-connect'
const scopeMode = process.argv[5] === 'scope-connect'
let profileAsset =
  !packagedMode && scopeMode
    ? fs
        .readdirSync(path.join(source, 'apps/desktop/dist/assets'))
        .find(
          name =>
            name.startsWith('profile-') &&
            name.endsWith('.js') &&
            fs.readFileSync(path.join(source, 'apps/desktop/dist/assets', name), 'utf8').includes('as selectProfile')
        )
    : null
const selectProfile = async name => {
  await page.evaluate(
    async ({ asset, name }) => {
      const profiles = await import(new URL('./assets/' + asset, location.href).href)
      await profiles.refreshProfiles()
      profiles.selectProfile(name)
    },
    { asset: profileAsset, name }
  )
  await expect
    .poll(
      () =>
        page.evaluate(
          async asset => (await import(new URL('./assets/' + asset, location.href).href)).$activeGatewayProfile.get(),
          profileAsset
        ),
      { timeout: 60000 }
    )
    .toBe(name)
  await page.evaluate(async name => {
    await window.hermesDesktop.getConnection(name)
  }, name)
}

const run = Date.now()
const hash = p => createHash('sha256').update(fs.readFileSync(p)).digest('hex')
const receipt = {
  produced_at: new Date().toISOString(),
  scope: packagedMode ? 'EXACT_EXTRACTED_SIGNED_PACKAGE' : 'ACTUAL_WINDOWS_SOURCE_ELECTRON',
  package_acceptance: false,
  registered_install_acceptance: false,
  runtime_coverage: packagedMode ? 'extracted-runtime-only' : 'source-only',
  source_python_sha256: hash(path.join(runtimeSource, 'hermes_cli/cloud_device_client.py')),
  renderer_sha256: packagedMode ? null : hash(path.join(source, 'apps/desktop/dist/index.html')),
  ...(packagedMode
    ? {
        signed_msix_path: process.argv[8],
        signed_msix_sha256: hash(process.argv[8]),
        payload_source_commit: payloadManifest.ref
      }
    : {}),
  stage: 'boot'
}
let app, page, browser, browserContext
const nativeInstalled = () =>
  page.evaluate(async () => {
    const result = await window.hermesDesktop.api({
      path: '/api/providers/oauth',
      profile: 'default',
      connectionId: 'local'
    })
    return result.providers.find(provider => provider.id === 'semantier-cloud')?.status.logged_in === true
  })
try {
  app = await _electron.launch({
    executablePath: executable,
    args: packagedMode ? ['--no-sandbox'] : [path.join(source, 'apps/desktop'), '--no-sandbox'],
    cwd: path.join(source, 'apps/desktop'),
    env,
    timeout: 90000
  })
  if (packagedMode) {
    receipt.application = await app.evaluate(async ({ app }) => {
      const fs = await import('node:fs'),
        path = await import('node:path'),
        crypto = await import('node:crypto')
      const assets = path.join(app.getAppPath(), 'dist/assets')
      return {
        isPackaged: app.isPackaged,
        version: app.getVersion(),
        resources: process.resourcesPath,
        appPath: app.getAppPath(),
        renderer_sha256: crypto
          .createHash('sha256')
          .update(fs.readFileSync(path.join(app.getAppPath(), 'dist/index.html')))
          .digest('hex'),
        profileAsset: fs
          .readdirSync(assets)
          .find(
            name =>
              name.startsWith('profile-') &&
              name.endsWith('.js') &&
              fs.readFileSync(path.join(assets, name), 'utf8').includes('as selectProfile')
          )
      }
    })
    assert.equal(receipt.application.isPackaged, true)
    assert.match(process.argv[9] || '', /^[a-f0-9]{40}$/, 'exact artifact source commit required')
    assert.equal(payloadManifest.ref, process.argv[9])
    receipt.production_modules = {}
    for (const relative of [
      'hermes_cli/cloud_device_client.py',
      'hermes_cli/web_routers/oauth.py',
      'agent/semantier_gateway.py',
      'hermes_cli/inventory.py',
      'tui_gateway/contracts/semantier_billing.py'
    ]) {
      const actual = hash(path.join(runtimeSource, relative))
      assert.equal(actual, hash(path.join(source, relative)), 'shipped source mismatch: ' + relative)
      receipt.production_modules[relative] = actual
    }
    assert.equal(
      path.resolve(receipt.application.resources),
      path.resolve(path.join(path.dirname(executable), 'resources'))
    )
    receipt.renderer_sha256 = receipt.application.renderer_sha256
    profileAsset = receipt.application.profileAsset
  }
  await app.evaluate(({ shell }) => {
    globalThis.__cloudSourceUrls = []
    shell.openExternal = async url => {
      globalThis.__cloudSourceUrls.push(url)
    }
  })
  page = await app.firstWindow()
  receipt.transport = []
  page.on('websocket', socket => {
    const methods = new Map()
    socket.on('framesent', ({ payload }) => {
      try {
        const frame = JSON.parse(String(payload))
        if (frame.method) methods.set(frame.id, frame.method)
      } catch {}
    })
    socket.on('framereceived', ({ payload }) => {
      try {
        const frame = JSON.parse(String(payload))
        const method = methods.get(frame.id)
        if (method && receipt.transport.length < 200)
          receipt.transport.push({ method, error_code: frame.error?.code, success: !frame.error })
      } catch {}
    })
  })
  page.setDefaultTimeout(60000)
  await page.waitForLoadState('domcontentloaded')
  await page.waitForFunction(
    () =>
      [...document.querySelectorAll('button')].some(b => /Semantier Cloud/.test(b.textContent)) ||
      document.querySelector('[contenteditable="true"]'),
    { timeout: 90000 }
  )
  await expect
    .poll(
      () =>
        page.locator('[data-glass-opaque][class*="--z-connecting"]').evaluateAll(nodes =>
          nodes.every(node => {
            const style = getComputedStyle(node)
            return style.pointerEvents === 'none' || style.opacity === '0' || style.display === 'none'
          })
        ),
      { timeout: 180000 }
    )
    .toBe(true)
  await expect(page.getByText('Loading Semantier settings', { exact: true })).toBeHidden({ timeout: 180000 })
  await expect
    .poll(
      async () =>
        (await page.getByRole('button', { name: /Semantier Cloud/ }).count()) > 0 ||
        (await page.locator('[data-glass-opaque][class*="--z-onboarding"]').count()) === 0,
      { timeout: 180000 }
    )
    .toBe(true)
  receipt.stage = 'provider_visible'
  if (scopeMode) await selectProfile('default')
  if ((process.argv[5] || '').startsWith('settings') || ['scope-connect', 'visual'].includes(process.argv[5])) {
    await page.keyboard.press('Control+,')
    await page.getByRole('button', { name: 'Providers', exact: true }).click()
    await page
      .getByRole('button', { name: /Semantier Cloud/ })
      .first()
      .waitFor({ timeout: 60000 })
    receipt.stage = 'settings_provider_visible'
    await page.screenshot({ path: path.join(root, `settings-${run}.png`) })
  }
  if (['connect', 'settings-connect', 'scope-connect', 'fresh-connect'].includes(process.argv[5])) {
    const fixtureName = process.argv[10] || (freshMode ? 'fresh' : 'cloud')
    assert.match(fixtureName, /^[a-z0-9-]+$/)
    const fixture = path.resolve(source, '../.tmp/windows-cloud-r1', fixtureName)
    const identity = JSON.parse(fs.readFileSync(path.join(fixture, 'identity.json'), 'utf8'))
    const sends = path.resolve(source, '../.tmp/windows-cloud-r1/proxy/controlled-sends.jsonl')
    const sendCount = () =>
      fs.existsSync(sends) ? fs.readFileSync(sends, 'utf8').trim().split('\n').filter(Boolean).length : 0
    const before = sendCount()
    if (scopeMode) {
      assert.ok(profileAsset)
      await page.evaluate(async () => {
        await window.hermesDesktop.getConnection('beta')
      })
    }
    if (await page.getByRole('button', { name: /Semantier Cloud/ }).count()) {
      await page
        .getByRole('button', { name: /Semantier Cloud/ })
        .first()
        .click()
      await expect.poll(() => app.evaluate(() => globalThis.__cloudSourceUrls.at(-1)), { timeout: 20000 }).toBeTruthy()
      const verification = await app.evaluate(() => globalThis.__cloudSourceUrls.at(-1))
      assert.equal(new URL(verification).origin, 'http://127.0.0.1:3358')
      assert.equal(new URL(verification).searchParams.has('proof'), false)
      browser = await chromium.connectOverCDP('http://127.0.0.1:9238')
      browserContext = await browser.newContext({ viewport: { width: 1440, height: 1000 } })
      const portal = await browserContext.newPage()
      if (freshMode && !identity.registered) {
        await portal.goto(verification + '&lang=en')
        await portal.getByRole('button', { name: 'Create account', exact: true }).first().click()
        const signupDialog = portal.getByRole('dialog')
        await expect
          .poll(
            async () =>
              (await portal.locator('input[name="password-confirm"]').count()) > 0 ||
              (await signupDialog.getByRole('button', { name: 'Create account', exact: true }).count()) > 0
          )
          .toBe(true)
        if (!(await portal.locator('input[name="password-confirm"]').count()))
          await signupDialog.getByRole('button', { name: 'Create account', exact: true }).click()
        await portal.locator('input[name="username"]').fill(identity.login)
        await portal.locator('input[name="password"]').fill(identity.password)
        await portal.locator('input[name="password-confirm"]').fill(identity.password)
        await portal.getByRole('button', { name: 'Register and continue', exact: true }).click()
        await expect
          .poll(
            async () =>
              (await browserContext.request.get('http://127.0.0.1:3358/api/billing/commerce/accounts')).status(),
            { timeout: 60000 }
          )
          .toBe(200)
        identity.registered = true
        fs.writeFileSync(path.join(fixture, 'identity.json'), JSON.stringify(identity))
        receipt.real_ui_signup = true
      } else {
        const login = await browserContext.request.post('http://127.0.0.1:3358/auth/password/login', {
          data: { login: identity.login, password: identity.password }
        })
        assert.equal(login.status(), 200, 'real password login')
      }
      if (freshMode) {
        const accounts = await browserContext.request.get('http://127.0.0.1:3358/api/billing/commerce/accounts')
        assert.equal(accounts.status(), 200)
        const rows = (await accounts.json()).data
        assert.equal(rows.length, 1, 'new personal account is unambiguous')
        fs.writeFileSync(path.join(fixture, 'account.json'), JSON.stringify(rows[0]))
      }
      const fixtureAccount = JSON.parse(fs.readFileSync(path.join(fixture, 'account.json'), 'utf8'))
      await portal.goto(verification + '&lang=en&account_id=' + encodeURIComponent(fixtureAccount.account_id))
      const accountSelector = portal.getByRole('combobox', { name: 'Select billing account' })
      if (await accountSelector.count()) await accountSelector.selectOption(fixtureAccount.account_id)
      const devicePath = '/api/cloud-provider/device/' + new URL(verification).searchParams.get('intent_ref')
      const deviceRead = await browserContext.request.get('http://127.0.0.1:3358' + devicePath)
      assert.equal(deviceRead.status(), 200)
      const view = await deviceRead.json()
      receipt.recovery_state = view.status
      assert.ok(['pending', 'approved', 'claimed', 'installed'].includes(view.status))
      receipt.purchase_created = false
      const originalConfigHash = hash(config)
      const betaConfig = path.join(home, 'profiles/beta/config.yaml')
      let betaConfigHash
      if (scopeMode) {
        await selectProfile('beta')
        betaConfigHash = hash(betaConfig)
      }
      if (view.status === 'pending') {
        await portal.getByRole('button', { name: 'Approve device', exact: true }).waitFor({ timeout: 60000 })
        const choose = portal.getByRole('link', { name: 'Choose a package', exact: true })
        if (scopeMode || process.argv[5] === 'settings-connect')
          await expect.poll(() => choose.count(), { timeout: 15000 }).toBe(0)
        if (await choose.count()) {
          await choose.click()
          await portal
            .getByRole('article', { name: 'Professional · Monthly', exact: true })
            .getByRole('link', { name: 'Choose monthly', exact: true })
            .click()
          await portal.getByRole('radio', { name: /WeChat Pay/ }).check()
          await portal.getByRole('checkbox', { name: /I understand the local test conditions/ }).check()
          await portal.getByRole('button', { name: /Confirm order & continue/ }).click()
          await portal.getByRole('button', { name: 'Simulate payment success', exact: true }).click()
          await portal.getByRole('link', { name: /Return to device connection/ }).click()
          receipt.purchase_created = true
        }
        if (freshMode) {
          fs.writeFileSync(
            path.join(fixture, 'browser-paid.json'),
            JSON.stringify({
              produced_at: new Date().toISOString(),
              account_id: fixtureAccount.account_id,
              real_ui_signup: receipt.real_ui_signup ?? false,
              purchase_created: receipt.purchase_created
            })
          )
          await expect.poll(() => fs.existsSync(path.join(fixture, 'route-ready.json')), { timeout: 240000 }).toBe(true)
        }
        await portal.getByRole('button', { name: 'Approve device', exact: true }).click()
      }
      if (scopeMode) {
        await new Promise(resolve => setTimeout(resolve, 11000))
        assert.equal(hash(config), originalConfigHash, 'inactive original profile must defer installation')
        assert.equal(hash(betaConfig), betaConfigHash, 'foreign profile must remain unchanged')
        assert.equal(fs.existsSync(path.join(home, 'profiles/beta/.cloud-device.bin')), false)
        receipt.inactive_profile_deferred = true
        receipt.foreign_profile_untouched = true
        await selectProfile('default')
      }
      if (process.argv[5] === 'settings-connect' || scopeMode) {
        await expect.poll(nativeInstalled, { timeout: 90000 }).toBe(true)
        receipt.native_installation_confirmed = true
      } else {
        await page.getByRole('button', { name: /Begin/i }).waitFor({ timeout: 90000 })
      }
    } else {
      if (freshMode) {
        assert.equal(identity.registered, true, 'fresh signup must already have an actual browser receipt')
        assert.ok(fs.existsSync(path.join(fixture, 'browser-paid.json')))
        assert.equal(await nativeInstalled(), true, 'restart recovery requires actual native installation')
        receipt.fresh_journey_recovery = true
        receipt.prior_browser_stage_sha256 = hash(path.join(fixture, 'browser-paid.json'))
      }
      receipt.recovery_state = 'installed_restart'
      receipt.purchase_created = false
    }
    receipt.stage = 'installed_model_confirmation'
    assert.equal(sendCount(), before, 'auth/focus/poll/catalog must not dispatch inference')
    await page.screenshot({ path: path.join(root, `connected-${run}.png`) })
    receipt.zero_inference_before_send = true
    receipt.key_copies = 0
    const begin = page.getByRole('button', { name: /Begin/i })
    if (await begin.count()) await begin.click()
    if (process.argv[5] === 'settings-connect' || scopeMode) {
      receipt.stage = scopeMode ? 'original_profile_return_installed' : 'settings_reauthorization_complete'
      assert.equal(sendCount(), before, 'Settings reauthorization must not infer')
    } else {
      const newSession = page.getByRole('button', { name: /^New session/ }).first()
      if (await newSession.count()) await newSession.click()
      const composer = page.locator('[contenteditable="true"]').first()
      await composer.waitFor({ timeout: 60000 })
      await composer.fill('Controlled Windows device source acceptance. Reply briefly.')
      const privacy = page.getByRole('button', { name: 'No thanks', exact: true })
      if (await privacy.count()) await privacy.click()
      await page.getByRole('button', { name: /^Send$/i }).click()
      await page.getByText('Device controlled response.', { exact: false }).first().waitFor({ timeout: 60000 })
      receipt.stage = 'explicit_controlled_chat'
      receipt.controlled_sends = sendCount() - before
      assert.ok(receipt.controlled_sends >= 1)
      await page.screenshot({ path: path.join(root, `chat-${run}.png`) })
    }
  }
  if (process.argv[5] === 'visual') {
    const sends = path.resolve(source, '../.tmp/windows-cloud-r1/proxy/controlled-sends.jsonl')
    const count = () => fs.readFileSync(sends, 'utf8').trim().split('\n').filter(Boolean).length
    const before = count()
    const win = await app.browserWindow(page)
    receipt.visual = []
    for (const mode of ['light', 'dark']) {
      await page.getByRole('button', { name: 'Appearance', exact: true }).click()
      await page.getByRole('button', { name: 'Theme', exact: true }).click()
      await page.getByRole('button', { name: mode === 'light' ? 'Light' : 'Dark', exact: true }).click()
      await expect.poll(() => page.evaluate(() => document.documentElement.dataset.hermesMode)).toBe(mode)
      await page.getByRole('button', { name: 'Providers', exact: true }).click()
      for (const [width, height] of [
        [1440, 1000],
        [1024, 760]
      ]) {
        await win.evaluate((window, size) => window.setSize(...size), [width, height])
        await expect(page.getByRole('button', { name: /Semantier Cloud/ }).first()).toBeVisible()
        await page.keyboard.press('Tab')
        const focus = await page.evaluate(() => ({
          tag: document.activeElement?.tagName,
          label:
            document.activeElement?.getAttribute('aria-label') || document.activeElement?.textContent?.slice(0, 100)
        }))
        assert.notEqual(focus.tag, 'BODY', 'keyboard focus reaches a control')
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true)
        const screenshot = `visual-en-${mode}-${width}-${run}.png`
        await page.screenshot({ path: path.join(root, screenshot) })
        receipt.visual.push({ locale: 'en', mode, width, height, screenshot, focus })
      }
    }
    await page.getByRole('button', { name: 'Billing', exact: true }).click()
    await expect(page.getByText(/CNY|¥/).first()).toBeVisible({ timeout: 60000 })
    receipt.billing_text = (await page.locator('body').innerText()).slice(-6500)
    await page.screenshot({ path: path.join(root, `billing-en-${run}.png`) })
    await page.getByRole('button', { name: 'Appearance', exact: true }).click()
    await page.getByRole('button', { name: 'General', exact: true }).click()
    await page.getByRole('button', { name: 'Switch language', exact: true }).click()
    await page.getByRole('combobox').fill('简体中文')
    await page.keyboard.press('ArrowDown')
    await page.keyboard.press('Enter')
    await page.getByRole('button', { name: '提供方', exact: true }).click()
    await expect(page.getByRole('button', { name: /Semantier Cloud/ }).first()).toBeVisible()
    await page.screenshot({ path: path.join(root, `visual-zh-dark-1024-${run}.png`) })
    receipt.visual.push({ locale: 'zh', mode: 'dark', width: 1024, height: 760 })
    await page.getByRole('button', { name: '外观', exact: true }).click()
    await page.getByRole('button', { name: '常规', exact: true }).click()
    await page.getByRole('button', { name: '切换语言', exact: true }).click()
    await page.getByRole('combobox').fill('English')
    await page.keyboard.press('ArrowDown')
    await page.keyboard.press('Enter')
    await page.getByRole('button', { name: 'Theme', exact: true }).click()
    await page.getByRole('button', { name: 'Light', exact: true }).click()
    assert.equal(count(), before, 'visual, account Billing and locale/theme refresh must not infer')
    receipt.zero_inference_visual_billing_refresh = true
    receipt.stage = 'native_visual_billing_accessibility'
  }
  receipt.package_acceptance = packagedMode
  receipt.title = await page.title()
  receipt.text = (await page.locator('body').innerText()).slice(0, 8000)
  await page.screenshot({ path: path.join(root, `provider-visible-${run}.png`) })
  fs.writeFileSync(path.join(root, `source-receipt-${run}.json`), JSON.stringify(receipt, null, 2))
  console.log(JSON.stringify({ stage: receipt.stage, package_acceptance: receipt.package_acceptance }))
} catch (error) {
  receipt.error = String(error).slice(0, 1000)
  const portalFailure = browserContext?.pages().at(-1)
  if (portalFailure) {
    receipt.portal_text = (
      await portalFailure
        .locator('body')
        .innerText()
        .catch(() => '')
    ).slice(0, 5000)
    await portalFailure.screenshot({ path: path.join(root, `portal-failure-${run}.png`) }).catch(() => {})
  }
  if (page) {
    receipt.text = (
      await page
        .locator('body')
        .innerText()
        .catch(() => '')
    ).slice(0, 8000)
    await page.screenshot({ path: path.join(root, `source-failure-${run}.png`) }).catch(() => {})
  }
  fs.writeFileSync(path.join(root, `source-receipt-${run}.json`), JSON.stringify(receipt, null, 2))
  throw error
} finally {
  await browserContext?.close().catch(() => {})
  await browser?.close().catch(() => {})
  await app?.close().catch(() => {})
}
