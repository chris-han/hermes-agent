import { execFileSync } from 'node:child_process'
import { X509Certificate } from 'node:crypto'
/** First real loop: native card -> real Runtime intent -> Workspace login in owned pane. */
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'

import { expect, test } from '@playwright/test'

import { writeEnvFile } from '../../../tests-js/scripts/mock-provider-config'

import { DesktopFixtureSession } from './fixture-session'
import { buildAppEnv, createSandbox } from './fixtures'

// eslint-disable-next-line no-empty-pattern -- Electron owns the renderer.
test('unconfigured card opens real Billing login in the embedded pane', async ({}, info) => {
  test.setTimeout(300_000)
  const origin = process.env.HERMES_E2E_BILLING_ORIGIN
  expect(origin, 'Explicit local Workspace origin required').toBeTruthy()
  expect(['localhost', '127.0.0.1']).toContain(new URL(origin!).hostname)
  const outage = process.env.HERMES_E2E_CATALOG_OUTAGE === '1'
  const noAccess = process.env.HERMES_E2E_EXPECT_NO_ACCESS === '1'
  expect(outage && noAccess).toBe(false)
  const docker = process.env.HERMES_E2E_DOCKER_CLI

  if (outage) { expect(docker, 'Explicit native Docker CLI required for owned network fault').toBeTruthy() }
  const caddy = 'semantier-candidate-20261112-caddy'
  const marker = '/data/.cloud-billing-catalog-outage'
  let faultActive = false

  const setCatalogFault = (enabled: boolean) => {
    execFileSync(docker!, ['exec', caddy, ...(enabled ? ['touch', marker] : ['rm', '-f', marker])], { stdio: 'pipe' })
    faultActive = enabled
  }

  const operationCount = (operation: 'start' | 'claim' | 'ack') => {
    const pattern = operation === 'start' ? '/api/cloud-provider/device/start' : '/' + operation + '"'

    return Number(execFileSync(docker!, ['exec', caddy, 'sh', '-c', `grep -F -c '${pattern}' /data/side-workspace-access.jsonl || :`], { encoding: 'utf8' }).trim())
  }

  const before = outage ? { start: operationCount('start'), claim: operationCount('claim'), ack: operationCount('ack') } : null
  const sandbox = createSandbox('billing-embedded')
  writeEnvFile(sandbox.hermesHome)
  fs.writeFileSync(sandbox.hermesHome + '/config.yaml', `language: en\ndisplay:\n  language: en\nbilling: null\nproviders:\n  semantier-cloud:\n    base_url: ${origin}/v1\n    portal_origin: ${origin}\n`)
  const ca = process.env.HERMES_E2E_BILLING_CA
  const certificate = process.env.HERMES_E2E_BILLING_CERTIFICATE

  if (new URL(origin!).protocol === 'https:') {
    expect(ca).toBeTruthy()
    expect(certificate).toBeTruthy()
    fs.appendFileSync(sandbox.hermesHome + '/config.yaml', '    http_version: "3"\n    ssl_ca_cert: ' + JSON.stringify(ca) + '\n')
  }

  const env = buildAppEnv(sandbox, { HERMES_DESKTOP_DEV_SERVER: process.env.HERMES_DESKTOP_DEV_SERVER! })
  const fixture = new DesktopFixtureSession(sandbox, env, info)

  // Credential identity stays native. Only a one-way digest leaves the owner
  // process; DPAPI ciphertext and its mutable readiness cache may change.
  const credentialIdentity = () => execFileSync(process.env.HERMES_PYTHON!, ['-P', '-c', `
import hashlib
from hermes_cli.cloud_device_client import _read,credential
state=_read()
ref=state['credential_ref'];base=state['origin']+'/v1'
secret=credential(ref,base_url=base)
print(hashlib.sha256((ref+'\\0'+base+'\\0'+secret).encode()).hexdigest())
`], { cwd: path.resolve(import.meta.dirname, '../../..'), env: { ...env, PYTHONPATH: path.resolve(import.meta.dirname, '../../..') }, encoding: 'utf8', timeout: 20_000 }).trim()


  const existingSite = http.createServer((request, response) => {
    if (request.url === '/login') {
      response.writeHead(302, { 'Set-Cookie': 'fixture_account=synthetic; HttpOnly; SameSite=Lax; Path=/', Location: '/account' })
      response.end()

      return
    }

    response.statusCode = request.headers.cookie?.includes('fixture_account=synthetic') ? 200 : 401
    response.setHeader('Content-Type', 'text/html')
    response.end(response.statusCode === 200 ? '<h1>Fixture account signed in</h1>' : '<h1>Sign in required</h1>')
  })

  try {
    await new Promise<void>(resolve => existingSite.listen(0, '127.0.0.1', resolve))
    const existingUrl = `http://localhost:${(existingSite.address() as any).port}/login`
    const { app, page } = await fixture.launch()
    await app.evaluate(({ ipcMain, app }, trust) => {
      if (trust.fingerprint) {
        app.on('certificate-error', (event: { preventDefault(): void }, _contents: unknown, url: string, _error: string, certificate: { data: string }, callback: (trusted: boolean) => void) => {
          const actual = new (process.getBuiltinModule('crypto').X509Certificate)(certificate.data)
          const valid = Date.parse(actual.validFrom) <= Date.now() && Date.now() <= Date.parse(actual.validTo)
          const hostname = new URL(url).hostname
          const hostMatches = hostname === '127.0.0.1' ? actual.checkIP(hostname) : actual.checkHost(hostname)

          if (new URL(url).origin === trust.origin && actual.fingerprint256 === trust.fingerprint && valid && hostMatches) {event.preventDefault(); callback(true)} else {callback(false)}
        })
      }

      ;(globalThis as any).__billingExternalCalls = 0
      ipcMain.removeHandler('hermes:openExternal')
      ipcMain.handle('hermes:openExternal', () => { (globalThis as any).__billingExternalCalls++; throw new Error('External launch forbidden') })
    }, { origin: origin!, fingerprint: certificate ? new X509Certificate(fs.readFileSync(certificate)).fingerprint256 : '' })
    await fixture.stage('browser.existing-account-setup', async () => {
      await page.getByRole('button', { name: 'Sign in to Semantier Model Provider', exact: true }).waitFor({ state: 'visible', timeout: 180_000 })
      await page.evaluate(async url => {
        const preview = await import(/* @vite-ignore */ String('/src/store/preview.ts'))
        preview.openPreview({ kind: 'url', url, source: url, label: 'Existing fixture account' })
      }, existingUrl)
      const decline = page.getByRole('button', { name: "Don't show again", exact: true })
      await decline.waitFor({ state: 'visible', timeout: 10_000 }).catch(() => undefined)

      if (await decline.isVisible()) { await decline.click() }
      await expect.poll(() => page.locator('webview').first().evaluate((view: any) => view.executeJavaScript('document.body.innerText')), { timeout: 30_000 }).toContain('Fixture account signed in')
    })
    const originalGuest = page.locator('webview').first()
    const originalGuestId = await originalGuest.evaluate((view: any) => view.getWebContentsId())
    await page.getByRole('button', { name: 'Sign in to Semantier Model Provider', exact: true }).click({ timeout: 180_000 })
    const guest = page.locator('webview').last()
    await expect(guest).toBeVisible({ timeout: 60_000 })
    await expect.poll(() => guest.evaluate((view: any) => view.getURL()).catch(() => ''), { timeout: 60_000 }).toContain('/billing/connect?intent_ref=dci_')
    await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText')).catch(() => ''), { timeout: 60_000 }).toMatch(/Sign in|登录/)
    expect(await app.evaluate(() => (globalThis as any).__billingExternalCalls)).toBe(0)
    expect(await originalGuest.evaluate((view: any) => view.getWebContentsId())).toBe(originalGuestId)
    expect(await originalGuest.evaluate((view: any) => view.getURL())).toBe(existingUrl.replace('/login', '/account'))

    const clickGuest = async (labels: string[], selector = 'button') => {
      await expect.poll(() => guest.evaluate((view: any, args) => view.executeJavaScript(`Array.from(document.querySelectorAll(${JSON.stringify(args.selector)})).some(node => ${JSON.stringify(args.labels)}.includes(node.textContent.trim()) && !node.disabled)`), { labels, selector }), { timeout: 15_000 }).toBe(true)
      const rect = await guest.evaluate((view: any, args) => view.executeJavaScript(`(() => { const node = Array.from(document.querySelectorAll(${JSON.stringify(args.selector)})).find(node => ${JSON.stringify(args.labels)}.includes(node.textContent.trim())); if (!node || node.disabled) throw new Error('Guest action unavailable'); node.scrollIntoView({block:'center'}); const rect=node.getBoundingClientRect(); return {x:rect.x,y:rect.y,width:rect.width,height:rect.height}; })()`), { labels, selector })
      const host = await guest.boundingBox()
      expect(host).toBeTruthy()
      await page.mouse.click(host!.x + rect.x + rect.width / 2, host!.y + rect.y + rect.height / 2)
    }

    const credentialsFile = process.env.HERMES_E2E_BILLING_IDENTITY_FILE

    if (credentialsFile) {
      await fixture.stage('billing.real-login', async () => {
        const declineSnapshot = page.getByRole('button', { name: "Don't show again", exact: true })

        if (await declineSnapshot.isVisible()) {await declineSnapshot.click()}
        await clickGuest(['Sign in', '登录'])
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript(`Boolean(document.querySelector('input[name="username"]'))`)), { timeout: 30_000 }).toBe(true)
        const guestId = await guest.evaluate((view: any) => view.getWebContentsId())
        await app.evaluate(async ({ webContents }, { guestId, credentialsFile }) => {
          const { readFileSync } = process.getBuiltinModule('fs')
          const credentials = JSON.parse(readFileSync(credentialsFile, 'utf8'))
          const target = webContents.fromId(guestId)!

          for (const [name, value] of [['username', credentials.login], ['password', credentials.password]]) {
            await target.executeJavaScript(`(() => {const input = document.querySelector('input[name="${name}"]'); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, ${JSON.stringify(value)}); input.dispatchEvent(new Event('input', {bubbles: true}));})()`)
          }


        }, { guestId, credentialsFile })
        await clickGuest(['Continue With Password', '使用密码继续'], 'form button[type="submit"]')
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText')), { timeout: 60_000 }).toMatch(/Log out|退出登录/)
      })
      await fixture.stage('billing.embedded-static-en-zh', async () => {
        await app.evaluate(({ BrowserWindow }) => {const window = BrowserWindow.getAllWindows()[0]; window.unmaximize(); window.setSize(1280, 900)})
        await expect.poll(async () => { const box=await guest.boundingBox(); const width=await guest.evaluate((view: any)=>view.executeJavaScript('innerWidth'));

 return Math.abs(width-(box?.width || 0)) }, { timeout: 15_000 }).toBeLessThan(3)

        for (const [language, toggle] of [['en', ['EN']], ['zh', ['中']]] as const) {
          await clickGuest([...toggle])
          await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.documentElement.lang')), { timeout: 30_000 }).toBe(language)
          const metrics = await guest.evaluate((view: any) => view.executeJavaScript(`(async () => {await document.fonts.ready; const root=document.querySelector('.semantier-billing'); return {font:getComputedStyle(root).fontFamily, width:document.documentElement.clientWidth, overflow:document.documentElement.scrollWidth>document.documentElement.clientWidth, animations:document.getAnimations().filter(a=>a.playState==='running').length};})()`))
          expect(metrics.font).toMatch(/Manrope/i)
          expect(metrics.overflow).toBe(false)
          expect(metrics.animations).toBe(0)
          await page.screenshot({ path: info.outputPath('embedded-billing-' + language + '.png') })
          console.log('BILLING_VISUAL', JSON.stringify({ language, ...metrics }))
        }
      })
      await fixture.stage('billing.explicit-approval', async () => {
        const account = process.env.HERMES_E2E_BILLING_ACCOUNT
        expect(account, 'Explicit permitted fixture account required').toBeTruthy()
        await expect.poll(() => guest.evaluate((view: any, account) => view.executeJavaScript(`Array.from(document.querySelectorAll('select option')).some(option => option.value === ${JSON.stringify(account)})`), account), { timeout: 30_000 }).toBe(true)
        await guest.evaluate((view: any, account) => view.executeJavaScript(`(() => {const select=Array.from(document.querySelectorAll('select')).find(node=>Array.from(node.options).some(option=>option.value===${JSON.stringify(account)})); if(select){select.value=${JSON.stringify(account)};select.dispatchEvent(new Event('change',{bubbles:true}));}})()`), account)

        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText')), { timeout: 30_000 }).toMatch(/Approve device|授权此设备/)

        if (outage) { setCatalogFault(true) }
        await clickGuest(['Approve device', '授权此设备'])
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText')), { timeout: 30_000 }).toMatch(/Approved|已授权/)
      })
    }

    if (credentialsFile) {
      await fixture.stage('cloud.native-install-and-model', async () => {
        const configPath = sandbox.hermesHome + '/config.yaml'
        await expect.poll(() => fs.readFileSync(configPath, 'utf8').includes('credential_ref:'), { timeout: 60_000 }).toBe(true)
        expect(fs.existsSync(sandbox.hermesHome + '/.cloud-device.bin')).toBe(true)

        if (outage) {
          await expect(fixture.button('Try again')).toBeVisible({ timeout: 60_000 })
          await expect(page.getByText(/Model catalog returned a server error/)).toBeVisible()
          const box = await guest.boundingBox()
          expect(box).toBeTruthy()
          expect(await page.evaluate(({ x, y }) => document.elementFromPoint(x, y)?.tagName, { x: box!.x + box!.width / 2, y: box!.y + box!.height / 2 })).toBe('WEBVIEW')
          const installed = credentialIdentity()
          setCatalogFault(false)
          await fixture.button('Try again').click()
          await expect(fixture.button('Begin')).toBeVisible({ timeout: 60_000 })
          expect(credentialIdentity() === installed).toBe(true)
          const counts = { start: operationCount('start') - before!.start, claim: operationCount('claim') - before!.claim, ack: operationCount('ack') - before!.ack }
          expect(counts).toEqual({ start: 1, claim: 1, ack: 1 })
          console.log('CATALOG_REAL_RECOVERY', JSON.stringify({ counts, transportFault: 'HTTP503', credentialIdentity: 'retained' }))
        }

        if (noAccess) {
          await expect(page.getByText(/This account has no enabled model/)).toBeVisible({ timeout: 60_000 })
          expect(fs.readFileSync(configPath, 'utf8').includes("default: ''")).toBe(true)
          await expect(fixture.button('Try again')).toBeVisible()
          console.log('CATALOG_REAL_EMPTY', JSON.stringify({ installed: true, selectedModel: false }))

          return
        }

        const model = page.getByRole('button', { name: /Model.*local-fixture/ })
        await expect.poll(async () => (await fixture.button('Begin').isVisible()) || (await model.isVisible()), { timeout: 90_000 }).toBe(true)

        if (await fixture.button('Begin').isVisible()) { await fixture.button('Begin').click() }
        await expect.poll(() => fs.readFileSync(configPath, 'utf8').match(/^ {2}default: ['"]?([^'"\n]+)['"]?$/m)?.[1], { timeout: 30_000 }).toBe('local-fixture')
        await expect(fixture.button('Begin')).not.toBeVisible({ timeout: 30_000 })
        await expect(model).toBeVisible()
        await expect(page.getByRole('button', { name: 'Try again', exact: true })).not.toBeVisible()
        expect(fs.readFileSync(configPath, 'utf8').includes('capability_revision:')).toBe(true)
      })

      if (noAccess) { return }
      await fixture.stage('billing.login-reuse-and-real-logout', async () => {
        const protectedStore = credentialIdentity()
        await page.getByRole('button', { name: 'Reload page', exact: true }).click()
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText')), { timeout: 45_000 }).toMatch(/Log out|退出登录/)
        await clickGuest(['Log out', '退出登录'])
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript(`fetch('/auth/context',{cache:'no-store'}).then(r=>r.json()).then(v=>v.authenticated)`)), { timeout: 30_000 }).toBe(false)
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText'))).not.toContain('semantier-billing-qa-1112')
        expect(await guest.evaluate((view: any) => view.executeJavaScript(`fetch('/api/billing/commerce/accounts',{cache:'no-store'}).then(r=>r.status)`))).toBe(401)
        const back = page.getByRole('button', { name: 'Back', exact: true })

        if (await back.isEnabled()) { await back.click() }
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText'))).not.toContain('unified-fixture')
        expect(credentialIdentity() === protectedStore).toBe(true)
        expect(await originalGuest.evaluate((view: any) => view.getWebContentsId())).toBe(originalGuestId)
        expect(await originalGuest.evaluate((view: any) => view.getURL())).toBe(existingUrl.replace('/login', '/account'))
        expect(await originalGuest.evaluate((view: any) => view.executeJavaScript(`fetch('/account').then(r=>r.status)`))).toBe(200)
      })
      await fixture.stage('cloud.controlled-local-chat', async () => {
        const composer = page.getByRole('textbox', { name: 'Message', exact: true })
        await composer.fill('Run the local connection check.')
        await composer.press('Enter')
        const response = page.getByText('Cloud connection verified.', { exact: true })
        const failure = page.getByRole('alert').filter({ hasText: "couldn't finish this reply" })
        await expect.poll(async () => (await response.isVisible()) || (await failure.isVisible()), { timeout: 90_000 }).toBe(true)

        if (await failure.isVisible()) {
          await failure.locator('summary').click()
          const directory = sandbox.hermesHome + '/logs'

          for (const name of fs.readdirSync(directory).filter(name => name.endsWith('.log'))) {
            const lines = fs.readFileSync(directory + '/' + name, 'utf8').split('\n').slice(-1200)

            const frames = lines.flatMap(line => {
              if (/^\s*File .*line \d+/.test(line)) { return [line] }
              const exception = line.match(/^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):/)

              return exception ? [exception[1]] : []
            }).slice(-50)

            if (frames.length) { console.log('OWNED_GATEWAY_FAILURE_STACK', JSON.stringify({ name, frames })) }
          }
        }

        await expect(response).toBeVisible()

      })
      await fixture.stage('cloud.restart-persistence', async () => {
        const installed = credentialIdentity()
        const restarted = await fixture.restart()
        await expect(restarted.page.getByRole('button', { name: /Model.*local-fixture/ })).toBeVisible({ timeout: 120_000 })
        await expect(restarted.page.getByRole('button', { name: 'Sign in to Semantier Model Provider', exact: true })).not.toBeVisible()
        expect(credentialIdentity() === installed).toBe(true)
        await restarted.page.screenshot({ path: info.outputPath('model-ready-after-restart.png') })
      })
    }

    if (!credentialsFile) { await page.screenshot({ path: info.outputPath('embedded-login.png') }) }
    console.log('EMBEDDED_REAL_LOGIN', JSON.stringify({ platform: process.platform, origin, externalCalls: 0, unconfigured: true }))
  } finally {
    try {
      if (faultActive) { setCatalogFault(false) }
    } finally {
      try { await fixture.close() } finally {
        existingSite.closeAllConnections()
        await new Promise<void>(resolve => existingSite.close(() => resolve()))
      }
    }
  }
})
