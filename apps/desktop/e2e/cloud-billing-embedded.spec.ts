/** First real loop: native card -> real Runtime intent -> Workspace login in owned pane. */
import fs from 'node:fs'

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
  const sandbox = createSandbox('billing-embedded')
  writeEnvFile(sandbox.hermesHome)
  fs.writeFileSync(sandbox.hermesHome + '/config.yaml', `language: en\ndisplay:\n  language: en\nbilling: null\nproviders:\n  semantier-cloud:\n    base_url: ${origin}/v1\n    portal_origin: ${origin}\n`)
  const fixture = new DesktopFixtureSession(sandbox, buildAppEnv(sandbox, { HERMES_DESKTOP_DEV_SERVER: process.env.HERMES_DESKTOP_DEV_SERVER! }), info)

  try {
    const { app, page } = await fixture.launch()
    await app.evaluate(({ ipcMain }) => {
      ;(globalThis as any).__billingExternalCalls = 0
      ipcMain.removeHandler('hermes:openExternal')
      ipcMain.handle('hermes:openExternal', () => { (globalThis as any).__billingExternalCalls++; throw new Error('External launch forbidden') })
    })
    await page.getByRole('button', { name: 'Sign in to Semantier Model Provider', exact: true }).click({ timeout: 180_000 })
    const guest = page.locator('webview').last()
    await expect(guest).toBeVisible({ timeout: 60_000 })
    await expect.poll(() => guest.evaluate((view: any) => view.getURL()).catch(() => ''), { timeout: 60_000 }).toContain('/billing/connect?intent_ref=dci_')
    await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText')).catch(() => ''), { timeout: 60_000 }).toMatch(/Sign in|登录/)
    expect(await app.evaluate(() => (globalThis as any).__billingExternalCalls)).toBe(0)

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
        const declineSnapshot = page.getByRole('button', { name: 'Not now', exact: true })

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
        await clickGuest(['Approve device', '授权此设备'])
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText')), { timeout: 30_000 }).toMatch(/Approved|已授权/)
      })
    }

    if (credentialsFile) {
      await fixture.stage('cloud.native-install-and-model', async () => {
        const configPath = sandbox.hermesHome + '/config.yaml'
        await expect.poll(() => fs.readFileSync(configPath, 'utf8').includes('credential_ref:'), { timeout: 60_000 }).toBe(true)
        expect(fs.existsSync(sandbox.hermesHome + '/.cloud-device.bin')).toBe(true)
        const model = page.getByRole('button', { name: /Model.*local-fixture/ })
        await expect.poll(async () => (await fixture.button('Begin').isVisible()) || (await model.isVisible()), { timeout: 90_000 }).toBe(true)

        if (await fixture.button('Begin').isVisible()) { await fixture.button('Begin').click() }
        await expect.poll(() => fs.readFileSync(configPath, 'utf8').match(/^ {2}default: ['"]?([^'"\n]+)['"]?$/m)?.[1], { timeout: 30_000 }).toBe('local-fixture')
        await expect(fixture.button('Begin')).not.toBeVisible({ timeout: 30_000 })
        await expect(model).toBeVisible()
        await expect(page.getByRole('button', { name: 'Try again', exact: true })).not.toBeVisible()
        expect(fs.readFileSync(configPath, 'utf8').includes('capability_revision:')).toBe(true)
      })
      await fixture.stage('billing.login-reuse-and-real-logout', async () => {
        const protectedStore = fs.readFileSync(sandbox.hermesHome + '/.cloud-device.bin')
        await page.getByRole('button', { name: 'Reload page', exact: true }).click()
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText')), { timeout: 45_000 }).toMatch(/Log out|退出登录/)
        await clickGuest(['Log out', '退出登录'])
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript(`fetch('/auth/context',{cache:'no-store'}).then(r=>r.json()).then(v=>v.authenticated)`)), { timeout: 30_000 }).toBe(false)
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText'))).not.toContain('semantier-billing-qa-1112')
        expect(await guest.evaluate((view: any) => view.executeJavaScript(`fetch('/api/billing/commerce/accounts',{cache:'no-store'}).then(r=>r.status)`))).toBe(401)
        const back = page.getByRole('button', { name: 'Back', exact: true })

        if (await back.isEnabled()) { await back.click() }
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText'))).not.toContain('unified-fixture')
        expect(fs.readFileSync(sandbox.hermesHome + '/.cloud-device.bin').equals(protectedStore)).toBe(true)
      })
      await fixture.stage('cloud.controlled-local-chat', async () => {
        const composer = page.getByRole('textbox', { name: 'Message', exact: true })
        await composer.fill('Run the local connection check.')
        await composer.press('Enter')
        const response = page.getByText('Cloud connection verified.', { exact: true })
        const failure = page.getByRole('alert').filter({ hasText: "couldn't finish this reply" })
        await expect.poll(async () => (await response.isVisible()) || (await failure.isVisible()), { timeout: 90_000 }).toBe(true)

        if (await failure.isVisible()) { await failure.locator('summary').click() }
        await expect(response).toBeVisible()

      })
      await fixture.stage('cloud.restart-persistence', async () => {
        const restarted = await fixture.restart()
        await expect(restarted.page.getByRole('button', { name: /Model.*local-fixture/ })).toBeVisible({ timeout: 120_000 })
        await expect(restarted.page.getByRole('button', { name: 'Sign in to Semantier Model Provider', exact: true })).not.toBeVisible()
        await restarted.page.screenshot({ path: info.outputPath('model-ready-after-restart.png') })
      })
    }

    if (!credentialsFile) { await page.screenshot({ path: info.outputPath('embedded-login.png') }) }
    console.log('EMBEDDED_REAL_LOGIN', JSON.stringify({ platform: process.platform, origin, externalCalls: 0, unconfigured: true }))
  } finally {
    await fixture.close()
  }
})
