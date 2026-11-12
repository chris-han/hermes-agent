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
    await expect.poll(() => guest.evaluate((view: any) => view.getURL()).catch(() => '')).toContain('/billing/connect?intent_ref=dci_')
    await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText')).catch(() => ''), { timeout: 60_000 }).toMatch(/Sign in|登录/)
    expect(await app.evaluate(() => (globalThis as any).__billingExternalCalls)).toBe(0)
    const credentialsFile = process.env.HERMES_E2E_BILLING_CREDENTIALS

    if (credentialsFile) {
      await fixture.stage('billing.real-login', async () => {
        await guest.evaluate((view: any) => view.executeJavaScript(`Array.from(document.querySelectorAll('button')).find(b => ['Sign in', '登录'].includes(b.textContent.trim()))?.click()`))
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript(`Boolean(document.querySelector('input[name="username"]'))`)), { timeout: 30_000 }).toBe(true)
        const guestId = await guest.evaluate((view: any) => view.getWebContentsId())
        await app.evaluate(async ({ webContents }, { guestId, credentialsFile }) => {
          const { readFileSync } = await import('node:fs')
          const credentials = JSON.parse(readFileSync(credentialsFile, 'utf8'))
          const target = webContents.fromId(guestId)!

          for (const [name, value] of [['username', credentials.login], ['password', credentials.password]]) {
            await target.executeJavaScript(`(() => {const input = document.querySelector('input[name="${name}"]'); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, ${JSON.stringify(value)}); input.dispatchEvent(new Event('input', {bubbles: true}));})()`)
          }

          await target.executeJavaScript(`document.querySelector('input[name="password"]').form.requestSubmit()`)
        }, { guestId, credentialsFile })
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText')), { timeout: 60_000 }).toMatch(/Log out|退出登录/)
      })
      await fixture.stage('billing.explicit-approval', async () => {
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText')), { timeout: 30_000 }).toMatch(/Approve device|授权此设备/)
        await guest.evaluate((view: any) => view.executeJavaScript(`Array.from(document.querySelectorAll('button')).find(b => ['Approve device', '授权此设备'].includes(b.textContent.trim()))?.click()`))
        await expect.poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText')), { timeout: 30_000 }).toMatch(/Approved|已授权/)
      })
    }

    await page.screenshot({ path: info.outputPath('embedded-login.png') })
    console.log('EMBEDDED_REAL_LOGIN', JSON.stringify({ platform: process.platform, origin, externalCalls: 0, unconfigured: true }))
  } finally { await fixture.close() }
})
