/** Chinese recovery states in the isolated Windows source-Electron fixture. */
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { expect, test } from '@playwright/test'
import { DesktopFixtureSession } from './fixture-session'
import { buildAppEnv, createSandbox } from './fixtures'

test('Chinese account recovery preserves the authenticated session', async ({}, info) => {
  test.skip(process.platform !== 'win32' || !process.env.FILEX_PORTABLE_FIXTURE || !process.env.HERMES_DESKTOP_DEV_SERVER,
    'Requires the isolated Windows source-Electron fixture')
  test.setTimeout(300_000)
  const fixturePath = process.env.FILEX_PORTABLE_FIXTURE!
  const account = JSON.parse(fs.readFileSync(fixturePath, 'utf8'))
  const services = JSON.parse(fs.readFileSync(path.resolve(path.dirname(fixturePath), '..', 'services.json'), 'utf8'))
  const origin: string = services.web
  const sandbox = createSandbox('account-recovery-zh')
  fs.writeFileSync(path.join(sandbox.hermesHome, 'config.yaml'), 'language: zh\ndisplay:\n  language: zh\n')
  const fixture = new DesktopFixtureSession(sandbox, buildAppEnv(sandbox, {
    HERMES_PYTHON: process.env.HERMES_PYTHON!, HERMES_DESKTOP_PYTHON: process.env.HERMES_PYTHON!,
    HERMES_DESKTOP_DEV_SERVER: process.env.HERMES_DESKTOP_DEV_SERVER!, HERMES_DESKTOP_CDP_PORT: 'off'
  }), info)
  try {
    const { app, page } = await fixture.launch()
    await expect(page.getByRole('button', { name: 'default', exact: true })).toHaveAttribute('aria-pressed', 'true', { timeout: 180_000 })
    expect((await page.evaluate(() => (window as any).hermesDesktop.connections.list())).primary).toBe('local')
    await page.getByRole('button', { name: '打开设置', exact: true }).click({ timeout: 90_000 })
    await page.getByRole('button', { name: '模型服务', exact: true }).click()
    await page.getByRole('button', { name: '账号', exact: true }).click()
    await page.getByLabel('工作区地址').fill(origin)
    const popupPromise = app.waitForEvent('window')
    await page.getByRole('button', { name: '登录', exact: true }).click()
    const popup = await popupPromise
    await popup.getByLabel('Login').fill(account.login)
    await popup.getByLabel('Password').fill(account.password)
    await popup.getByRole('button', { name: 'Sign in fixture', exact: true }).click()
    await expect(page.getByText('Fixture', { exact: true })).toBeVisible({ timeout: 60_000 })
    const partition = 'persist:semantier-workspace-' + createHash('sha256').update(origin).digest('hex')
    const cookies = await app.evaluate(async ({ session }, name) => session.fromPartition(name).cookies.get({}), partition)
    const cookie = cookies.map((item: { name: string; value: string }) => `${item.name}=${item.value}`).join('; ')
    const gate = async (action: string) => {
      const response = await fetch(origin + '/__fixture/filex-gate', { method: 'POST',
        headers: { cookie, Origin: origin, 'X-Semantier-CSRF': '1', 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }) })
      expect(response.status).toBe(200)
    }
    const openAccount = async () => {
      await page.getByRole('button', { name: '文件同步', exact: true }).click()
      await page.getByRole('button', { name: '模型服务', exact: true }).click()
      await page.getByRole('button', { name: '账号', exact: true }).click()
    }
    for (const [action, message, screenshot] of [
      ['identity-offline', '暂时无法验证工作区身份，请重试。', 'identity-503-zh'],
      ['identity-403', '此账户或组织无权访问工作区。', 'identity-403-zh'],
      ['identity-delay', '暂时无法验证工作区身份，请重试。', 'identity-timeout-zh']
    ]) {
      await gate(action)
      await openAccount()
      await expect(page.getByRole('alert')).toContainText(message, { timeout: 30_000 })
      await expect(page.getByRole('button', { name: '重试', exact: true })).toBeVisible()
      await expect(page.getByRole('button', { name: '登录', exact: true })).toHaveCount(0)
      expect(await app.evaluate(async ({ session }, name) => session.fromPartition(name).cookies.get({}), partition)).toEqual(cookies)
      await info.attach(screenshot, { body: await page.screenshot(), contentType: 'image/png' })
      await gate('identity-online')
      await page.getByRole('button', { name: '重试', exact: true }).click()
      await expect(page.getByText('Fixture', { exact: true })).toBeVisible({ timeout: 30_000 })
    }
    await gate('identity-401')
    await openAccount()
    await expect(page.getByRole('button', { name: '登录', exact: true })).toBeVisible()
    expect(await app.evaluate(async ({ session }, name) => session.fromPartition(name).cookies.get({}), partition)).toEqual(cookies)
    await info.attach('identity-401-zh', { body: await page.screenshot(), contentType: 'image/png' })
    await gate('identity-online')
    await openAccount()
    await expect(page.getByText('Fixture', { exact: true })).toBeVisible({ timeout: 30_000 })
    expect((await page.evaluate(() => (window as any).hermesDesktop.connections.list())).primary).toBe('local')
  } finally {
    await fixture.close()
  }
})
