/** Isolated Windows source-Electron stop request against a fixture-owned authority. */
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import { expect, test } from '@playwright/test'
import { DesktopFixtureSession } from './fixture-session'
import { buildAppEnv, createSandbox } from './fixtures'

test('turning Winman off interrupts the owned fixture turn', async ({}, info) => {
  test.skip(process.platform !== 'win32' || !process.env.WINMAN_STOP_FIXTURE || !process.env.HERMES_DESKTOP_DEV_SERVER,
    'Requires the isolated Windows source-Electron fixture')
  test.setTimeout(240_000)
  const account = JSON.parse(fs.readFileSync(process.env.WINMAN_STOP_FIXTURE!, 'utf8'))
  const origin: string = account.origin
  const sandbox = createSandbox('winman-stop-native')
  fs.writeFileSync(sandbox.hermesHome + '/config.yaml', 'language: en\ndisplay:\n  language: en\n')
  const fixture = new DesktopFixtureSession(sandbox, buildAppEnv(sandbox, {
    HERMES_PYTHON: process.env.HERMES_PYTHON!, HERMES_DESKTOP_PYTHON: process.env.HERMES_PYTHON!,
    HERMES_DESKTOP_DEV_SERVER: process.env.HERMES_DESKTOP_DEV_SERVER!, HERMES_DESKTOP_CDP_PORT: 'off'
  }), info)
  try {
    const { app, page } = await fixture.launch()
    await expect(page.getByRole('button', { name: 'default', exact: true })).toHaveAttribute('aria-pressed', 'true', { timeout: 180_000 })
    await page.getByRole('button', { name: 'Open settings', exact: true }).click()
    await page.getByRole('button', { name: 'Model service', exact: true }).click()
    await page.getByRole('button', { name: 'Accounts', exact: true }).click()
    await page.getByLabel('Workspace address').fill(origin)
    const popupPromise = app.waitForEvent('window')
    await page.getByRole('button', { name: 'Sign in', exact: true }).click()
    const popup = await popupPromise
    await popup.getByLabel('Login').fill(account.login)
    await popup.getByLabel('Password').fill(account.password)
    await popup.getByRole('button', { name: 'Sign in fixture', exact: true }).click()
    await expect(page.getByText('Preference off.', { exact: false })).toBeVisible({ timeout: 60_000 })
    await page.getByRole('button', { name: 'Request enablement', exact: true }).click()
    await page.getByRole('button', { name: 'Confirm request', exact: true }).click()
    await expect(page.getByText('Enablement requested.', { exact: false })).toBeVisible({ timeout: 20_000 })
    const partition = 'persist:semantier-workspace-' + createHash('sha256').update(origin).digest('hex')
    const cookies = await app.evaluate(async ({ session }, name) => session.fromPartition(name).cookies.get({}), partition)
    const cookie = cookies.map((item: { name: string; value: string }) => `${item.name}=${item.value}`).join('; ')
    const headers = { cookie, Origin: origin, 'X-Semantier-CSRF': '1' }
    const secondLogin = await fetch(origin + '/auth/password/login', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ login: account.login, password: account.password }) })
    expect(secondLogin.status).toBe(200)
    const secondCookie = secondLogin.headers.getSetCookie().map(value => value.split(';')[0]).join('; ')
    expect(secondCookie).toBeTruthy()
    const readSecondPreference = async () => {
      const response = await fetch(origin + '/api/winman/capability', { headers: { cookie: secondCookie } })
      expect(response.status).toBe(200)
      return response.json()
    }
    expect(await readSecondPreference()).toMatchObject({ desired: { enabled: true }, execution: 'unavailable' })
    expect((await fetch(origin + '/__fixture/start', { method: 'POST', headers })).status).toBe(200)
    const otherLogin = await fetch(origin + '/auth/password/login', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ login: account.other_login, password: account.password }) })
    expect(otherLogin.status).toBe(200)
    const otherCookie = otherLogin.headers.getSetCookie().map(value => value.split(';')[0]).join('; ')
    expect(otherCookie).toBeTruthy()
    const otherHeaders = { cookie: otherCookie!, Origin: origin, 'X-Semantier-CSRF': '1', 'Content-Type': 'application/json' }
    const otherOff = (scope: unknown) => fetch(origin + '/api/winman/capability', { method: 'PUT',
      headers: otherHeaders, body: JSON.stringify({ enabled: false, expected_scope: scope }) })
    expect((await otherOff(account.owner_scope)).status).toBe(403)
    expect((await otherOff(account.other_scope)).status).toBe(200)
    const beforeOwnerOff = await fetch(origin + '/__fixture/status', { headers })
    expect(await beforeOwnerOff.json()).toMatchObject({ interrupted: false, stop_requested: false, released: false })
    await page.getByRole('button', { name: 'Turn preference off', exact: true }).click()
    await expect(page.getByText('Preference off.', { exact: false })).toBeVisible({ timeout: 20_000 })
    expect(await readSecondPreference()).toMatchObject({ desired: { enabled: false }, execution: 'unavailable' })
    const response = await fetch(origin + '/__fixture/status', { headers })
    expect(response.status).toBe(200)
    const status = await response.json()
    if (account.noncooperative) {
      expect(status).toMatchObject({ interrupted: false, stop_requested: true, released: false,
        rows: { follower: { status: 'queued' } } })
      expect(status.rows.running.status).not.toBe('terminal')
      await expect(page.getByText('existing work status is unknown', { exact: false })).toBeVisible()
    } else {
      expect(status).toMatchObject({ interrupted: true, stop_requested: true, released: true,
        rows: { running: { status: 'terminal', outcome: 'interrupted' }, follower: { status: 'queued' } } })
    }
  } finally {
    await fixture.close()
  }
})
