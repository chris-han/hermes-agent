/** Synthetic legacy pairing adopted by the actual Windows source Electron app. */
import { execFile } from 'node:child_process'
import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import { promisify } from 'node:util'

import { expect, test } from '@playwright/test'
import { DesktopFixtureSession } from './fixture-session'
import { buildAppEnv, createSandbox } from './fixtures'

// eslint-disable-next-line no-empty-pattern -- Electron supplies its own page.
test('Windows adopts an opted-in legacy pair using a device grant', async ({}, info) => {
  test.skip(process.platform !== 'win32' || !process.env.FILEX_PORTABLE_FIXTURE || !process.env.HERMES_DESKTOP_DEV_SERVER,
    'Requires the isolated Windows source-Electron fixture')
  test.setTimeout(300_000)
  const fixturePath = process.env.FILEX_PORTABLE_FIXTURE!
  const account = JSON.parse(await fs.readFile(fixturePath, 'utf8'))
  const origin: string = JSON.parse(await fs.readFile(path.resolve(path.dirname(fixturePath), '..', 'services.json'), 'utf8')).web
  const sandbox = createSandbox('file-sync-legacy-native')
  const folder = path.join(sandbox.root, 'documents')
  await fs.mkdir(folder)
  await fs.writeFile(path.join(sandbox.hermesHome, 'config.yaml'), 'language: en\ndisplay:\n  language: en\n')
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
    await expect(page.getByText('Fixture', { exact: true })).toBeVisible({ timeout: 60_000 })
    const scope = (await page.evaluate(() => (window as any).hermesDesktop.connections.list())).accountContext.scope
    const partition = 'persist:semantier-workspace-' + createHash('sha256').update(origin).digest('hex')
    const cookies = await app.evaluate(async ({ session }, name) => session.fromPartition(name).cookies.get({}), partition)
    const cookie = cookies.map((value: { name: string; value: string }) => `${value.name}=${value.value}`).join('; ')
    const response = await fetch(origin + '/api/file-sync/binding', { method: 'POST',
      headers: { cookie, Origin: origin, 'X-Semantier-CSRF': '1' } })
    expect(response.status).toBe(200)
    const old = await response.json()
    const native = await app.evaluate(({ app, safeStorage }, token) => {
      if (!safeStorage.isEncryptionAvailable()) {throw new Error('Windows credential storage unavailable')}
      return { userData: app.getPath('userData'), sealed: safeStorage.encryptString(token).toString('base64') }
    }, old.token)
    const id = randomUUID()
    const stateDir = path.join(native.userData, 'file-sync')
    const pairDir = path.join(stateDir, id)
    const canonical = await fs.realpath(folder)
    await fs.mkdir(pairDir, { recursive: true })
    await promisify(execFile)(process.env.FILEX_TEST_BINARY!, ['sync', 'add', canonical, old.remoteRoot, '--account', id], {
      env: { ...process.env, FILEX_URL: origin + old.enginePath, FILEX_TOKEN: old.token,
        FILEX_SYNC_DIR: pairDir, FILEX_CLI_CONFIG: path.join(pairDir, 'cli.yaml') }, timeout: 30_000
    })
    await fs.writeFile(path.join(folder, 'retained.txt'), 'Legacy fixture bytes\n')
    await fs.writeFile(path.join(stateDir, 'folder-owners.json'), JSON.stringify({ version: 1, selectedBindingIds: [id], bindings: [{
      id, serverOrigin: origin, userId: scope.userId, workspaceId: scope.workspaceId, organizationId: scope.organizationId,
      canonicalLocalFolder: canonical, remoteRoot: old.remoteRoot, sealedToken: native.sealed, enabled: false, paired: true
    }] }))
    await page.getByRole('button', { name: 'File Sync', exact: true }).click()
    await expect(page.getByText(canonical, { exact: true })).toBeVisible({ timeout: 30_000 })
    await page.getByRole('button', { name: 'Resume', exact: true }).click()
    const name = `legacy-adopted-${randomUUID()}.md`
    await fs.writeFile(path.join(folder, name), 'New device grant bytes\n')
    await expect.poll(async () => {
      const result = await fetch(origin + '/api/file-sync/files?download=true&path=' + name, { headers: { cookie,
        'X-Semantier-User-Id': account.user_id, 'X-Semantier-Workspace-Id': account.workspace_id } })
      return result.status === 206 ? result.text() : ''
    }, { timeout: 40_000 }).toBe('New device grant bytes\n')
    const registry = JSON.parse(await fs.readFile(path.join(stateDir, 'folder-owners.json'), 'utf8'))
    expect(registry.bindings[0]).toMatchObject({ id, paired: true, enabled: true, deviceGrant: true })
    const token = await app.evaluate(({ safeStorage }, value) => safeStorage.decryptString(Buffer.from(value, 'base64')), registry.bindings[0].sealedToken)
    expect(token).toMatch(/^fxd_[A-Za-z0-9_-]{43}$/)
    expect(token).not.toBe(old.token)
    expect(await fs.readFile(path.join(folder, 'retained.txt'), 'utf8')).toBe('Legacy fixture bytes\n')
    await page.getByRole('button', { name: 'Pause', exact: true }).click()
    await expect(page.getByText('Paused', { exact: true })).toBeVisible()
    expect((await page.evaluate(() => (window as any).hermesDesktop.connections.list())).primary).toBe('local')
  } finally { await fixture.close() }
})
