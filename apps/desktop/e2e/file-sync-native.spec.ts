/** Windows source Electron with disposable account and FileX services. */
import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { expect, test } from '@playwright/test'

import { DesktopFixtureSession } from './fixture-session'
import { buildAppEnv, createSandbox } from './fixtures'

// eslint-disable-next-line no-empty-pattern -- Electron supplies its own page.
test('explicit File Sync in Windows Electron uses the selected account and local gateway', async ({}, info) => {
  test.skip(process.platform !== 'win32' || !process.env.FILEX_PORTABLE_FIXTURE || !process.env.HERMES_DESKTOP_DEV_SERVER,
    'Requires the connected Windows source-Electron fixture')
  test.setTimeout(300_000)
  const fixturePath = process.env.FILEX_PORTABLE_FIXTURE!
  const account = JSON.parse(fs.readFileSync(fixturePath, 'utf8'))
  const services = JSON.parse(fs.readFileSync(path.resolve(path.dirname(fixturePath), '..', 'services.json'), 'utf8'))
  const origin: string = services.web
  const sandbox = createSandbox('file-sync-native')
  const folder = path.join(sandbox.root, 'documents')
  fs.mkdirSync(folder)
  fs.writeFileSync(path.join(sandbox.hermesHome, 'config.yaml'), 'language: en\ndisplay:\n  language: en\n')
  const fixture = new DesktopFixtureSession(sandbox, buildAppEnv(sandbox, {
    HERMES_PYTHON: process.env.HERMES_PYTHON!, HERMES_DESKTOP_PYTHON: process.env.HERMES_PYTHON!,
    HERMES_DESKTOP_DEV_SERVER: process.env.HERMES_DESKTOP_DEV_SERVER!, HERMES_DESKTOP_CDP_PORT: 'off'
  }), info)
  try {
    const { app, page } = await fixture.launch()
    await fixture.stage('desktop.ready', async () => {
      await expect(page.getByRole('button', { name: 'default', exact: true })).toHaveAttribute('aria-pressed', 'true', { timeout: 180_000 })
    })
    const before = await page.evaluate(() => (window as any).hermesDesktop.connections.list())
    expect(before.primary).toBe('local')
    await page.getByRole('button', { name: 'Open settings', exact: true }).click()
    await expect(page.getByRole('button', { name: 'Model service', exact: true })).toBeVisible({ timeout: 60_000 })
    await page.getByRole('button', { name: 'Model service', exact: true }).click()
    await page.getByRole('button', { name: 'Accounts', exact: true }).click()
    await page.getByLabel('Workspace address').fill(origin)
    const popupPromise = app.waitForEvent('window')
    await page.getByRole('button', { name: 'Sign in', exact: true }).click()
    const popup = await popupPromise
    await popup.getByLabel('Login').fill(account.login)
    await popup.getByLabel('Password').fill(account.password)
    await popup.getByRole('button', { name: 'Sign in fixture', exact: true }).click()
    if (process.env.NATIVE_MODEL_E2E === '1') {
      await expect.poll(() => page.evaluate(() => (window as any).hermesDesktop.workspace.modelDefault()),
        { timeout: 90_000 }).toMatchObject({ model: 'deepseek-v4.1-flash-1017' })
    } else {
      await expect(page.getByText('Fixture', { exact: true })).toBeVisible({ timeout: 60_000 })
    }
    await expect(page.getByText('Preference off.', { exact: false })).toBeVisible({ timeout: 20_000 })
    await page.getByRole('button', { name: 'File Sync', exact: true }).click()
    await expect(page.getByText('No folder selected', { exact: true })).toBeVisible()
    await page.getByRole('button', { name: 'Model service', exact: true }).click()
    await page.getByRole('button', { name: 'Accounts', exact: true }).click()
    await page.getByRole('button', { name: 'Request enablement', exact: true }).click()
    await page.getByRole('button', { name: 'Confirm request', exact: true }).click()
    await expect(page.getByText('Enablement requested.', { exact: false })).toBeVisible({ timeout: 20_000 })
    await page.getByRole('button', { name: 'File Sync', exact: true }).click()
    await expect(page.getByText('No folder selected', { exact: true })).toBeVisible()
    await app.evaluate(({ dialog }, selected) => {
      ;(dialog as any).showOpenDialog = async () => ({ canceled: false, filePaths: [selected] })
      ;(dialog as any).showMessageBox = async () => ({ response: 1, checkboxChecked: false })
    }, folder)
    await page.getByRole('button', { name: 'Choose and enable', exact: true }).click()
    await expect(page.getByText(await fs.promises.realpath(folder), { exact: true })).toBeVisible({ timeout: 60_000 })
    const name = `native-${randomUUID()}.md`
    fs.writeFileSync(path.join(folder, name), 'Windows fixture upload\n')
    const partition = 'persist:semantier-workspace-' + createHash('sha256').update(origin).digest('hex')
    const cookies = await app.evaluate(async ({ session }, partitionName) => session.fromPartition(partitionName).cookies.get({}), partition)
    const cookie = cookies.map((value: { name: string; value: string }) => `${value.name}=${value.value}`).join('; ')
    await expect.poll(async () => {
      const response = await fetch(origin + '/api/file-sync/files?download=true&path=' + name, { headers: { cookie,
        'X-Semantier-User-Id': account.user_id, 'X-Semantier-Workspace-Id': account.workspace_id } })
      return response.status === 206 ? response.text() : ''
    }, { timeout: 40_000 }).toBe('Windows fixture upload\n')
    if (process.env.FILEX_OUTAGE_FIXTURE === '1') {
      const gate = async (action: string) => {
        const response = await fetch(origin + '/__fixture/filex-gate', { method: 'POST',
          headers: { cookie, 'content-type': 'application/json' }, body: JSON.stringify({ action }) })
        expect(response.status).toBe(200)
      }
      const offlineName = `native-offline-${randomUUID()}.md`
      await gate('offline')
      fs.writeFileSync(path.join(folder, offlineName), 'Explicit Windows resume\n')
      await expect.poll(() => page.evaluate(() => (window as any).hermesDesktop.fileSync.status()),
        { timeout: 40_000 }).toMatchObject({ state: 'Error' })
      await gate('online')
      await new Promise(resolve => setTimeout(resolve, 6500))
      const offlineRead = () => fetch(origin + '/api/file-sync/files?download=true&path=' + offlineName, { headers: { cookie,
        'X-Semantier-User-Id': account.user_id, 'X-Semantier-Workspace-Id': account.workspace_id } })
      expect((await offlineRead()).status).not.toBe(206)
      await page.getByRole('button', { name: 'Resume', exact: true }).click()
      await expect.poll(async () => {
        const response = await offlineRead()
        return response.status === 206 ? response.text() : ''
      }, { timeout: 40_000 }).toBe('Explicit Windows resume\n')
      const partialName = `native-partial-${randomUUID()}.md`
      await gate('upload-offline')
      fs.writeFileSync(path.join(folder, partialName), 'Explicit partial resume\n')
      await expect.poll(() => page.evaluate(() => (window as any).hermesDesktop.fileSync.status()),
        { timeout: 40_000 }).toMatchObject({ state: 'Error' })
      await gate('upload-online')
      await new Promise(resolve => setTimeout(resolve, 6500))
      const partialRead = () => fetch(origin + '/api/file-sync/files?download=true&path=' + partialName, { headers: { cookie,
        'X-Semantier-User-Id': account.user_id, 'X-Semantier-Workspace-Id': account.workspace_id } })
      expect((await partialRead()).status).not.toBe(206)
      await page.getByRole('button', { name: 'Resume', exact: true }).click()
      await expect.poll(async () => {
        const response = await partialRead()
        return response.status === 206 ? response.text() : ''
      }, { timeout: 40_000 }).toBe('Explicit partial resume\n')
    }
    await page.getByRole('button', { name: 'Model service', exact: true }).click()
    await page.getByRole('button', { name: 'Accounts', exact: true }).click()
    await page.getByRole('button', { name: 'Turn preference off', exact: true }).click()
    await expect(page.getByText('Preference off.', { exact: false })).toBeVisible({ timeout: 20_000 })
    await page.getByRole('button', { name: 'File Sync', exact: true }).click()
    await expect(page.getByText(await fs.promises.realpath(folder), { exact: true })).toBeVisible()
    const afterOff = `native-off-${randomUUID()}.md`
    fs.writeFileSync(path.join(folder, afterOff), 'File Sync remains active\n')
    await expect.poll(async () => {
      const response = await fetch(origin + '/api/file-sync/files?download=true&path=' + afterOff, { headers: { cookie,
        'X-Semantier-User-Id': account.user_id, 'X-Semantier-Workspace-Id': account.workspace_id } })
      return response.status === 206 ? response.text() : ''
    }, { timeout: 40_000 }).toBe('File Sync remains active\n')
    await page.getByRole('button', { name: 'Pause', exact: true }).click()
    await expect(page.getByText('Paused', { exact: true })).toBeVisible()
    const bindingHeaders = { cookie, Origin: origin, 'X-Semantier-CSRF': '1' }
    const originalBinding = await fetch(origin + '/api/file-sync/binding', { method: 'POST', headers: bindingHeaders })
    expect(originalBinding.status).toBe(200)
    const originalCredential = await originalBinding.json()
    expect((await fetch(origin + '/api/file-sync/binding', { method: 'DELETE', headers: bindingHeaders })).status).toBe(204)
    expect((await page.evaluate(() => (window as any).hermesDesktop.connections.list())).primary).toBe(before.primary)
    const { page: restarted } = await fixture.restart()
    await expect(restarted.getByRole('button', { name: 'default', exact: true })).toHaveAttribute('aria-pressed', 'true', { timeout: 180_000 })
    expect((await restarted.evaluate(() => (window as any).hermesDesktop.connections.list())).primary).toBe('local')
    await restarted.getByRole('button', { name: 'Open settings', exact: true }).click()
    await restarted.getByRole('button', { name: 'File Sync', exact: true }).click()
    await expect(restarted.getByText(await fs.promises.realpath(folder), { exact: true })).toBeVisible({ timeout: 30_000 })
    await expect(restarted.getByText('Paused', { exact: true })).toBeVisible()
    await restarted.getByRole('button', { name: 'Resume', exact: true }).click()
    const reissuedBinding = await fetch(origin + '/api/file-sync/binding', { method: 'POST', headers: bindingHeaders })
    expect(reissuedBinding.status).toBe(200)
    const replacementCredential = await reissuedBinding.json()
    expect(replacementCredential.token).not.toBe(originalCredential.token)
    expect(replacementCredential.remoteRoot).toBe(originalCredential.remoteRoot)
    const afterRestart = `native-restart-${randomUUID()}.md`
    fs.writeFileSync(path.join(folder, afterRestart), 'Explicit resume after restart\n')
    await expect.poll(async () => {
      const response = await fetch(origin + '/api/file-sync/files?download=true&path=' + afterRestart, { headers: { cookie,
        'X-Semantier-User-Id': account.user_id, 'X-Semantier-Workspace-Id': account.workspace_id } })
      return response.status === 206 ? response.text() : ''
    }, { timeout: 40_000 }).toBe('Explicit resume after restart\n')
    await restarted.getByRole('button', { name: 'Pause', exact: true }).click()
    if (process.env.NATIVE_MODEL_E2E === '1') {
      await restarted.getByRole('button', { name: 'Close settings', exact: true }).click()
      const composer = restarted.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()
      await composer.fill('Reply with the fixture response.')
      await composer.press('Enter')
      await expect(restarted.getByText('fixture response', { exact: true })).toBeVisible({ timeout: 90_000 })
    }
  } finally {
    await fixture.close()
  }
})
