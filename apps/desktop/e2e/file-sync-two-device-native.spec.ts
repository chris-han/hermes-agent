/** Two isolated Windows source-Electron applications, one disposable account. */
import { createHash, randomUUID } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { expect, test, type ElectronApplication, type Page } from '@playwright/test'

import { DesktopFixtureSession } from './fixture-session'
import { buildAppEnv, createSandbox } from './fixtures'

// eslint-disable-next-line no-empty-pattern -- Electron supplies its own pages.
test('two Windows devices require separate File Sync consent and stop independently', async ({}, info) => {
  test.skip(process.platform !== 'win32' || !process.env.FILEX_PORTABLE_FIXTURE || !process.env.HERMES_DESKTOP_DEV_SERVER,
    'Requires the isolated connected Windows source-Electron fixture')
  test.setTimeout(420_000)
  const fixturePath = process.env.FILEX_PORTABLE_FIXTURE!
  const account = JSON.parse(fs.readFileSync(fixturePath, 'utf8'))
  const origin: string = JSON.parse(fs.readFileSync(path.resolve(path.dirname(fixturePath), '..', 'services.json'), 'utf8')).web
  const sandboxes = [createSandbox('file-sync-device-a'), createSandbox('file-sync-device-b')]
  const folders = sandboxes.map(sandbox => {
    const folder = path.join(sandbox.root, 'documents')
    fs.mkdirSync(folder)
    fs.writeFileSync(path.join(sandbox.hermesHome, 'config.yaml'), 'language: en\ndisplay:\n  language: en\n')
    return folder
  })
  const fixtures = sandboxes.map(sandbox => new DesktopFixtureSession(sandbox, buildAppEnv(sandbox, {
    HERMES_PYTHON: process.env.HERMES_PYTHON!, HERMES_DESKTOP_PYTHON: process.env.HERMES_PYTHON!,
    HERMES_DESKTOP_DEV_SERVER: process.env.HERMES_DESKTOP_DEV_SERVER!, HERMES_DESKTOP_CDP_PORT: 'off'
  }), info))

  async function signIn(app: ElectronApplication, page: Page) {
    await expect(page.getByRole('button', { name: 'default', exact: true })).toHaveAttribute('aria-pressed', 'true', { timeout: 180_000 })
    expect((await page.evaluate(() => (window as any).hermesDesktop.connections.list())).primary).toBe('local')
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
    await expect(page.getByText('Fixture', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.getByRole('button', { name: 'File Sync', exact: true }).click()
    await expect(page.getByText('No folder selected', { exact: true })).toBeVisible()
  }

  async function choose(app: ElectronApplication, page: Page, folder: string) {
    await app.evaluate(({ dialog }, selected) => {
      ;(dialog as any).showOpenDialog = async () => ({ canceled: false, filePaths: [selected] })
      ;(dialog as any).showMessageBox = async () => ({ response: 1, checkboxChecked: false })
    }, folder)
    await page.getByRole('button', { name: 'Choose and enable', exact: true }).click()
    await expect(page.getByText(await fs.promises.realpath(folder), { exact: true })).toBeVisible({ timeout: 60_000 })
  }

  try {
    const a = await fixtures[0].launch()
    await signIn(a.app, a.page)
    await choose(a.app, a.page, folders[0])
    const first = `device-a-${randomUUID()}.md`
    fs.writeFileSync(path.join(folders[0], first), 'Device A fixture bytes\n')
    const partition = 'persist:semantier-workspace-' + createHash('sha256').update(origin).digest('hex')
    const cookies = await a.app.evaluate(async ({ session }, name) => session.fromPartition(name).cookies.get({}), partition)
    const cookie = cookies.map((value: { name: string; value: string }) => `${value.name}=${value.value}`).join('; ')
    const read = async (name: string) => {
      const response = await fetch(origin + '/api/file-sync/files?download=true&path=' + name, { headers: { cookie,
        'X-Semantier-User-Id': account.user_id, 'X-Semantier-Workspace-Id': account.workspace_id } })
      return response.status === 206 ? response.text() : ''
    }
    await expect.poll(() => read(first), { timeout: 40_000 }).toBe('Device A fixture bytes\n')
    const b = await fixtures[1].launch()
    await signIn(b.app, b.page)
    expect(await fs.promises.readdir(folders[1])).toEqual([])
    await choose(b.app, b.page, folders[1])
    await expect.poll(() => fs.promises.readFile(path.join(folders[1], first), 'utf8').catch(() => ''), { timeout: 40_000 })
      .toBe('Device A fixture bytes\n')
    await a.page.getByRole('button', { name: 'Pause', exact: true }).click()
    await expect(a.page.getByText('Paused', { exact: true })).toBeVisible()
    const second = `device-b-${randomUUID()}.md`
    fs.writeFileSync(path.join(folders[1], second), 'Device B fixture bytes\n')
    await expect.poll(() => read(second), { timeout: 40_000 }).toBe('Device B fixture bytes\n')
    await new Promise(resolve => setTimeout(resolve, 6500))
    expect(fs.existsSync(path.join(folders[0], second))).toBe(false)
    expect((await a.page.evaluate(() => (window as any).hermesDesktop.connections.list())).primary).toBe('local')
    expect((await b.page.evaluate(() => (window as any).hermesDesktop.connections.list())).primary).toBe('local')
    await b.page.getByRole('button', { name: 'Pause', exact: true }).click()
    await expect(b.page.getByText('Paused', { exact: true })).toBeVisible()
    await b.page.getByRole('button', { name: 'Model service', exact: true }).click()
    await b.page.getByRole('button', { name: 'Accounts', exact: true }).click()
    await b.page.getByRole('button', { name: 'Sign out', exact: true }).click()
    await expect(b.page.getByRole('button', { name: 'Sign in', exact: true })).toBeVisible({ timeout: 30_000 })
    await b.page.getByLabel('Workspace address').fill(origin)
    const otherPopupPromise = b.app.waitForEvent('window')
    await b.page.getByRole('button', { name: 'Sign in', exact: true }).click()
    const otherPopup = await otherPopupPromise
    await otherPopup.getByLabel('Login').fill(account.other_login)
    await otherPopup.getByLabel('Password').fill(account.password)
    await otherPopup.getByRole('button', { name: 'Sign in fixture', exact: true }).click()
    await expect(b.page.getByText('Other', { exact: true })).toBeVisible({ timeout: 60_000 })
    await b.page.getByRole('button', { name: 'File Sync', exact: true }).click()
    await expect(b.page.getByText('No folder selected', { exact: true })).toBeVisible()
    await expect(b.page.getByText(await fs.promises.realpath(folders[1]), { exact: true })).toHaveCount(0)
    await b.page.getByRole('button', { name: 'Choose and enable', exact: true }).click()
    await expect(b.page.getByRole('alert')).toContainText(/another account or folder binding/i)
    expect(await fs.promises.readFile(path.join(folders[1], second), 'utf8')).toBe('Device B fixture bytes\n')
  } finally {
    await fixtures[1].close()
    await fixtures[0].close()
  }
})
