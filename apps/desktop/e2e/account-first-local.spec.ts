/** Isolated source Electron: synthetic account login, real IPC and local gateway. */
import { createHash, randomBytes } from 'node:crypto'
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'

import { expect, test } from '@playwright/test'

import { DesktopFixtureSession } from './fixture-session'
import { buildAppEnv, createSandbox } from './fixtures'

// eslint-disable-next-line no-empty-pattern -- Electron supplies its own page.
test('one account login preserves the local gateway and leaves capabilities off', async ({}, info) => {
  test.skip(!['linux', 'win32'].includes(process.platform) || !process.env.HERMES_DESKTOP_DEV_SERVER, 'Requires the source-Electron fixture')
  test.setTimeout(process.platform === 'win32' ? 300_000 : 180_000)
  const requests: string[] = []
  const server = http.createServer((req, res) => {
    const pathname = new URL(req.url!, 'http://fixture').pathname
    requests.push(req.method + ' ' + pathname)
    if (pathname === '/login' && req.method === 'POST') {
      res.writeHead(303, { 'Set-Cookie': 'fixture-session=authorized; HttpOnly; SameSite=Lax; Path=/', Location: '/' })
      res.end(); return
    }
    if (pathname === '/') {
      res.writeHead(200, { 'content-type': 'text/html' })
      res.end('<form method="post" action="/login"><button>Sign in fixture</button></form>'); return
    }
    res.setHeader('content-type', 'application/json')
    if (pathname === '/auth/context') {
      const authenticated = (req.headers.cookie || '').includes('fixture-session=authorized')
      res.end(JSON.stringify(authenticated ? { authenticated: true,
        user: { user_id: 'fixture-user', workspace_id: 'fixture-workspace', name: 'Fixture' },
        workspace: { id: 'fixture-workspace' }, organization_id: 'fixture-org' } : { authenticated: false }))
      return
    }
    if (pathname === '/api/winman/capability') {
      res.end(JSON.stringify({ desired: { version: 1, enabled: false, revision: 0 }, execution: 'unavailable' })); return
    }
    res.writeHead(503); res.end('{}')
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  const address = server.address()
  if (!address || typeof address === 'string') {throw new Error('Fixture origin unavailable')}
  const connected = process.env.ACCOUNT_MODEL_E2E_FIXTURE
    ? JSON.parse(fs.readFileSync(process.env.ACCOUNT_MODEL_E2E_FIXTURE, 'utf8')) : null
  const origin = connected?.origin || `http://127.0.0.1:${address.port}`
  const sandbox = createSandbox('account-first-local')
  fs.writeFileSync(sandbox.hermesHome + '/config.yaml', 'language: en\ndisplay:\n  language: en\nauxiliary:\n  title_generation:\n    model_upgrade_enabled: false\n')
  const root = path.resolve(import.meta.dirname, '../../..')
  const support = path.join(sandbox.root, 'python-fixture')
  fs.mkdirSync(support)
  const wrapper = process.platform === 'win32' ? process.env.HERMES_PYTHON! : path.join(support, 'python')
  if (process.platform === 'linux') {
    const key = randomBytes(32).toString('base64url') + '='
    fs.writeFileSync(path.join(support, 'sitecustomize.py'), `
import os, sys
from pathlib import Path
if Path(os.environ.get('HERMES_HOME', '/nonexistent')).resolve() == Path(${JSON.stringify(sandbox.hermesHome)}):
    from cryptography.fernet import Fernet
    from hermes_cli import cloud_device_client as device
    cipher = Fernet(${JSON.stringify(key)}.encode())
    device._protect = cipher.encrypt
    device._unprotect = cipher.decrypt
    sys.executable = ${JSON.stringify(wrapper)}
`)
    const quote = (value: string) => "'" + value.replaceAll("'", "'\\''") + "'"
    fs.writeFileSync(wrapper, `#!/bin/sh\nexport PYTHONPATH=${quote(support + ':' + root)}\nexec ${quote(process.env.HERMES_PYTHON!)} "$@"\n`, { mode: 0o700 })
  }
  const fixture = new DesktopFixtureSession(sandbox, buildAppEnv(sandbox, {
    HERMES_PYTHON: wrapper, HERMES_DESKTOP_PYTHON: wrapper, HERMES_DESKTOP_DEV_SERVER: process.env.HERMES_DESKTOP_DEV_SERVER!,
    HERMES_DESKTOP_CDP_PORT: 'off'
  }), info)
  try {
    const { app, page } = await fixture.launch()
    page.on('pageerror', error => console.log('FIXTURE_RENDERER_ERROR', error.message))
    let failures = 0
    page.on('requestfailed', request => {
      if (failures++ < 5) {
        const target = new URL(request.url())
        console.log('FIXTURE_RESOURCE_FAILURE', target.origin === process.env.HERMES_DESKTOP_DEV_SERVER,
          target.pathname.slice(0, 120), request.failure()?.errorText)
      }
    })
    await fixture.stage('desktop.ready', async () => {
      try {
        await expect(page.getByRole('button', { name: 'default', exact: true })).toHaveAttribute('aria-pressed', 'true', {
          timeout: process.platform === 'win32' ? 180_000 : 90_000
        })
      } catch (error) {
        const state = await page.evaluate(() => ({ ready: document.readyState, rootChildren: document.getElementById('root')?.childElementCount ?? -1 }))
        console.log('FIXTURE_RENDERER_STATE', JSON.stringify(state), 'failedRequests', failures)
        throw error
      }
    })
    const localBefore = await page.evaluate(() => (window as any).hermesDesktop.connections.list())
    expect(localBefore.primary).toBe('local')
    const partition = 'persist:semantier-workspace-' + createHash('sha256').update('https://cloud.semantier.cn').digest('hex')
    await app.evaluate(({ session }, partitionName) => {
      session.fromPartition(partitionName).webRequest.onBeforeRequest({ urls: ['https://cloud.semantier.cn/*'] },
        (_details: unknown, callback: (value: { cancel: boolean }) => void) => callback({ cancel: true }))
    }, partition)
    await page.getByRole('button', { name: 'Open settings', exact: true }).click()
    await page.getByRole('button', { name: 'Model service', exact: true }).click()
    await page.getByRole('button', { name: 'Accounts', exact: true }).click()
    await expect(page.getByText('Semantier Account', { exact: true })).toBeVisible()
    await page.getByLabel('Workspace address').fill(origin)
    await expect(page.getByText('Sign in to use eligible account services.', { exact: true })).toBeVisible()
    const popupPromise = app.waitForEvent('window')
    await page.getByRole('button', { name: 'Sign in', exact: true }).click()
    const popup = await popupPromise
    if (connected) {
      await popup.getByLabel('Login').fill(connected.login)
      await popup.getByLabel('Password').fill(connected.password)
    }
    await popup.getByRole('button', { name: 'Sign in fixture', exact: true }).click()
    if (connected?.refusal) {
      await expect.poll(() => page.evaluate(() => (window as any).hermesDesktop.workspace.modelStatus()),
        { timeout: 90_000 }).toMatchObject({ state: 'denied', issue: connected.foreign_membership_denial ? undefined : connected.refusal })
      const message = connected.foreign_membership_denial ? 'Access denied for this account.'
        : 'The billing account is unavailable or does not match this organization.'
      await expect(page.getByText(message, { exact: true })).toBeVisible()
      await expect(page.getByRole('button', { name: 'Sign in', exact: true })).toHaveCount(0)
      expect(await page.evaluate(() => (window as any).hermesDesktop.workspace.modelDefault())).toBeNull()
    } else if (connected) {
      await expect.poll(() => page.evaluate(() => (window as any).hermesDesktop.workspace.modelDefault()),
        { timeout: 90_000 }).toMatchObject({ model: 'deepseek-v4.1-flash-1017' })
    } else {
      await expect(page.getByText('Fixture', { exact: true })).toBeVisible({ timeout: 20_000 })
    }
    await expect(page.getByText('Preference off.', { exact: false })).toBeVisible({ timeout: 20_000 })
    if (connected) {
      await page.getByRole('button', { name: 'Request enablement', exact: true }).click()
      await page.getByRole('button', { name: 'Confirm request', exact: true }).click()
      await expect(page.getByText('Enablement requested.', { exact: false })).toBeVisible({ timeout: 20_000 })
      await page.getByRole('button', { name: 'Turn preference off', exact: true }).click()
      await expect(page.getByText('Preference off.', { exact: false })).toBeVisible({ timeout: 20_000 })
    }
    await page.getByRole('button', { name: 'File Sync', exact: true }).click()
    await expect(page.getByText('No folder selected', { exact: true })).toBeVisible()
    await page.getByRole('button', { name: 'Close settings', exact: true }).click()
    await expect(page.getByRole('button', { name: 'default', exact: true })).toHaveAttribute('aria-pressed', 'true')
    expect((await page.evaluate(() => (window as any).hermesDesktop.connections.list())).primary).toBe(localBefore.primary)
    if (connected && process.env.ACCOUNT_MODEL_E2E_PROMPT === '1') {
      const composer = page.locator('[data-slot="composer-root"] [contenteditable="true"]').filter({ visible: true }).first()
      await composer.fill('Reply with the fixture response.')
      await composer.press('Enter')
      await expect(page.getByText('fixture response', { exact: true })).toBeVisible({ timeout: 90_000 })
    }
    if (!connected) {
      expect(requests.filter(request => request === 'POST /login')).toHaveLength(1)
      expect(requests.some(request => /\/api\/(?:file-sync\/binding|winman\/messages)/.test(request))).toBe(false)
    }
  } finally {
    await fixture.close()
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()))
  }
})
