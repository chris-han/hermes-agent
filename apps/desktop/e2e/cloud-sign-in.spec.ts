/** Real Electron -> native IPC -> scoped OAuth -> local Cloud, without production accounts. */
import { execFileSync } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'

import { expect, test } from '@playwright/test'

import { writeEnvFile } from '../../../tests-js/scripts/mock-provider-config'

import { startCloudH3Ingress } from './cloud-h3-ingress'
import { DesktopFixtureSession } from './fixture-session'
import { buildAppEnv, createSandbox } from './fixtures'

const root = path.resolve(import.meta.dirname, '../../..')
const quote = (value: string) => "'" + value.replaceAll("'", "'\\''") + "'"

// eslint-disable-next-line no-empty-pattern -- Electron supplies its own page; only Playwright TestInfo is used.
test('default Cloud sign-in recovers, approves and persists through real Desktop IPC', async ({}, info) => {
  test.skip(
    !['linux', 'win32'].includes(process.platform) || !process.env.HERMES_DESKTOP_DEV_SERVER,
    'Requires WSL Electron development renderer and fixture-owned X display'
  )
  test.setTimeout(process.platform === 'win32' ? 420_000 : 240_000)
  expect(fs.existsSync(path.join(root, '.env')), 'Use a clean source worktree').toBe(false)
  const sandbox = createSandbox('cloud-sign-in')
  writeEnvFile(sandbox.hermesHome)
  fs.writeFileSync(path.join(sandbox.hermesHome, 'config.yaml'), 'language: en\nproviders: null\nbilling: null\n')
  let unavailable = true
  let approved = false
  let startCount = 0
  let claimCount = 0
  let ackCount = 0
  let intent: any
  let intentRef = 'synthetic-intent'
  let rejectFirstPoll = true
  let origin = ''
  const secret = 'synthetic-only-' + randomBytes(16).toString('hex')
  const requests: string[] = []

  const cloud = http.createServer(async (req, res) => {
    const route = req.url!.split('?')[0]
    requests.push(route)

    if (route === '/billing/connect') {
      res.setHeader('Content-Type', 'text/html')
      res.end(
        '<h1>Approve synthetic device</h1><form action="/approve" method="POST"><button>Approve device</button></form>'
      )

      return
    }

    if (route === '/approve') {
      approved = true
      res.end('Synthetic approval complete')

      return
    }

    let raw = ''

    for await (const chunk of req) {
      raw += chunk
    }

    const body = raw ? JSON.parse(raw) : {}
    res.setHeader('Content-Type', 'application/json')

    if (route === '/api/cloud-provider/device/start') {
      startCount++

      if (unavailable) {
        res.writeHead(503)
        res.end('{"detail":"private fixture diagnostic"}')

        return
      }

      intent = body
      intentRef = 'synthetic-intent-' + startCount
      approved = false
    }

    const binding = {
      runtime_ref: intent?.runtime_ref,
      profile_ref: intent?.profile_ref,
      generation: intent?.generation,
      account_id: 'synthetic-account',
      currency: 'USD'
    }

    const envelope: any = {
      contract_version: 'semantier-cloud-device-connection/v1',
      intent_ref: intentRef,
      expires_at: '2099-01-01T00:00:00+00:00',
      retry_after: 5,
      status: 'pending',
      data: {}
    }

    if (route === '/api/cloud-provider/device/start') {
      envelope.data.verification_uri = origin + '/billing/connect?intent_ref=' + intentRef
    } else if (route === `/api/cloud-provider/device/${intentRef}/poll`) {
      if (rejectFirstPoll) {
        rejectFirstPoll = false
        res.writeHead(410)
        res.end('{"detail":"expired"}')

        return
      }

      envelope.status = approved ? 'approved' : 'pending'
      envelope.data = binding
    } else if (route === `/api/cloud-provider/device/${intentRef}/claim`) {
      claimCount++
      envelope.status = 'claimed'
      envelope.data = {
        ...binding,
        claim_id: body.claim_id,
        organization_id: 'synthetic-org',
        credential: { token_id: 'synthetic-token', secret, expires_at: '2099-02-01T00:00:00+00:00' },
        inference_origin: origin,
        billing_origin: origin,
        catalog_path: '/api/cloud-provider/catalog',
        readiness_path: '/api/cloud-provider/device/readiness',
        installation_expires_at: '2099-01-01T00:00:00+00:00',
        renewal: 'reauthorization_required'
      }
    } else if (route === `/api/cloud-provider/device/${intentRef}/ack`) {
      ackCount++
      envelope.status = 'installed'
      envelope.data = { claim_id: body.claim_id, installed_generation: body.installed_generation }
    } else if (route === '/api/cloud-provider/catalog') {
      expect(req.headers.authorization).toBe('Bearer ' + secret)
      res.end(
        JSON.stringify({
          object: 'list',
          data: [
            {
              id: 'synthetic-model',
              revision: 'fixture-v1',
              context_window: 32768,
              capabilities: { reasoning: false },
              request_parameters: { reasoning_controls: {} }
            }
          ]
        })
      )

      return
    } else if (route === '/api/billing/commerce/state') {
      const amount = { amount: '100.00', currency: 'USD' }
      res.end(
        JSON.stringify({
          backend: 'semantier',
          schema_version: 'semantier.billing.v2',
          billing_account_id: 'synthetic-account',
          organization: { id: 'synthetic-org' },
          merchant: null,
          currency: 'USD',
          environment: 'DEVELOPMENT',
          subscription: null,
          credits: { allowance: amount, purchased: amount, available: amount, held: { amount: '0', currency: 'USD' } },
          usage: { known_cost: { amount: '0', currency: 'USD' }, period: 'lifetime' },
          capabilities: {
            can_read_usage: true,
            can_manage_billing: false,
            can_open_portal: false,
            can_charge: false,
            can_change_plan: false,
            can_auto_top_up: false
          },
          portal: { url: origin },
          as_of: '2026-11-10T00:00:00Z',
          account_provisioning: { local_status: 'READY', inference_ready: true, reason: null }
        })
      )

      return
    } else if (route === '/api/cloud-provider/device/readiness') {
      Object.assign(envelope, {
        account_id: 'synthetic-account',
        currency: 'USD',
        account_provisioning: 'ready',
        catalog: 'ready',
        device: 'ready'
      })
    } else {
      res.writeHead(404)
      res.end('{}')

      return
    }

    res.end(JSON.stringify(envelope))
  })

  await new Promise<void>(resolve => cloud.listen(0, '127.0.0.1', resolve))
  origin = `http://127.0.0.1:${(cloud.address() as any).port}`

  const h3 = await startCloudH3Ingress(
    sandbox.root,
    origin,
    process.env.HERMES_PYTHON || path.join(root, '.venv/bin/python')
  )

  if (h3) {
    origin = h3.origin
  }

  // Only the external Windows DPAPI boundary is replaced in this Linux fixture.
  // The runtime client, journal, scope, protocol, persistence and IPC remain real.
  const support = path.join(sandbox.root, 'python-fixture')
  fs.mkdirSync(support)
  const nativePython = process.env.HERMES_PYTHON || path.join(root, '.venv/bin/python')
  const wrapper = process.platform === 'win32' ? nativePython : path.join(support, 'python')
  const key = randomBytes(32).toString('base64url') + '='
  fs.writeFileSync(
    path.join(support, 'sitecustomize.py'),
    `
import os, sys
from pathlib import Path
if Path(os.environ.get('HERMES_HOME', '/nonexistent')).resolve() == Path(${JSON.stringify(sandbox.hermesHome)}):
    from cryptography.fernet import Fernet
    from hermes_cli import cloud_device_client as device
    cipher = Fernet(${JSON.stringify(key)}.encode())
    device._protect = cipher.encrypt
    device._unprotect = cipher.decrypt
    device.DEFAULT_CLOUD_ORIGIN = ${JSON.stringify(origin)}
    sys.executable = ${JSON.stringify(wrapper)}
    with open(${JSON.stringify(path.join(support, 'adapter-loaded.log'))}, 'a') as trace:
        trace.write(str(os.getpid()) + '\\n')
`
  )
  const python = process.env.HERMES_PYTHON || path.join(root, '.venv/bin/python')

  if (process.platform !== 'win32') {
    fs.writeFileSync(
      wrapper,
      `#!/bin/sh\nexport PYTHONPATH=${quote(support + ':' + root)}\nexec ${quote(python)} "$@"\n`,
      { mode: 0o700 }
    )
  }

  if (process.platform === 'win32') {
    fs.writeFileSync(
      path.join(sandbox.hermesHome, 'config.yaml'),
      'language: en\ndisplay:\n  language: en\nbilling: null\nproviders:\n  semantier-cloud:\n    base_url: ' +
        origin +
        '/v1\n    portal_origin: ' +
        origin +
        '\n' +
        (h3 ? '    http_version: "3"\n    ssl_ca_cert: ' + JSON.stringify(h3.certificate) + '\n' : '')
    )
  }

  const env = buildAppEnv(sandbox, {
    HERMES_DESKTOP_DEV_SERVER: process.env.HERMES_DESKTOP_DEV_SERVER!,
    HERMES_DESKTOP_CDP_PORT: 'off',
    HERMES_PYTHON: wrapper,
    HERMES_DESKTOP_PYTHON: wrapper
  })

  const fixture = new DesktopFixtureSession(sandbox, env, info)

  try {
    const launched = await fixture.launch()
    let app = launched.app
    const page = launched.page
    await fixture.stage('desktop.ready', async () => {
      await expect(page.getByRole('button', { name: 'default', exact: true })).toHaveAttribute('aria-pressed', 'true', {
        timeout: process.platform === 'win32' ? 180_000 : 90_000
      })
      // Use the actual first-run control, just as a user does. No store injection.
      await page
        .getByRole('button', { name: 'Sign in to Semantier Model Provider', exact: true })
        .click({ timeout: 60_000 })
      await expect(page.getByRole('button', { name: 'Try again', exact: true })).toBeVisible({ timeout: 60_000 })
      await expect(page.getByText(/Semantier Cloud could not complete sign-in/)).toBeVisible()
      await expect(page.getByRole('button', { name: 'Use an API key', exact: true })).toBeVisible()
      expect(startCount).toBe(1)
    })
    unavailable = false
    await app.evaluate(({ ipcMain }, expected) => {
      ;(globalThis as any).__cloudVerificationUrl = null
      // WSL's OS adapter uses cmd.exe, so replace only this external-open boundary.
      ipcMain.removeHandler('hermes:openExternal')
      ipcMain.handle('hermes:openExternal', async (_event: unknown, url: string) => {
        if (!url.startsWith(expected + '/billing/connect?')) {
          throw new Error('Unexpected external origin')
        }

        ;(globalThis as any).__cloudVerificationUrl = url
      })
    }, origin)
    await page.getByRole('button', { name: 'Try again', exact: true }).click()
    await expect.poll(() => app!.evaluate(() => (globalThis as any).__cloudVerificationUrl)).toBeTruthy()
    // Approval is an external service boundary; submit its local HTML form.
    await fixture.stage('cloud.authorize', async () => {
      const { chromium } = await import('@playwright/test')
      const browser = await chromium.launch({ channel: 'chromium', headless: true })

      try {
        const consent = await browser.newPage({ ignoreHTTPSErrors: Boolean(h3) })
        await consent.goto(await app.evaluate(() => (globalThis as any).__cloudVerificationUrl))
        await consent.getByRole('button', { name: 'Approve device' }).click()
        await expect(consent.getByText('Synthetic approval complete')).toBeVisible()
        await expect(
          page.getByText('This sign-in session expired. Try again to open a new authorization page.', { exact: true })
        ).toBeVisible({ timeout: 30_000 })
        const retiredUrl = await app.evaluate(() => (globalThis as any).__cloudVerificationUrl)
        await page.getByRole('button', { name: 'Try again', exact: true }).click()
        await expect.poll(() => app!.evaluate(() => (globalThis as any).__cloudVerificationUrl)).not.toBe(retiredUrl)
        await consent.goto(await app.evaluate(() => (globalThis as any).__cloudVerificationUrl))
        await consent.getByRole('button', { name: 'Approve device' }).click()
        await expect(consent.getByText('Synthetic approval complete')).toBeVisible()
      } finally {
        await browser.close()
      }
    })
    await fixture.stage('cloud.confirm', async () => {
      // Assert the rendered journey rather than importing a second HMR store instance.
      const model = page.getByRole('button', { name: /Model.*synthetic-model/ })
      await expect
        .poll(async () => (await fixture.button('Begin').isVisible()) || (await model.isVisible()), { timeout: 60_000 })
        .toBe(true)

      if (await fixture.button('Begin').isVisible()) {
        await fixture.button('Begin').click()
      }
      await expect(model).toBeVisible()
      await expect(fixture.button('Begin')).not.toBeVisible()
      await expect(page.getByRole('button', { name: 'Try again', exact: true })).not.toBeVisible()
      await expect(
        page.getByRole('button', { name: 'Sign in to Semantier Model Provider', exact: true })
      ).not.toBeVisible()
    })
    expect(startCount).toBe(3)
    expect(claimCount).toBe(1)
    expect(ackCount).toBe(1)
    const config = fs.readFileSync(path.join(sandbox.hermesHome, 'config.yaml'), 'utf8')
    expect(config).toContain('credential_ref: synthetic-token')
    expect(config).not.toContain(secret)
    expect(fs.readFileSync(path.join(sandbox.hermesHome, '.cloud-device.bin')).includes(Buffer.from(secret))).toBe(
      false
    )
    expect(fs.existsSync(path.join(sandbox.hermesHome, 'profiles/semantier-local'))).toBe(false)
    // Same lifecycle helper stops the owning gateway before relaunching the same home.
    const restarted = await fixture.restart()
    app = restarted.app

    const recovered = execFileSync(
      wrapper,
      [
        '-c',
        `from hermes_cli.cloud_device_client import credential; assert credential('synthetic-token', base_url=${JSON.stringify(origin + '/v1')}); print('fixture-credential-recovered')`
      ],
      { cwd: root, env, timeout: 20_000, encoding: 'utf8' }
    )

    expect(recovered.trim()).toBe('fixture-credential-recovered')
    const reopened = restarted.page
    await expect(reopened.getByRole('button', { name: 'default', exact: true })).toHaveAttribute(
      'aria-pressed',
      'true',
      { timeout: process.platform === 'win32' ? 180_000 : 90_000 }
    )
    await expect(reopened.getByRole('button', { name: /synthetic-model/ })).toBeVisible({ timeout: 60_000 })
    await expect(
      reopened.getByRole('button', { name: 'Sign in to Semantier Model Provider', exact: true })
    ).not.toBeVisible()

    if (h3) {
      const entries = fs
        .readFileSync(h3.log, 'utf8')
        .trim()
        .split('\n')
        .map(line => JSON.parse(line))

      for (const operation of ['start', 'poll', 'claim', 'ack']) {
        expect(
          entries.some(entry => entry.request?.proto === 'HTTP/3.0' && entry.request?.uri?.endsWith('/' + operation))
        ).toBe(true)
      }
    }

    expect(startCount).toBe(3)
    expect(claimCount).toBe(1)
    expect(fs.existsSync(path.join(sandbox.hermesHome, 'profiles/semantier-local'))).toBe(false)
    console.log(
      'CLOUD_DEV_E2E',
      JSON.stringify({
        profile: 'default',
        startCount,
        claimCount,
        ackCount,
        expiredSessionRecovery: 'PASS',
        restart: 'PASS',
        transport: h3 ? 'http3' : 'local-http',
        protectedStore: process.platform === 'win32' ? 'windows-native-dpapi' : 'fixture-fernet-not-windows-dpapi',
        requests
      })
    )
  } finally {
    console.log(
      'FIXTURE_ADAPTER',
      fs.existsSync(path.join(support, 'adapter-loaded.log'))
        ? fs.readFileSync(path.join(support, 'adapter-loaded.log'), 'utf8').trim()
        : 'NOT_LOADED'
    )

    try {
      await fixture.close()
    } finally {
      await h3?.close()
      await new Promise<void>(resolve => cloud.close(() => resolve()))
    }
  }
})
