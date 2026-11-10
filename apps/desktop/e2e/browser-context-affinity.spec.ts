/** Real Electron webview + real gateway; deterministic provider calls, not LLM selection evidence. */
import { execFile } from 'node:child_process'
import fs from 'node:fs'
import http from 'node:http'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { expect, test } from '@playwright/test'

import { waitForChatReady } from '../../../tests-js/scripts/desktop-chat-smoke'
import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'

import { DesktopFixtureSession } from './fixture-session'
import { buildAppEnv, createSandbox } from './fixtures'

const root = path.resolve(import.meta.dirname, '../../..')

// eslint-disable-next-line no-empty-pattern -- Electron supplies its own page; only Playwright TestInfo is used.
test('authenticated owned Electron page remains distinct from independent automation', async ({}, info) => {
  test.skip(
    process.platform !== 'linux' || !process.env.HERMES_DESKTOP_DEV_SERVER,
    'WSL source-renderer lane requires HERMES_DESKTOP_DEV_SERVER and local browser tools'
  )
  // gateway.run also loads the source checkout's dotenv. Never let this synthetic
  // journey hydrate developer messaging credentials or auto-install adapters.
  expect(
    fs.existsSync(path.join(root, '.env')),
    'Run this lane from a clean source worktree without a project .env'
  ).toBe(false)
  test.setTimeout(240_000)
  const results: { name: string; content: string }[] = []
  const requests: any[] = []
  let clicks = 0

  const site = http.createServer((req, res) => {
    if (req.url === '/login') {
      res.writeHead(302, {
        'Set-Cookie': 'fixture_session=synthetic; HttpOnly; SameSite=Lax; Path=/',
        Location: '/account'
      })
      res.end()

      return
    }

    const signedIn = req.headers.cookie?.includes('fixture_session=synthetic')

    if (req.url === '/action' && signedIn) {
      clicks++
      res.end('done')

      return
    }

    res.setHeader('Content-Type', 'text/html')
    res.end(
      signedIn
        ? "<h1>Synthetic account signed in</h1><button id=\"harmless\" onclick=\"fetch('/action').then(()=>document.querySelector('output').textContent='clicked')\">Harmless action</button><output>unchanged</output>"
        : '<h1>Sign in required</h1>'
    )
  })

  await new Promise<void>(resolve => site.listen(0, '127.0.0.1', resolve))
  const siteUrl = `http://127.0.0.1:${(site.address() as any).port}`

  const provider = http.createServer(async (req, res) => {
    if (req.method === 'GET') {
      res.setHeader('Content-Type', 'application/json')
      res.end(JSON.stringify({ data: [{ id: 'mock-model', object: 'model' }] }))

      return
    }

    let raw = ''

    for await (const chunk of req) {
      raw += chunk
    }

    const body = JSON.parse(raw)
    requests.push(body)
    const messages = body.messages || []

    const readTurn = messages.findLastIndex(
      (m: any) => m.role === 'user' && JSON.stringify(m.content).includes('AFFINITY_READ')
    )

    const lastUser = readTurn >= 0 ? readTurn : messages.findLastIndex((m: any) => m.role === 'user')
    const prompt = JSON.stringify(messages[lastUser]?.content || '')
    const toolResults = messages.slice(lastUser + 1).filter((m: any) => m.role === 'tool')

    for (const m of toolResults) {
      if (!results.some(r => r.name === (m.name || m.tool_call_id))) {
        results.push({ name: m.name || m.tool_call_id, content: String(m.content) })
      }
    }

    const steps = [
      ['desktop_preview', { action: 'read' }],
      ['drive_preview', { action: 'elements' }],
      ['drive_preview', { action: 'click', selector: '#harmless' }]
    ] as const

    const step = prompt.includes('AFFINITY_READ') ? steps[toolResults.length] : undefined

    const message = step
      ? {
          role: 'assistant',
          content: null,
          tool_calls: [
            {
              id: `affinity-${toolResults.length}`,
              type: 'function',
              function: { name: 'tool_call', arguments: JSON.stringify({ name: step[0], arguments: step[1] }) }
            }
          ]
        }
      : { role: 'assistant', content: prompt.includes('AFFINITY_READ') ? 'AFFINITY_DONE' : 'AFFINITY_READY' }

    const finish = step ? 'tool_calls' : 'stop'

    if (body.stream) {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' })
      const delta = step ? { role: 'assistant', tool_calls: [{ index: 0, ...message.tool_calls![0] }] } : message
      res.write(
        `data: ${JSON.stringify({ id: 'fixture', object: 'chat.completion.chunk', created: 0, model: body.model, choices: [{ index: 0, delta, finish_reason: step ? finish : null }] })}\n\n`
      )

      if (step) {
        res.end('data: [DONE]\n\n')

        return
      }

      res.end(
        `data: ${JSON.stringify({ id: 'fixture', object: 'chat.completion.chunk', created: 0, model: body.model, choices: [{ index: 0, delta: {}, finish_reason: finish }] })}\n\ndata: [DONE]\n\n`
      )
    } else {
      res.setHeader('Content-Type', 'application/json')
      res.end(
        JSON.stringify({
          id: 'fixture',
          choices: [{ index: 0, message, finish_reason: finish }],
          usage: { prompt_tokens: 10, completion_tokens: 10, total_tokens: 20 }
        })
      )
    }
  })

  await new Promise<void>(resolve => provider.listen(0, '127.0.0.1', resolve))
  const sandbox = createSandbox('browser-affinity')
  const providerUrl = `http://127.0.0.1:${(provider.address() as any).port}`
  writeMockProviderConfig(
    sandbox.hermesHome,
    providerUrl,
    'language: en',
    'agent:\n  verify_on_stop: false\nbrowser:\n  backend: browser-use\n  cloud_provider: local\n  engine: chrome\n  allow_private_urls: true\n  use_real_profile: false\n'
  )
  writeEnvFile(sandbox.hermesHome)

  const env = buildAppEnv(sandbox, {
    HERMES_DESKTOP_DEV_SERVER: process.env.HERMES_DESKTOP_DEV_SERVER!,
    HERMES_DESKTOP_CDP_PORT: 'off',
    HERMES_PYTHON: process.env.HERMES_PYTHON || path.join(root, '.venv/bin/python')
  })

  const fixture = new DesktopFixtureSession(sandbox, env, info)

  try {
    const { page } = await fixture.launch()
    await fixture.stage('browser.affinity', async () => {
      page.on('console', msg => {
        if (msg.type() === 'error') {
          console.log('RENDERER_ERROR', msg.text().slice(0, 350))
        }
      })
      const composer = await waitForChatReady(page)
      await composer.fill('AFFINITY_INIT')
      await composer.press('Enter')
      await expect(page.getByText('AFFINITY_READY', { exact: true })).toBeVisible({ timeout: 60_000 })
      await page.getByRole('button', { name: 'No thanks', exact: true }).click()
      await page.evaluate(async url => {
        const { openPreview } = await import('/src/store/' + 'preview.ts')
        openPreview({ kind: 'url', source: 'user', label: 'Synthetic authenticated page', url })
      }, siteUrl + '/login')
      const guest = page.locator('webview').first()
      await guest.waitFor()
      await expect
        .poll(async () => guest.evaluate((el: any) => el.executeJavaScript('document.body.innerText')).catch(() => ''))
        .toContain('Synthetic account signed in')
      await page.getByRole('button', { name: 'Not now', exact: true }).click()
      console.log('W0_AUTHENTICATED_WEBVIEW_READY')

      const owner = await page.evaluate(async () => {
        const preview = await import('/src/store/' + 'preview.ts')
        const session = await import('/src/store/' + 'session.ts')

        return {
          runtime: session.$activeSessionId.get(),
          stored: session.$selectedStoredSessionId.get(),
          tabs: preview.$previewTabs.get().map((t: any) => ({ id: t.id, sessionId: t.sessionId, profile: t.profile }))
        }
      })

      await composer.click()
      await composer.pressSequentially('AFFINITY_READ this authenticated right-side page')
      await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeEnabled()
      await page.getByRole('button', { name: 'Send', exact: true }).click()
      await expect(page.getByText('AFFINITY_DONE', { exact: true })).toBeVisible({ timeout: 60_000 })
      console.log(
        'AFFINITY_RECEIPT',
        JSON.stringify({ owner, results, clicks, scriptedProvider: true, autonomousToolSelection: 'NOT_RUN' })
      )
      expect(results[0]?.content).toContain('Synthetic account signed in')
      expect(results[1]?.content).toContain('Harmless action')
      expect(clicks).toBe(1)
      await expect
        .poll(() => guest.evaluate((el: any) => el.executeJavaScript('document.body.innerText')))
        .toContain('clicked')

      const independent = await promisify(execFile)(
        env.HERMES_PYTHON,
        [path.join(root, 'tests-js/scripts/browser-affinity-independent.py'), siteUrl],
        {
          cwd: root,
          timeout: 120_000,
          env: {
            ...env,
            PYTHONPATH: root,
            HERMES_RUNTIME_DIR: process.env.HERMES_RUNTIME_DIR || path.join(os.homedir(), '.hermes/tools')
          }
        }
      )

      expect(independent.stdout).toContain('"named_session_isolation": "PASS"')
      console.log('INDEPENDENT_BROWSER_EXEC', independent.stdout.trim())
      await expect
        .poll(() => guest.evaluate((el: any) => el.executeJavaScript('document.body.innerText')))
        .toContain('clicked')
      expect(clicks).toBe(1)

      const guestId = await guest.evaluate((el: any) => el.getWebContentsId())
      await page.evaluate(async () => (await import('/src/store/' + 'layout.ts')).toggleRightSide())
      expect(await guest.evaluate((el: any) => el.getWebContentsId())).toBe(guestId)
      await page.evaluate(async () => (await import('/src/store/' + 'layout.ts')).toggleRightSide())
      await expect
        .poll(() => guest.evaluate((el: any) => el.executeJavaScript('document.body.innerText')))
        .toContain('clicked')

      // Real socket loss/resume; a new turn must bind to the same mounted page.
      await page.evaluate(async () =>
        (await import('/src/store/' + 'gateway.ts')).activeGateway().invalidate('synthetic reconnect')
      )
      await expect
        .poll(() =>
          page.evaluate(async () => (await import('/src/store/' + 'gateway.ts')).activeGateway()?.connectionState)
        )
        .toBe('open')
      results.length = 0
      await composer.fill('AFFINITY_READ_RECONNECT this same authenticated pane')
      await page.getByRole('button', { name: 'Send', exact: true }).click()
      await expect(page.getByText('AFFINITY_DONE', { exact: true })).toHaveCount(2, { timeout: 60_000 })
      expect(results[0]?.content).toContain('Synthetic account signed in')
      expect(clicks).toBe(2)
      expect(await guest.evaluate((el: any) => el.getWebContentsId())).toBe(guestId)

      // Exercise the production request owner and live-target registries in this real renderer.
      const boundaries = await page.evaluate(async owner => {
        const { handleServerRequest } = await import(
          '/src/app/session/hooks/use-message-stream/gateway-event/' + 'server-requests.ts'
        )

        const { actOnActivePreview } = await import('/src/app/chat/right-rail/' + 'preview-act.ts')
        const { readActivePreview } = await import('/src/app/chat/right-rail/' + 'preview-reader.ts')
        const preview = await import('/src/store/' + 'preview.ts')
        let declined = false
        let answered = false
        handleServerRequest(
          {
            id: 'synthetic-foreign',
            method: 'preview.act',
            profile: 'default',
            params: { session_id: 'foreign-session', action: 'click', selector: '#harmless' },
            decline: () => {
              declined = true
            },
            respond: () => {
              answered = true
            },
            fail: () => {
              answered = true
            }
          },
          { activeSessionIdRef: { current: owner.runtime }, sessionStateByRuntimeIdRef: { current: new Map() } },
          owner.runtime
        )
        const foreignRead = await readActivePreview({}, { sessionId: 'foreign-session', profile: 'default' })
        preview.setPreviewTabPinned(owner.tabs[0].id, true)

        const profileRead = await readActivePreview(
          {},
          { sessionId: 'foreign-profile-session', runtimeId: null, profile: 'foreign-profile' }
        )

        preview.setPreviewTabPinned(owner.tabs[0].id, false)

        const stale = await actOnActivePreview({ kind: 'click', ref: 'btn-removed-synthetic' }, undefined, {
          sessionId: owner.stored,
          profile: 'default'
        })

        preview.closeRightRailTab(owner.tabs[0].id)

        return {
          declined,
          answered,
          foreignRead,
          profileRead,
          stale,
          closed: await actOnActivePreview({ kind: 'click', selector: '#harmless' }, undefined, {
            sessionId: owner.stored,
            profile: 'default'
          })
        }
      }, owner)

      expect(boundaries.declined).toBe(true)
      expect(boundaries.answered).toBe(false)
      expect(boundaries.foreignRead).toBeNull()
      expect(boundaries.profileRead).toBeNull()
      expect(boundaries.stale.success).toBe(false)
      expect(boundaries.closed.success).toBe(false)
      expect(clicks).toBe(2)
      console.log(
        'ELECTRON_BOUNDARIES',
        JSON.stringify({
          hideRestore: 'PASS',
          reconnect: 'PASS',
          foreignSession: 'PASS',
          foreignProfile: 'PASS',
          staleTarget: 'PASS',
          closedTarget: 'PASS'
        })
      )
    })
  } finally {
    console.log('W0_TOOL_RESULTS', JSON.stringify(results))

    try {
      await fixture.close()
    } finally {
      await new Promise<void>(resolve => provider.close(() => resolve()))
      await new Promise<void>(resolve => site.close(() => resolve()))
    }
  }
})
