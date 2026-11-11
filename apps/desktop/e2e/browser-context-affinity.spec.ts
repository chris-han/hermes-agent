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

import { toolResultPayload } from './autonomous-evaluator'
import { DesktopFixtureSession } from './fixture-session'
import { buildAppEnv, createSandbox } from './fixtures'

const root = path.resolve(import.meta.dirname, '../../..')

function nativeIntentReply(body: any, res: http.ServerResponse) {
  if (body.tools?.length !== 1 || body.tools[0]?.function?.name !== 'browser_target_intent') {return false}
  const request = JSON.parse(body.messages.at(-1).content).request
  const opening = request.startsWith('Open the synthetic dashboard beside our conversation:')
  const visible = opening || request.includes('page currently open') || request.includes('webpage visible') || request.includes('that exact page')
  res.setHeader('Content-Type', 'application/json')
  res.end(JSON.stringify({
    id: 'fixture-intent',
    choices: [{ index: 0, finish_reason: 'tool_calls', message: {
      role: 'assistant', content: null,
      tool_calls: [{ id: 'fixture-intent', type: 'function', function: {
        name: 'browser_target_intent', arguments: JSON.stringify({
          target_kind: visible ? 'VISIBLE_PANE' : 'NONE', operation: opening ? 'open' : visible ? 'existing' : 'none',
          ...(opening ? { url: request.match(/https?:\/\/[^\s]+/)[0] } : {})
        })
      } }]
    } }],
    usage: { prompt_tokens: 10, completion_tokens: 10, total_tokens: 20 }
  }))

  return true
}

function scriptedStep(opening: boolean, prompt: string, steps: ReadonlyArray<readonly [string, unknown] | undefined>, openingSteps: ReadonlyArray<readonly [string, unknown] | undefined>, index: number) {
  return opening ? openingSteps[index] : (prompt.includes('Inspect the page currently open') || prompt.includes('Click the test button')) ? steps[index] : undefined
}

function scriptedCall(step: readonly [string, unknown]) {
  return { name: step[0] === 'browser_exec' ? 'browser_exec' : 'tool_call',
    arguments: JSON.stringify(['tool_call', 'browser_exec'].includes(step[0]) ? step[1] : {calls: [{name: step[0], arguments: step[1]}]}) }
}

// eslint-disable-next-line no-empty-pattern -- Electron supplies its own page; only Playwright TestInfo is used.
test('authenticated owned Electron page remains distinct from independent automation', async ({}, info) => {
  test.skip(
    !['linux', 'win32'].includes(process.platform) || !process.env.HERMES_DESKTOP_DEV_SERVER,
    'WSL source-renderer lane requires HERMES_DESKTOP_DEV_SERVER and local browser tools'
  )
  // gateway.run also loads the source checkout's dotenv. Never let this synthetic
  // journey hydrate developer messaging credentials or auto-install adapters.
  expect(
    fs.existsSync(path.join(root, '.env')),
    'Run this lane from a clean source worktree without a project .env'
  ).toBe(false)
  test.setTimeout(process.platform === 'win32' ? 420_000 : 240_000)
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
        ? "<h1>Synthetic account signed in</h1><button id=\"harmless\" onclick=\"fetch('/action').then(()=>document.querySelector('output').textContent='clicked')\">Harmless action</button><output>unchanged</output><input id=fixture-input aria-label=Fixture-input>"
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

    // The native semantic proposal is separate from the normal Agent tool loop.
    // This deterministic provider fixture tests execution, never autonomous routing.
    if (nativeIntentReply(body, res)) {return}

    const messages = body.messages || []

    const readTurn = messages.findLastIndex(
      (m: any) => m.role === 'user' && ['Inspect the page currently open in my right-side browser.', 'Click the test button on the webpage visible beside this conversation.', 'Open the synthetic dashboard beside our conversation:', 'Continue using that exact page; check it live.'].some(text => JSON.stringify(m.content).includes(text))
    )

    const lastUser = readTurn >= 0 ? readTurn : messages.findLastIndex((m: any) => m.role === 'user')
    const prompt = JSON.stringify(messages[lastUser]?.content || '')
    const toolResults = messages.slice(lastUser + 1).filter((m: any) => m.role === 'tool')

    const describeIds = new Set(messages.slice(lastUser + 1).flatMap((m: any) =>
      (m.tool_calls || []).filter((call: any) => call.function?.name === 'tool_describe').map((call: any) => call.id)))

    const actionResultCount = toolResults.filter((m: any) => !describeIds.has(m.tool_call_id)).length

    for (const m of toolResults) {
      if (!results.some(r => r.name === (m.name || m.tool_call_id))) {
        results.push({ name: m.name || m.tool_call_id, content: String(m.content) })
      }
    }

    const forbiddenIndependent = `new_tab(${JSON.stringify(siteUrl + '/login')}); wait_for_load(); print(js('document.body.innerText')); js("document.querySelector('#harmless').click()")`
    const steps = [
      ['desktop_preview', { action: 'read' }],
      ['drive_preview', { action: 'elements' }],
      ['drive_preview', { action: 'fill', selector: '#fixture-input', text: 'SHOULD_NOT_LAND' }],
      ['drive_preview', { action: 'click' }],
      ['tool_call', { calls: '{not-json' }],
      ['browser_exec', { code: forbiddenIndependent, session: 'rejected-one' }],
      ['browser_exec', { code: forbiddenIndependent, session: 'rejected-two' }],
      ['drive_preview', { action: 'click', ref: 'btn-unavailable' }],
      ['drive_preview', { action: 'elements', full: true }],
      ['drive_preview', { action: 'click', selector: '#harmless' }],
      ['drive_preview', { action: 'type', selector: '#fixture-input', text: 'fixture typed' }]
    ] as const

    const opening = prompt.includes('Open the synthetic dashboard beside our conversation:')

    const openingSteps = [undefined, undefined,
      ['drive_preview', { action: 'open', url: siteUrl + '/login' }],
      ['drive_preview', { action: 'click', selector: '#harmless' }]
    ] as const

    const step = scriptedStep(opening, prompt, steps, openingSteps, actionResultCount)

    const message = step
      ? {
          role: 'assistant',
          content: null,
          tool_calls: [
            {
              id: `affinity-${toolResults.length}`,
              type: 'function',
              function: scriptedCall(step)
            }
          ]
        }
      : { role: 'assistant', content: opening ? 'BINDING_OPEN_DONE' : (prompt.includes('Inspect the page currently open') || prompt.includes('Click the test button')) ? 'AFFINITY_DONE' : 'AFFINITY_READY' }

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
    'language: en\ndisplay:\n  language: en',
    'auxiliary:\n  background_review:\n    enabled: false\n  title_generation:\n    enabled: false\nagent:\n  max_iterations: 16\n  verify_on_stop: false\nbrowser:\n  backend: browser-use\n  cloud_provider: local\n  engine: chrome\n  allow_private_urls: true\n  use_real_profile: false\n'
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
      const composer = await waitForChatReady(page, process.platform === 'win32' ? 180_000 : 120_000)
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
      await composer.pressSequentially('Inspect the page currently open in my right-side browser.')
      await expect(page.getByRole('button', { name: 'Send', exact: true })).toBeEnabled()
      await page.getByRole('button', { name: 'Send', exact: true }).click()
      await expect(page.getByText('AFFINITY_DONE', { exact: true })).toBeVisible({ timeout: 60_000 })
      console.log(
        'AFFINITY_RECEIPT',
        JSON.stringify({ owner, results, clicks, scriptedProvider: true, autonomousToolSelection: 'NOT_RUN' })
      )
      expect(results[0]?.content).toContain('Synthetic account signed in')
      expect(results[1]?.content).toContain('Harmless action')

      const disclosed = results.map(result => {
        try { return JSON.parse(result.content) } catch { return null }
      }).find(result => result?.tools?.drive_preview)

      expect(disclosed.tools.drive_preview.parameters.properties.action.enum).toContain('type')
      expect(disclosed.tools.drive_preview.parameters.properties.action.enum).not.toContain('fill')
      expect(disclosed.tools.drive_preview.parameters.properties.text.type).toBe('string')
      const rejections = results.flatMap(result => {
        const data = toolResultPayload(result.content)
        return data?.code ? [data] : []
      })
      expect(rejections.map(result => result.code)).toEqual(['UNKNOWN_ACTION', 'MISSING_REQUIRED_FIELD', 'INVALID_ARGUMENT', 'TARGET_MISMATCH', 'TARGET_MISMATCH', 'INVALID_ELEMENT_REF'])
      expect(rejections.every(result => result.recovery?.same_context)).toBe(true)
      fs.writeFileSync(info.outputPath('execution-contract-rejections.json'), JSON.stringify({
        scope: 'Scripted native reject/correct journey; not autonomous model qualification', rejections
      }, null, 2))
      fs.writeFileSync(info.outputPath('provider-requests.json'), JSON.stringify(requests, null, 2))

      const firstMain = requests.find(request => request.tools?.some((tool: any) => tool.function?.name === 'tool_call')
        && request.messages?.some((message: any) => message.role === 'user' && message.content === 'Inspect the page currently open in my right-side browser.'))

      expect(firstMain).toBeTruthy()

      const contractsBeforeMain = firstMain.messages.filter((message: any) => message.role === 'tool').map((message: any) => {
        try { return JSON.parse(message.content) } catch { return null }
      }).find((result: any) => result?.tools?.drive_preview)

      expect(contractsBeforeMain?.tools.drive_preview.parameters).toEqual(disclosed.tools.drive_preview.parameters)
      fs.writeFileSync(info.outputPath('native-contract-before-main.json'), JSON.stringify({
        scope: 'Scripted provider; actual native preparation before first main request; no autonomous selection claim',
        contracts: contractsBeforeMain.tools,
        nativeCalls: firstMain.messages.flatMap((message: any) => message.tool_calls || [])
      }, null, 2))

      for (const message of firstMain.messages) {
        for (const call of message.tool_calls || []) {
          if (call.function?.name === 'tool_call') {expect(JSON.parse(call.function.arguments).calls).toBeInstanceOf(Array)}
        }
      }

      console.log('NATIVE_TOOL_CONTRACT', JSON.stringify({ fullSchemaBeforeMain: 'PASS', nativeWrapperShape: 'PASS' }))
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

      expect(await guest.evaluate((el: any) => el.executeJavaScript("document.querySelector('#fixture-input').value"))).toBe('fixture typed')
      await page.screenshot({ path: info.outputPath('authenticated-pane.png') })
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
      await composer.fill('Click the test button on the webpage visible beside this conversation.')
      await page.getByRole('button', { name: 'Send', exact: true }).click()
      await expect(page.getByText('AFFINITY_DONE', { exact: true })).toHaveCount(2, { timeout: 60_000 })
      expect(results[0]?.content).toContain('Synthetic account signed in')
      expect(clicks).toBe(2)
      expect(await guest.evaluate((el: any) => el.getWebContentsId())).toBe(guestId)

      await fixture.stage('browser.two-windows', async () => {
        const nextWindow = fixture.app!.waitForEvent('window')
        await page.evaluate(() => (window as any).hermesDesktop.openWindow())
        const second = await nextWindow
        await second.waitForLoadState('domcontentloaded')
        await expect(second.getByRole('button', { name: 'default', exact: true })).toBeVisible({ timeout: 180_000 })
        await second.bringToFront()
        await second.evaluate(async url => {
          const preview = await import('/src/store/' + 'preview.ts')
          preview.openPreview({ kind: 'url', source: 'user', label: 'Second window fixture', url })
        }, siteUrl + '/account')
        const secondGuestId = await second.locator('webview').first().evaluate((el: any) => el.getWebContentsId())
        expect(secondGuestId).not.toBe(guestId)
        await second.evaluate(async () => {
          const preview = await import('/src/store/' + 'preview.ts')

          for (const tab of preview.$previewTabs.get()) {preview.closeRightRailTab(tab.id)}
        })

        const refused = await second.evaluate(async owner => {
          const { readActivePreview } = await import('/src/app/chat/right-rail/' + 'preview-reader.ts')
          const { actOnActivePreview } = await import('/src/app/chat/right-rail/' + 'preview-act.ts')

          return {
            read: await readActivePreview({}, { sessionId: owner.stored, profile: 'default' }),
            action: await actOnActivePreview({ kind: 'click', selector: '#harmless' }, undefined, {
              sessionId: owner.stored, profile: 'default'
            })
          }
        }, owner)

        expect(refused.read).toBeNull()
        expect(refused.action.success).toBe(false)
        expect(clicks).toBe(2)
        await page.bringToFront()
        expect(await guest.evaluate((el: any) => el.getWebContentsId())).toBe(guestId)
        await expect.poll(() => guest.evaluate((el: any) => el.executeJavaScript('document.body.innerText'))).toContain('Synthetic account signed in')
        await second.close()
        console.log('TWO_WINDOWS', JSON.stringify({ isolation: 'PASS', focusSwitch: 'PASS' }))
      })

      await fixture.stage('browser.binding-open-and-unavailable', async () => {
        results.length = 0
        await composer.fill('Open the synthetic dashboard beside our conversation: ' + siteUrl + '/login')
        await page.getByRole('button', { name: 'Send', exact: true }).click()
        await expect(page.getByText('BINDING_OPEN_DONE', { exact: true })).toBeVisible({ timeout: 60_000 })
        expect(clicks).toBe(3)
        expect(results.every(row => !row.content.includes('does not exist') && !row.content.includes('must be one of'))).toBe(true)
        expect(requests.some(body => (body.messages || []).some((message: any) =>
          message.tool_calls?.some((call: any) => call.function?.arguments?.includes('drive_preview') && call.function?.arguments?.includes('open'))))).toBe(true)
        const requestsBefore = requests.length
        await page.evaluate(() => {
          const state = (globalThis as any)[Symbol.for('hermes.desktop.gatewayRegistryState')]

          if (!state?.config?.onServerRequest) { throw new Error('Missing native bridge for fault fixture') }
          ;(globalThis as any).__bindingRestore = state.config.onServerRequest
          ;(globalThis as any).__bindingDeclined = []

          state.config.onServerRequest = (request: any) => {
            if (request.method === 'preview.read' || request.method === 'preview.act') {
              ;(globalThis as any).__bindingDeclined.push(request.method)
              request.decline()
            } else { (globalThis as any).__bindingRestore(request) }
          }
        })

        try {
          await composer.fill('Continue using that exact page; check it live.')
          await page.getByRole('button', { name: 'Send', exact: true }).click()
          await expect(page.getByText('I cannot access that page live in this conversation right now.', { exact: false })).toBeVisible({ timeout: 90_000 })
          expect(await page.evaluate(() => (globalThis as any).__bindingDeclined)).toContain('preview.read')
          // Background model-availability probes have no Chat Completions messages.
          // Count every actual Agent request, including an unexpected main completion.
          const attempted = requests.slice(requestsBefore).filter(body => Array.isArray(body.messages))
          expect(attempted).toHaveLength(1)
          expect(attempted[0].tools[0].function.name).toBe('browser_target_intent')
          expect(clicks).toBe(3)
          console.log('NATIVE_BINDING_B_D', JSON.stringify({ open: 'PASS', deferredNormalization: 'PASS', liveUnavailable: 'PASS', mainModelAfterUnavailable: 0 }))
        } finally {
          await page.evaluate(() => {
            const state = (globalThis as any)[Symbol.for('hermes.desktop.gatewayRegistryState')]
            state.config.onServerRequest = (globalThis as any).__bindingRestore
          })
        }
      })

      await page.evaluate(async () => {
        const session = await import('/src/store/' + 'session.ts')

        return session.$messages.get().map((message: any) => ({ ...message,
          parts: message.parts?.filter((part: any) => part.type !== 'reasoning')
        }))
      }).then(messages => fs.writeFileSync(info.outputPath('native-turns.json'), JSON.stringify(messages, null, 2)))

      // Exercise the production request owner and live-target registries in this real renderer.
      const boundaries = await page.evaluate(async owner => {
        const { handleServerRequest } = await import(
          '/src/app/session/hooks/use-message-stream/gateway-event/' + 'server-requests.ts'
        )

        const { actOnActivePreview } = await import('/src/app/chat/right-rail/' + 'preview-act.ts')
        const { readActivePreview } = await import('/src/app/chat/right-rail/' + 'preview-reader.ts')
        const preview = await import('/src/store/' + 'preview.ts')
        const lease = await readActivePreview({ require_live: true }, { sessionId: owner.stored, profile: 'default' })

        if (!lease?.binding) { throw new Error('Missing actual Electron live lease') }
        const view = document.querySelector('webview') as any
        await new Promise<void>((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error('Fixture navigation timed out')), 15000)
          view.addEventListener('dom-ready', () => { clearTimeout(timer); resolve() }, { once: true })
          view.reload()
        })
        let navigationReadError = ''

        try { await readActivePreview({ require_live: true, expected_binding: lease.binding }, { sessionId: owner.stored, profile: 'default' }) }
        catch (error) { navigationReadError = String(error) }

        const navigationAction = await new Promise<any>((resolve, reject) => {
          handleServerRequest({ id: 'synthetic-stale-lease', method: 'preview.act', profile: 'default',
            params: { session_id: owner.runtime, action: 'click', selector: '#harmless', expected_binding: lease.binding },
            respond: (result: any) => resolve(JSON.parse(result.value)),
            decline: () => reject(new Error('Owner request unexpectedly declined')), fail: reject
          }, { activeSessionIdRef: { current: owner.runtime }, sessionStateByRuntimeIdRef: { current: new Map() }, sessionInterrupted: () => false }, owner.runtime)
        })

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
          navigationReadError,
          navigationAction,
          foreignRead,
          profileRead,
          stale,
          closed: await actOnActivePreview({ kind: 'click', selector: '#harmless' }, undefined, {
            sessionId: owner.stored,
            profile: 'default'
          })
        }
      }, owner)

      expect(boundaries.navigationReadError).toContain('STALE_TARGET')
      expect(boundaries.navigationAction.success).toBe(false)
      expect(JSON.stringify(boundaries.navigationAction)).toContain('STALE_TARGET')
      expect(boundaries.declined).toBe(true)
      expect(boundaries.answered).toBe(false)
      expect(boundaries.foreignRead).toBeNull()
      expect(boundaries.profileRead).toBeNull()
      expect(boundaries.stale.success).toBe(false)
      expect(boundaries.closed.success).toBe(false)
      expect(clicks).toBe(3)
      console.log(
        'ELECTRON_BOUNDARIES',
        JSON.stringify({
          hideRestore: 'PASS',
          reconnect: 'PASS',
          navigationLease: 'PASS',
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
