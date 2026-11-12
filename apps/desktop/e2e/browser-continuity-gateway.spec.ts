/** Full source-Electron path: scripted provider -> public tools -> owned OOPIF. */
import http from 'node:http'
import path from 'node:path'

import { expect, test } from '@playwright/test'

import { waitForChatReady } from '../../../tests-js/scripts/desktop-chat-smoke'
import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'

import { DesktopFixtureSession } from './fixture-session'
import { buildAppEnv, createSandbox } from './fixtures'

const root = path.resolve(import.meta.dirname, '../../..')
const listen = (server: http.Server, host: string) => new Promise<void>(resolve => server.listen(0, host, resolve))

const close = (server: http.Server) =>
  new Promise<void>(resolve => {
    server.closeAllConnections()
    server.close(() => resolve())
  })

// eslint-disable-next-line no-empty-pattern -- Electron owns its browser window.
test('desktop browser preserves discovery failures and reconnects within the same active turn', async ({}, info) => {
  test.skip(
    !process.env.HERMES_DESKTOP_DEV_SERVER,
    'source Electron fixture requires WSL/Linux and the source renderer'
  )
  test.setTimeout(420_000)
  const results: any[] = []
  let waitingForCancel = false
  let canceledEffects = 0
  let releaseCanceled!: () => void
  const canceled = new Promise<void>(resolve => { releaseCanceled = resolve })
  let waitingForReopen = false
  let releaseReopen!: () => void
  const reopened = new Promise<void>(resolve => { releaseReopen = resolve })

  const site = http.createServer((req, res) => {
    if (req.url === '/cancel-effect') { canceledEffects++; return res.end('recorded') }
    if (req.url === '/login') {
      res.writeHead(302, { 'Set-Cookie': 'frame_fixture=synthetic; HttpOnly; SameSite=Lax; Path=/', Location: '/app' })

      return res.end()
    }

    res.setHeader('Content-Type', 'text/html')

    if (req.url === '/app') {
      if (!req.headers.cookie?.includes('frame_fixture=synthetic')) {
        res.statusCode = 403

        return res.end('Sign in required')
      }

      return res.end(
        `<h1>Authenticated outer</h1><iframe src="http://localhost:${(site.address() as any).port}/child" style="width:600px;height:300px"></iframe>`
      )
    }

    if (req.url === '/nested') {return res.end('<p>Nested public fixture</p><button>Nested public action</button>')}

    return res.end(
      `<p>Frame account details</p><button onclick="document.querySelector('#hit').textContent='Click trusted: '+event.isTrusted">Frame action</button><label>Frame field<input oninput="if(this.value.includes('CANCELED_SHOULD_NOT_APPEAR'))fetch('/cancel-effect');document.querySelector('#typed').textContent='Type trusted: '+event.isTrusted+' '+this.value"></label><p id="hit"></p><p id="typed"></p><iframe src="http://127.0.0.1:${(site.address() as any).port}/nested" style="width:400px;height:100px"></iframe>`
    )
  })

  await listen(site, '::')
  const url = `http://127.0.0.1:${(site.address() as any).port}`

  const provider = http.createServer(async (req, res) => {
    if (req.method === 'GET') {
      res.setHeader('Content-Type', 'application/json')

      return res.end(JSON.stringify({ data: [{ id: 'mock-model', object: 'model' }] }))
    }

    let raw = ''

    for await (const chunk of req) {
      raw += chunk
    }

    const body = JSON.parse(raw)

    if (body.tools?.length === 1 && body.tools[0]?.function?.name === 'browser_target_intent') {
      const prompt = JSON.parse(body.messages.at(-1).content).request
      const opening = prompt.includes('Open the frame fixture')
      const closing = prompt.includes('Close the frame fixture')
      const visible = opening || closing || prompt.includes('Inspect the frame fixture') || prompt.includes('CONTINUITY_TASK') || prompt.includes('CANCEL_RECONNECT_TASK')
      res.setHeader('Content-Type', 'application/json')

      return res.end(
        JSON.stringify({
          id: 'fixture-intent',
          choices: [
            {
              index: 0,
              finish_reason: 'tool_calls',
              message: {
                role: 'assistant',
                content: null,
                tool_calls: [
                  {
                    id: 'intent',
                    type: 'function',
                    function: {
                      name: 'browser_target_intent',
                      arguments: JSON.stringify({
                        target_kind: visible ? 'VISIBLE_PANE' : 'NONE',
                        operation: closing ? 'close' : opening ? 'open' : visible ? 'existing' : 'none',
                        ...(opening ? { url: url + (prompt.includes('destination') ? '/app' : prompt.includes('same document') ? '/app#section' : prompt.includes('cross document') ? '/other' : '/login') } : {})
                      })
                    }
                  }
                ]
              }
            }
          ],
          usage: { prompt_tokens: 10, completion_tokens: 10, total_tokens: 20 }
        })
      )
    }

    const messages = body.messages || []
    const userIndex = messages.findLastIndex((m: any) => m.role === 'user')
    const prompt = JSON.stringify(messages[userIndex]?.content || '')
    const toolResults = messages.slice(userIndex + 1).filter((m: any) => m.role === 'tool')

    for (const entry of toolResults) {
      if (!results.some(item => item.id === entry.tool_call_id)) {
        results.push({ id: entry.tool_call_id, content: String(entry.content) })
      }
    }

    const parse = (value: string) => {
      try {
        return JSON.parse(value)
      } catch {
        return null
      }
    }

    const inventory = toolResults
      .map((entry: any) => parse(entry.content))
      .find((entry: any) => Array.isArray(entry?.elements))

    const count = toolResults.filter((entry: any) => !String(entry.content).includes('"tools":')).length
    const button = inventory?.elements?.find((element: any) => element.label === 'Frame action')?.ref
    const field = inventory?.elements?.find((element: any) => element.label === 'Frame field')?.ref
    let call: any

    if (prompt.includes('Inspect the frame fixture') && count === 2 && button) {
      call = { name: 'drive_preview', arguments: { action: 'click', ref: button } }
    }

    if (prompt.includes('Inspect the frame fixture') && count === 3 && field) {
      call = { name: 'drive_preview', arguments: { action: 'type', ref: field, text: 'synthetic' } }
    }

    if (prompt.includes('Inspect the frame fixture') && count === 4) {
      call = { name: 'drive_preview', arguments: { action: 'press', ref: field, key: 'Enter' } }
    }

    if (prompt.includes('Inspect the frame fixture') && count === 5) {
      call = { name: 'desktop_preview', arguments: { action: 'read' } }
    }

    if (prompt.includes('CONTINUITY_TASK')) {
      if (count === 2) { call = { name: 'drive_preview', arguments: { action: 'click', ref: 'invalid-fixture-ref' } } }
      if (count === 3) { call = { name: 'desktop_preview', arguments: { action: 'close' } } }
      if (count === 4) {
        waitingForReopen = true
        await reopened
        call = { name: 'desktop_preview', arguments: { action: 'read' } }
      }
      if (count === 5) { call = { name: 'drive_preview', arguments: { action: 'hover', ref: button ?? 'old-fixture-ref' } } }
      if (count === 6) { call = { name: 'drive_preview', arguments: { action: 'elements' } } }
      if (count === 7) { call = { name: 'desktop_preview', arguments: { action: 'read' } } }
    }

    if (prompt.includes('Close the frame fixture') && count === 2) { call = { name: 'desktop_preview', arguments: { action: 'close' } } }

    if (prompt.includes('CANCEL_RECONNECT_TASK') && count === 2) {
      waitingForCancel = true
      await canceled
      call = { name: 'drive_preview', arguments: { action: 'type', ref: field, text: 'CANCELED_SHOULD_NOT_APPEAR' } }
    }
    const message = call
      ? {
          role: 'assistant',
          content: null,
          tool_calls: [
            {
              id: `${prompt.includes('CONTINUITY_TASK') ? 'continuity' : 'frame'}-${count}`,
              type: 'function',
              function: { name: 'tool_call', arguments: JSON.stringify({ calls: [call] }) }
            }
          ]
        }
      : { role: 'assistant', content: prompt.includes('Close the frame fixture') ? 'CLOSE_DONE' : prompt.includes('CONTINUITY_TASK') ? 'CONTINUITY_DONE' : prompt.includes('Inspect the frame fixture') ? 'FRAME_DONE' : prompt.includes('Open the frame fixture') ? 'OPEN_DONE' : 'FRAME_READY' }

    if (body.stream) {
      res.writeHead(200, { 'Content-Type': 'text/event-stream' })
      const delta = call ? { role: 'assistant', tool_calls: [{ index: 0, ...message.tool_calls![0] }] } : message
      res.write(
        `data: ${JSON.stringify({ id: 'fixture', object: 'chat.completion.chunk', created: 0, model: body.model, choices: [{ index: 0, delta, finish_reason: null }] })}\n\n`
      )

      return res.end(
        `data: ${JSON.stringify({ id: 'fixture', object: 'chat.completion.chunk', created: 0, model: body.model, choices: [{ index: 0, delta: {}, finish_reason: call ? 'tool_calls' : 'stop' }] })}\n\ndata: [DONE]\n\n`
      )
    }

    res.setHeader('Content-Type', 'application/json')
    res.end(
      JSON.stringify({
        id: 'fixture',
        choices: [{ index: 0, message, finish_reason: call ? 'tool_calls' : 'stop' }],
        usage: { prompt_tokens: 10, completion_tokens: 10, total_tokens: 20 }
      })
    )
  })

  await listen(provider, '127.0.0.1')
  const sandbox = createSandbox('preview-frames')
  writeMockProviderConfig(
    sandbox.hermesHome,
    `http://127.0.0.1:${(provider.address() as any).port}`,
    'language: en\ndisplay:\n  language: en',
    'auxiliary:\n  background_review:\n    enabled: false\n  title_generation:\n    enabled: false\nagent:\n  max_iterations: 12\n  verify_on_stop: false\n'
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
    page.on('console', msg => { if (msg.text().startsWith('TRACE_')) {console.log(msg.text())} })
    const composer = await waitForChatReady(page, 120_000)
    await page.evaluate(() => {
      const watched = new WeakSet<Element>()

      const watch = (node: Element) => {
        if (node.tagName !== 'WEBVIEW' || watched.has(node)) {return}
        watched.add(node)
        const view = node as any

        const info = () => {
          try { return { guest: view.getWebContentsId?.(), url: view.getURL?.(), loading: view.isLoading?.() } }
          catch { return {} }
        }

        console.log('TRACE_MOUNT', JSON.stringify(info()))

        for (const name of ['did-start-navigation', 'did-redirect-navigation', 'did-navigate', 'did-navigate-in-page', 'did-start-loading', 'did-stop-loading', 'dom-ready']) {
          node.addEventListener(name, (event: any) => console.log('TRACE_EVENT', JSON.stringify({ name, eventUrl: event.url, ...info() })))
        }
      }

      new MutationObserver(records => records.forEach(record => {
        record.addedNodes.forEach(node => {
          if (node instanceof Element) { watch(node); node.querySelectorAll('webview').forEach(watch) }
        })
        record.removedNodes.forEach(node => {
          if (node instanceof Element && node.tagName === 'WEBVIEW') {console.log('TRACE_UNMOUNT')}
        })
      })).observe(document.body, { childList: true, subtree: true })
    })
    await composer.fill('FRAME_INIT')
    await composer.press('Enter')
    await expect(page.getByText('FRAME_READY', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.getByRole('button', { name: 'No thanks', exact: true }).click()
    let opens = 0

    const snapshot = async (stage: string) => {
      const state = await page.evaluate(async () => {
        const p = await import('/src/store/' + 'preview.ts')
        const r = await import('/src/app/chat/right-rail/' + 'preview-reader.ts')
        const s = await import('/src/store/' + 'session.ts')
        const runtime = s.$activeSessionId.get()
        const stored = s.$selectedStoredSessionId.get()
        const owner = { profile: 'default', runtimeId: runtime, sessionId: stored }
        const view = document.querySelector('webview') as any
        let guest: any = null

        try { guest = { id: view?.getWebContentsId?.(), url: view?.getURL?.(), loading: view?.isLoading?.() } } catch { /* Guest is not attached yet. */ }

        return { runtime, stored, binding: r.activePreviewBinding(owner, true), tabs: p.previewTabsFor(owner), pages: p.$browserPages.get(), guest }
      })

      console.log('TRACE_SNAPSHOT', JSON.stringify({ stage, ...state }))

      return state
    }

    const openTurn = async (suffix: string) => {
      console.log('TRACE_REQUEST', JSON.stringify({ suffix }))
      await composer.fill(`Open the frame fixture ${suffix}.`)
      await composer.press('Enter')
      await expect(page.getByText('OPEN_DONE', { exact: true })).toHaveCount(++opens, { timeout: 25000 })

      return snapshot('open-' + suffix)
    }

    const initial = await openTurn('original')
    const guest = page.locator('webview').first()
    await guest.waitFor()
    await expect
      .poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText')).catch(() => ''))
      .toContain('Authenticated outer')
    await page.getByRole('button', { name: 'Not now', exact: true }).click()
    console.log('OWNER', JSON.stringify(await page.evaluate(async () => { const p = await import('/src/store/' + 'preview.ts'); const s = await import('/src/store/' + 'session.ts');

 return { runtime:s.$activeSessionId.get(), stored:s.$selectedStoredSessionId.get(), tabs:p.$previewTabs.get().map((t:any)=>({id:t.id,sessionId:t.sessionId,profile:t.profile})) } })))
    await page.waitForTimeout(3000)
    console.log('BINDING', JSON.stringify(await page.evaluate(async () => { const p=await import('/src/app/chat/right-rail/' + 'preview-reader.ts'); const s=await import('/src/store/' + 'session.ts'); const id=s.$activeSessionId.get();

 return { default:p.activePreviewBinding(), owned:p.activePreviewBinding({runtimeId:id,sessionId:id,profile:'default'}) } })))
    await composer.fill('Inspect the frame fixture already open beside this conversation.')
    await composer.press('Enter')
    await expect(page.getByText('FRAME_DONE', { exact: true })).toBeVisible({ timeout: 30_000 })
    const beforeReconnect = await snapshot('continuity-before')
    await composer.fill('CONTINUITY_TASK: read this synthetic page. Do not close, reload, or navigate it.')
    await composer.press('Enter')
    await expect.poll(() => waitingForReopen, { timeout: 30000 }).toBe(true)
    const preserved = await snapshot('continuity-preserved')
    expect(preserved.binding?.guest_id).toBe(beforeReconnect.binding?.guest_id)
    expect(results.some(item => item.content.includes('USER_INTENT_REQUIRED'))).toBe(true)
    // Explicit fixture-owned user close/reopen while the provider holds this turn.
    await page.evaluate(async ({ intendedUrl }) => {
      const p = await import('/src/store/' + 'preview.ts')
      const r = await import('/src/app/chat/right-rail/' + 'preview-reader.ts')
      const old = r.activePreviewBinding()
      if (!old) { throw new Error('missing fixture guest') }
      p.closeRightRailTab(old.tab_id)
      p.openBrowserTab()
      p.openPreview({ kind: 'url', label: 'synthetic replacement', source: intendedUrl, url: intendedUrl })
      p.openBrowserTab()
    }, { intendedUrl: url + '/app' })
    await expect.poll(async () => (await snapshot('continuity-reopened')).guest?.loading, { timeout: 20000 }).toBe(false)
    const replacement = await snapshot('continuity-guest-b')
    expect(replacement.binding?.guest_id).not.toBe(beforeReconnect.binding?.guest_id)
    expect(replacement.runtime).toBe(beforeReconnect.runtime)
    releaseReopen()
    await expect(page.getByText('CONTINUITY_DONE', { exact: true })).toBeVisible({ timeout: 30000 })
    const continuityResult = (count: number) => JSON.parse(results.find(item => item.id === `continuity-${count}`)?.content ?? '{}')
    expect(continuityResult(2).code).toBe('INVALID_ELEMENT_REF')
    expect(continuityResult(3).code).toBe('USER_INTENT_REQUIRED')
    expect(continuityResult(4)).toMatchObject({ success: true, binding: { guest_id: replacement.binding!.guest_id } })
    expect(continuityResult(5).code).toBe('STALE_TARGET')
    expect(continuityResult(6).elements.some((element: any) => element.label === 'Nested public action')).toBe(true)
    expect(continuityResult(7)).toMatchObject({ success: true, binding: { guest_id: replacement.binding!.guest_id } })
    const afterReconnect = await snapshot('continuity-after')
    expect(afterReconnect.binding?.guest_id).toBe(replacement.binding?.guest_id)
    await composer.fill('Inspect the frame fixture already open beside this conversation.')
    await composer.press('Enter')
    await expect(page.getByText('FRAME_DONE', { exact: true })).toHaveCount(2, { timeout: 30000 })
    await composer.fill('CANCEL_RECONNECT_TASK: inspect the current synthetic page.')
    await composer.press('Enter')
    await expect.poll(() => waitingForCancel, { timeout: 30000 }).toBe(true)
    await page.getByRole('button', { name: /^Stop(?: generation| response)?$/ }).first().click()
    releaseCanceled()
    await expect(page.getByRole('button', { name: /^Stop(?: generation| response)?$/ })).toHaveCount(0)
    await composer.fill('Close the frame fixture pane as explicitly requested by the user.')
    await composer.press('Enter')
    await expect(page.getByText('CLOSE_DONE', { exact: true })).toBeVisible({ timeout: 30000 })
    await expect(page.locator('webview')).toHaveCount(0)
    expect(canceledEffects).toBe(0)
    await info.attach('continuity-results', { body: JSON.stringify(results), contentType: 'application/json' })
  } finally {
    await info.attach('public-gateway-results', { body: JSON.stringify(results), contentType: 'application/json' })
    console.log('FRAME_GATEWAY_RESULT_COUNT', results.length)

    try {
      await fixture.close()
    } finally {
      await close(provider)
      await close(site)
    }
  }
})
