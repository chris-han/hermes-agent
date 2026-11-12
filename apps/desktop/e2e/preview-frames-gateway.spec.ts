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
test('desktop tools read and drive an authenticated cross-origin frame', async ({}, info) => {
  test.skip(
    !process.env.HERMES_DESKTOP_DEV_SERVER,
    'source Electron fixture requires the source renderer'
  )
  test.setTimeout(240_000)
  const results: any[] = []
  const site = http.createServer((req, res) => {
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
    return res.end(
      `<p>Frame account details</p><button onclick="document.querySelector('#hit').textContent='Click trusted: '+event.isTrusted">Frame action</button><label>Frame field<input oninput="document.querySelector('#typed').textContent='Type trusted: '+event.isTrusted+' '+this.value"></label><p id="hit"></p><p id="typed"></p>`
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
      const visible = prompt.includes('Inspect the frame fixture')
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
                        operation: visible ? 'existing' : 'none'
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
      call = { name: 'desktop_preview', arguments: { action: 'read' } }
    }
    const message = call
      ? {
          role: 'assistant',
          content: null,
          tool_calls: [
            {
              id: `frame-${count}`,
              type: 'function',
              function: { name: 'tool_call', arguments: JSON.stringify({ calls: [call] }) }
            }
          ]
        }
      : { role: 'assistant', content: prompt.includes('Inspect the frame fixture') ? 'FRAME_DONE' : 'FRAME_READY' }
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
    const composer = await waitForChatReady(page, 120_000)
    await composer.fill('FRAME_INIT')
    await composer.press('Enter')
    await expect(page.getByText('FRAME_READY', { exact: true })).toBeVisible({ timeout: 60_000 })
    await page.getByRole('button', { name: 'No thanks', exact: true }).click()
    await page.evaluate(async target => {
      const { openPreview } = await import('/src/store/' + 'preview.ts')
      openPreview({ kind: 'url', source: 'user', label: 'Synthetic frame fixture', url: target })
    }, url + '/login')
    const guest = page.locator('webview').first()
    await guest.waitFor()
    await expect
      .poll(() => guest.evaluate((view: any) => view.executeJavaScript('document.body.innerText')).catch(() => ''))
      .toContain('Authenticated outer')
    await page.getByRole('button', { name: 'Not now', exact: true }).click()
    await composer.fill('Inspect the frame fixture already open beside this conversation.')
    await composer.press('Enter')
    await expect(page.getByText('FRAME_DONE', { exact: true })).toBeVisible({ timeout: 90_000 })
    const combined = results.map(result => result.content).join('\n')
    expect(combined).toContain('Frame account details')
    expect(combined).toContain('frame:')
    expect(combined).toContain('Click trusted: true')
    expect(combined).toContain('Type trusted: true synthetic')
  } finally {
    console.log('FRAME_GATEWAY_RESULTS', JSON.stringify(results))
    try {
      await fixture.close()
    } finally {
      await close(provider)
      await close(site)
    }
  }
})
