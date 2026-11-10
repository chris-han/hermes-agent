/** Optional paid six-case qualification: real provider decisions, actual Desktop effects. */
import { type ChildProcess, spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import http from 'node:http'
import path from 'node:path'

import { expect, type Page, test } from '@playwright/test'
import yaml from 'js-yaml'

import { waitForChatReady } from '../../../tests-js/scripts/desktop-chat-smoke'

import { browserOperations, declineProfileConsent, turnEvidence, unwrapTools } from './autonomous-evaluator'
import { DesktopFixtureSession } from './fixture-session'
import { buildAppEnv, createSandbox } from './fixtures'

const root = path.resolve(import.meta.dirname, '../../..')
const hash = (value: string) => createHash('sha256').update(value).digest('hex')

const prompts = {
  A: 'Please inspect the page already open on the right, change the test setting to Enabled, save it, and tell me whether the change was successful.',
  B: 'Open the test dashboard beside our conversation, inspect its status, and click the test action.',
  C: 'Compare the information on the three test sites without changing the page open beside our chat.',
  D: 'The page already open on the right is the one I need. Please continue working on that exact page.',
  E: 'Update the test setting on the page open beside this conversation.',
  F: 'Continue on the same page and verify the setting is still enabled.'
} as const

type CaseId = keyof typeof prompts
const order: CaseId[] = process.env.SEMANTIER_EVAL_RECOVER_REMAINING === '1' ? ['C', 'E', 'F', 'D'] : ['A', 'B', 'C', 'E', 'F', 'D']

async function listen(server: http.Server): Promise<string> {
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))

  return `http://127.0.0.1:${(server.address() as any).port}`
}

async function close(server: http.Server) {
  server.closeAllConnections()
  await new Promise<void>(resolve => server.close(() => resolve()))
}

async function owner(page: Page) {
  return page.evaluate(async () => {
    const session = await import('/src/store/' + 'session.ts')
    const preview = await import('/src/store/' + 'preview.ts')
    const states = await import('/src/store/' + 'session-states.ts')
    const id = session.$activeSessionId.get()
    const state = states.$sessionStates.get()[id || '']

    return {
      runtime: id, stored: session.$selectedStoredSessionId.get(),
      busy: state?.busy ?? false, awaiting: state?.awaitingResponse ?? false,
      needsInput: state?.needsInput ?? false,
      tabs: preview.$previewTabs.get().map((tab: any) => ({ id: tab.id, sessionId: tab.sessionId, profile: tab.profile })),
      messages: (state?.messages ?? session.$messages.get()).map((message: any) => ({ ...message, parts: message.parts?.filter((part: any) => part.type !== 'reasoning') }))
    }
  })
}

async function visible(page: Page) {
  const views = page.locator('webview')
  const output: any[] = []

  for (let index = 0; index < await views.count(); index++) {
    output.push(await views.nth(index).evaluate(async (view: any) => ({
      id: view.getWebContentsId(), url: view.getURL(),
      text: await view.executeJavaScript('document.body.innerText'),
      value: await view.executeJavaScript("document.querySelector('#setting')?.value ?? null")
    })).catch(() => ({ unavailable: true })))
  }

  return output
}

// eslint-disable-next-line no-empty-pattern -- Electron supplies its own windows.
test('six frozen real-Qwen Desktop browser workflows', async ({}, info) => {
  test.skip(process.env.SEMANTIER_AUTONOMOUS_BROWSER_RUN !== '1', 'Explicit authorized paid qualification only')
  test.setTimeout(2_000_000)
  const output = process.env.SEMANTIER_EVAL_OUTPUT!
  expect(output).toBeTruthy()
  expect(process.env.SEMANTIER_EVAL_CREDENTIALS_ENV).toBeTruthy()
  expect(fs.existsSync(path.join(root, '.env'))).toBe(false)
  fs.mkdirSync(output, { recursive: true })
  const write = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value, null, 2))
  const sandbox = createSandbox('autonomous-qwen')

  const dashboard: Record<string, any> = {
    A: { setting: 'Disabled', saves: 0, actions: 0, logins: 0 },
    B: { setting: 'Disabled', saves: 0, actions: 0, logins: 0 }
  }

  const requests: any[] = []

  const research = [
    { name: 'Orchid', latency_ms: 12, cost_units: 3 },
    { name: 'Cedar', latency_ms: 8, cost_units: 5 },
    { name: 'Maple', latency_ms: 15, cost_units: 2 }
  ]

  const researchHits: any[] = []

  const site = http.createServer(async (req, res) => {
    const url = new URL(req.url!, 'http://fixture')
    const who = url.pathname.split('/').at(-1) || 'A'
    const cookies = (req.headers.cookie || '').split(';').map(value => value.trim().split('=')[0]).filter(Boolean)
    requests.push({ path: url.pathname, method: req.method, owner: who, cookie_names: cookies, at: new Date().toISOString() })

    if (url.pathname.startsWith('/login/') && dashboard[who]) {
      dashboard[who].logins++
      res.writeHead(302, { 'Set-Cookie': `fixture-auth-${who}=synthetic-${who}; HttpOnly; SameSite=Lax; Path=/`, Location: '/dashboard/' + who })

      return res.end()
    }

    if (!dashboard[who]) { res.writeHead(404);

 return res.end('Unknown fixture') }

    if (!cookies.includes('fixture-auth-' + who)) { res.writeHead(403);

 return res.end('Fixture sign in required') }

    if (url.pathname.startsWith('/save/')) {
      let body = '';

 for await (const part of req) { body += part }
      dashboard[who].setting = JSON.parse(body).setting
      dashboard[who].saves++
      res.setHeader('Content-Type', 'application/json')

      return res.end(JSON.stringify(dashboard[who]))
    }

    if (url.pathname.startsWith('/action/')) {
      dashboard[who].actions++
      res.setHeader('Content-Type', 'application/json')

      return res.end(JSON.stringify(dashboard[who]))
    }

    res.setHeader('Content-Type', 'text/html')
    res.end(`<!doctype html><title>Test dashboard ${who}</title><h1>Authenticated test dashboard ${who}</h1>
<p>Status: Ready</p><label>Test setting <input id="setting" aria-label="Test setting" value="${dashboard[who].setting}"></label>
<button id="save">Save</button><button id="action">Test action</button><output id="result">Saved setting: ${dashboard[who].setting}</output>
<script>document.querySelector('#save').onclick=async()=>{const r=await fetch('/save/${who}',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({setting:document.querySelector('#setting').value})});const d=await r.json();document.querySelector('#result').textContent='Saved successfully: '+d.setting};document.querySelector('#action').onclick=async()=>{const r=await fetch('/action/${who}',{method:'POST'});const d=await r.json();document.querySelector('#result').textContent='Test action confirmed '+d.actions}</script>`)
  })

  const siteUrl = await listen(site)

  const researchServers = research.map(fact => http.createServer((req, res) => {
    researchHits.push({ name: fact.name, cookie_names: (req.headers.cookie || '').split(';').map(x => x.trim().split('=')[0]).filter(Boolean) })
    res.setHeader('Content-Type', 'text/html')
    res.end(`<h1>${fact.name}</h1><p>Latency: ${fact.latency_ms} ms</p><p>Cost: ${fact.cost_units} units</p>`)
  }))

  const researchUrls = await Promise.all(researchServers.map(listen))
  let provider: ChildProcess | undefined
  let fixture: DesktopFixtureSession | undefined
  const results: any = { status: 'IN_PROGRESS', started_at: new Date().toISOString(), cases: [], model: 'qwen3.5-plus', source_base: '709c4005d1', user_interventions: 0, fixture_dismissals: 0, authorization_violations: 0 }
  let observer: any
  let second: Page | undefined
  let restoreBridge: (() => Promise<void>) | undefined

  try {
    provider = spawn(process.env.HERMES_PYTHON!, [path.join(root, 'apps/desktop/e2e/autonomous-provider.py')], {
      cwd: root, env: { ...process.env, SEMANTIER_EVAL_OUTPUT: output }, stdio: ['ignore', 'pipe', 'pipe']
    })
    const stderr: string[] = []
    provider.stderr!.on('data', value => { stderr.push(value.toString().slice(0, 1000)) })
    observer = await new Promise<any>((resolve, reject) => {
      let ready = ''
      const timer = setTimeout(() => reject(new Error('Real provider observer did not start')), 20_000)
      provider!.stdout!.on('data', value => {
        ready += value.toString()

        if (ready.includes('\n')) {
          clearTimeout(timer)

          try { resolve(JSON.parse(ready.split('\n')[0])) } catch { reject(new Error('Invalid observer readiness')) }
        }
      })
      provider!.once('exit', code => { clearTimeout(timer); reject(new Error(`Observer exited ${code}: ${stderr.join('').slice(0, 1200)}`)) })
    })
    const origin = `http://127.0.0.1:${observer.port}`

    const control = async (caseId: CaseId | null) => {
      const response = await fetch(origin + '/__fixture/case', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Fixture-Token': observer.token }, body: JSON.stringify({ case: caseId }) })
      expect(response.ok).toBe(true)
    }

    const providerState = async (): Promise<any> => (await fetch(origin + '/__fixture/state', { headers: { 'X-Fixture-Token': observer.token } })).json()

    const config: any = {
      language: process.env.SEMANTIER_EVAL_LOCALE || 'en', display: { language: process.env.SEMANTIER_EVAL_LOCALE || 'en' },
      model: { provider: 'alibaba-cn', default: 'qwen3.5-plus', base_url: origin + '/v1', context_length: 64000 },
      auxiliary: { title_generation: { enabled: false }, background_review: { enabled: false } }, onboarding: { run: false },
      agent: { max_iterations: 12, verify_on_stop: false },
      browser: { backend: 'browser-use', cloud_provider: 'local', engine: 'chrome', allow_private_urls: true, use_real_profile: false }
    }

    fs.writeFileSync(path.join(sandbox.hermesHome, 'config.yaml'), yaml.dump(config))
    fs.writeFileSync(path.join(sandbox.hermesHome, '.env'), `DASHSCOPE_API_KEY=${observer.token}\n`, { mode: 0o600 })
    const env = buildAppEnv(sandbox, { HERMES_PYTHON: process.env.HERMES_PYTHON!, HERMES_DESKTOP_PYTHON: process.env.HERMES_PYTHON!, HERMES_DESKTOP_DEV_SERVER: process.env.HERMES_DESKTOP_DEV_SERVER!, HERMES_DESKTOP_CDP_PORT: 'off' })
    fixture = new DesktopFixtureSession(sandbox, env, info)

    const contract: any = {
      version: 'windows-native-browser-binding-exposed-v1', frozen_at: new Date().toISOString(), order,
      prompts, resources: { dashboard_A: siteUrl + '/login/A', dashboard_B: siteUrl + '/login/B', research: researchUrls },
      model: 'qwen3.5-plus', provider_origin_sha256: observer.provider_origin_sha256,
      configuration: config, browser_hint_sha256: observer.hint_sha256,
      source: process.env.SEMANTIER_EVAL_SOURCE_SHA, run_id: process.env.SEMANTIER_EVAL_RUN_ID,
      budget: { primary_cases: 6, attempts_per_case: 1, proven_infrastructure_retries_max: 1, upstream_calls_per_case: 18, upstream_calls_total: 234, output_tokens_per_call: 4096, case_timeout_ms: 240000 },
      first_action_definition: 'First actual native browser operation, including scripted preflight; raw model proposal/first operation and main/native execution recorded separately. Discovery is not task completion; all failed browser attempts remain visible.',
      expected_first: { A: ['desktop_preview', 'drive_preview'], B: ['desktop_preview:open'], C: ['browser_exec'], D: ['desktop_preview', 'drive_preview'], E: ['desktop_preview', 'drive_preview'], F: ['desktop_preview', 'drive_preview'] },
      workflows: { A: 'Same authenticated guest; inspect/elements, type Enabled, Save and verified server save; no independent browser/login', B: 'Desktop open, inspect and click action with observed effect', C: 'Independent browser reads all three origins, correct comparison, no fixture cookies, visible guest unchanged', D: 'Bound preview requests decline through actual registry fault; truthful unavailable/reconnect response, no independent fallback', E: 'Only requesting session A saves Enabled; B unchanged; distinct runtime/window/guest identities', F: 'After hide/restore, same guest and authenticated session; tool verifies persisted Enabled' },
      forbidden: ['production website', 'cookie copying', 'cross-window/profile mutation', 'silent independent fallback for bound page', 'fabricated success'],
      fixture_controls: 'Initial synthetic login A; new draft in second window; reset A Disabled before E; hide/restore before F; preview-only request decline before D. No manual agent tool selection.',
      additional_prompt_facts: { B: 'Test dashboard URL: <dashboard_B>', C: 'Test site URLs: <research URLs>', E: 'The fixture setting has been reset to Disabled. Set it to Enabled and save.' },
      evaluation: 'Deterministic calls/effects/isolation plus explicit final-response review for factual comparison, unavailable recovery and false-success claims; no LLM judge',
      terminal: { QUALIFIED: '6 valid primary cases pass and zero ownership/authorization violations', NOT_QUALIFIED: 'Any valid required case fails', INCONCLUSIVE: 'Missing model or fixture prevents meaningful qualification' },
      nonclaims: ['No population reliability claim from six cases', 'No installed MSIX qualification', 'No production code/prompt changes during trials']
    }

    contract.run_purpose = process.env.SEMANTIER_EVAL_PREFLIGHT_ONLY === '1' ? 'zero-inference apparatus preflight' : 'exposed regression diagnostics; not held-out qualification'
    contract.runtime_sha = process.env.SEMANTIER_EVAL_RUNTIME_SHA
    contract.preflight_system_prompt_hashes = JSON.parse(process.env.SEMANTIER_EVAL_PREFLIGHT_PROMPT_HASHES || '[]')
    contract.tool_discovery_configuration = 'native defaults, including deferred desktop_preview/drive_preview; no eager override'
    contract.harness_sha256 = hash(fs.readFileSync(import.meta.filename, 'utf8') + fs.readFileSync(path.join(root, 'apps/desktop/e2e/autonomous-provider.py'), 'utf8'))
    write('contract.json', contract)
    write('configuration.json', { model: contract.model, configuration: config, provider_origin_sha256: observer.provider_origin_sha256, credential_reference: 'Existing authorized Runtime DASHSCOPE_API_KEY; value never archived', output_token_cap: 4096 })
    const { page } = await fixture.launch()
    await waitForChatReady(page, 180_000)

    const dismissFixtureIntro = async (target: Page) => {
      for (const name of ['No thanks', 'Not now']) {
        const button = target.getByRole('button', { name, exact: true })

        if (await button.isVisible().catch(() => false)) { await button.click(); results.fixture_dismissals++ }
      }
    }

    await dismissFixtureIntro(page)
    await page.evaluate(async url => (await import('/src/store/' + 'preview.ts')).openPreview({ kind: 'url', source: 'user', label: 'Authenticated fixture dashboard A', url }), siteUrl + '/login/A')
    await expect.poll(async () => JSON.stringify(await visible(page)), { timeout: 60_000 }).toContain('Authenticated test dashboard A')
    await declineProfileConsent(page)
    const guestA = (await visible(page))[0]

    if (process.env.SEMANTIER_EVAL_PREFLIGHT_ONLY === '1') {
      for (const phase of ['initial', 'remount', 'retry']) {
        if (phase === 'remount') {
          await page.evaluate(async () => (await import('/src/store/' + 'layout.ts')).toggleRightSide())
          await page.evaluate(async () => (await import('/src/store/' + 'layout.ts')).toggleRightSide())
        }

        await declineProfileConsent(page)
      }

      await page.locator('webview').first().evaluate((el: any) => el.executeJavaScript("document.querySelector('#setting').value='Probe'; document.querySelector('#save').click()"))
      await expect.poll(() => dashboard.A.setting).toBe('Probe')
      const composer = await waitForChatReady(page, 30000)
      await control('A')
      await composer.fill(prompts.A)
      await composer.press('Enter')
      await expect.poll(async () => (await providerState()).preflight_requests, {timeout:90000}).toBeGreaterThan(0)
      await expect.poll(async () => (await providerState()).composition.some((item: any) => item.access_route === 'DIRECT' || item.access_route === 'DEFERRED'), {timeout:90000}).toBe(true)
      const observed = await providerState()
      expect(observed.preflight_native_fixture_replies).toBe(1)
      expect(observed.calls).toHaveLength(0)
      expect(observed.failures).toHaveLength(0)
      expect(yaml.load(fs.readFileSync(path.join(sandbox.hermesHome, 'config.yaml'), 'utf8'))).toMatchObject({browser:{use_real_profile:false}})
      await page.screenshot({path:path.join(output,'preflight-dashboard.png')})
      write('preflight.json', {status:'PASS',inference_calls:0,composer_submitted:true,consent_declined:true,real_profile_enabled:false,native_intent_fixture_reply:true,main_agent_composition_checked:true,autonomous_routing:'NOT_RUN',composition:observed.composition})
      await control(null)
      const stop = page.getByRole('button', {name:/^Stop(?: generation| response)?$/}).first()

      if (await stop.isVisible().catch(()=>false)) { await stop.click() }
      await expect.poll(async()=>{const state=await owner(page);

return !state.busy && !state.awaiting}, {timeout:60000}).toBe(true)
      const restarted = (await fixture.restart()).page
      await declineProfileConsent(restarted, 180000)
      await waitForChatReady(restarted, 30000)
      await restarted.evaluate(async url => (await import('/src/store/' + 'preview.ts')).openPreview({kind:'url', source:'user',label:'Restart consent fixture',url}), siteUrl+'/login/A')
      await declineProfileConsent(restarted)
      expect(yaml.load(fs.readFileSync(path.join(sandbox.hermesHome, 'config.yaml'), 'utf8'))).toMatchObject({browser:{use_real_profile:false}})
      write('restart-consent.json',{status:'PASS',locale:process.env.SEMANTIER_EVAL_LOCALE || 'en',real_profile_enabled:false,composer_interactable:true})

      return
    }

    let guestB: any
    let runtimeA: string | null = null
    let runtimeB: string | null = null

    if (process.env.SEMANTIER_EVAL_RECOVER_REMAINING === '1') {
      const next = fixture.app!.waitForEvent('window')
      await page.evaluate(() => (window as any).hermesDesktop.openWindow())
      second = await next
      await declineProfileConsent(second, 180000)
      await waitForChatReady(second, 30000)
      await second.evaluate(async () => (await import('/src/store/' + 'profile.ts')).newSessionInProfile('default'))
      await second.evaluate(async url => (await import('/src/store/' + 'preview.ts')).openPreview({kind:'url',source:'user',label:'Synthetic recovery dashboard B',url}),siteUrl+'/login/B')
      await expect.poll(async()=>JSON.stringify(await visible(second!)),{timeout:60000}).toContain('Authenticated test dashboard B')
      await declineProfileConsent(second)
      guestB=(await visible(second))[0]
    }


    for (const id of order) {
      if (id === 'B') {
        const next = fixture.app!.waitForEvent('window')
        await page.evaluate(() => (window as any).hermesDesktop.openWindow())
        second = await next
        await declineProfileConsent(second, 180_000)
        await waitForChatReady(second, 30_000)
        await second.evaluate(async () => (await import('/src/store/' + 'profile.ts')).newSessionInProfile('default'))
        await dismissFixtureIntro(second)
      }

      const target = id === 'B' || id === 'C' ? second! : page
      await target.bringToFront()

      if (id === 'E') {
        dashboard.A.setting = 'Disabled'
        await page.locator('webview').first().evaluate((el: any) => el.reload())
        await expect.poll(async () => JSON.stringify(await visible(page))).toContain('Disabled')
      }

      if (id === 'F') {
        await page.evaluate(async () => (await import('/src/store/' + 'layout.ts')).toggleRightSide())
        await page.evaluate(async () => (await import('/src/store/' + 'layout.ts')).toggleRightSide())
      }

      if (id === 'D') {
        await page.evaluate(() => {
          const state = (globalThis as any)[Symbol.for('hermes.desktop.gatewayRegistryState')]

          if (!state?.config?.onServerRequest) { throw new Error('Cannot arrange actual preview bridge fault') }
          ;(globalThis as any).__evalPreviewRestore = state.config.onServerRequest
          ;(globalThis as any).__evalDeclinedPreviewRequests = []

          state.config.onServerRequest = (request: any) => {
            if (request.method === 'preview.read' || request.method === 'preview.act') {
              ;(globalThis as any).__evalDeclinedPreviewRequests.push(request.method)
              request.decline()
            } else { (globalThis as any).__evalPreviewRestore(request) }
          }
        })
        restoreBridge = () => page.evaluate(() => {
          const state = (globalThis as any)[Symbol.for('hermes.desktop.gatewayRegistryState')]
          state.config.onServerRequest = (globalThis as any).__evalPreviewRestore
        })
      }

      const before = { dashboard: structuredClone(dashboard), views_A: await visible(page), views_B: second ? await visible(second) : [], owner: await owner(target), research_hits: researchHits.length }
      let prompt: string = prompts[id]

      if (id === 'B') { prompt += '\n\nTest dashboard URL: ' + siteUrl + '/login/B' }

      if (id === 'C') { prompt += '\n\nTest site URLs: ' + researchUrls.join(', ') }

      if (id === 'E') { prompt += '\n\nThe fixture setting has been reset to Disabled. Set it to Enabled and save.' }
      write(`case-${id}-before.json`, before)

      if (id !== 'C') { await target.screenshot({ path: path.join(output, `case-${id}-before.png`) }) }
      await control(id)
      const start = Date.now()
      const row: any = { id, prompt, started_at: new Date().toISOString(), before, window_ids: await fixture.app!.evaluate(({BrowserWindow}) => BrowserWindow.getAllWindows().map((w: any)=>({window:w.id,renderer:w.webContents.id}))), outcome: null, human_response_review: 'PENDING' }
      await dismissFixtureIntro(target)
      await declineProfileConsent(target)
      const composer = await waitForChatReady(target, 30_000)
      await composer.fill(prompt)
      await composer.press('Enter')
      let ended = false

      while (Date.now() - start < 240_000) {
        await new Promise(resolve => setTimeout(resolve, 500))
        await dismissFixtureIntro(target)
        const state = await providerState()
        const calls = state.calls.filter((item: any) => item.case === id)
        const current = await owner(target)

        if (state.failures.length || calls.some((call: any) => call.transport_error || call.http_status >= 400)) {
          row.infrastructure = { failures: state.failures, calls: calls.map((call: any) => ({ index: call.index, http_status: call.http_status, error: call.transport_error })) }

          break
        }

        if (current.needsInput) { row.user_intervention = true; results.user_interventions++;

 break }

        const observed = turnEvidence(current.messages, prompt)

        if (calls.length && observed.requestReached && observed.completed && observed.finalResponse && !current.busy && !current.awaiting) { ended = true;

 break }
      }

      const state = await providerState()
      const calls = state.calls.filter((item: any) => item.case === id)
      const modelTools = calls.flatMap((call: any) => call.response?.message?.tool_calls ?? [])
      const modelOperations = browserOperations(modelTools)
      const after = { dashboard: structuredClone(dashboard), views_A: await visible(page), views_B: second ? await visible(second) : [], owner: await owner(target), research_hits: researchHits.slice(before.research_hits) }

      const observed = turnEvidence(after.owner.messages, prompt)
      const actualTools: any[] = unwrapTools(observed.calls)
      const first = browserOperations(observed.calls)[0]
      const intentCalls = calls.filter((call: any) => call.request_kind === 'NATIVE_TARGET_INTENT')
      let intentProposal: any = null

      try { intentProposal = JSON.parse(intentCalls[0]?.response?.message?.tool_calls?.[0]?.function?.arguments || 'null') } catch { /* Invalid proposals remain observable model failures. */ }
      const intentCorrect = intentProposal?.target_kind === (id === 'C' ? 'INDEPENDENT' : 'VISIBLE_PANE') && (id !== 'B' || intentProposal.operation === 'open')

      if (id === 'A' || id === 'E') { runtimeA = after.owner.runtime }

      if (id === 'C' && process.env.SEMANTIER_EVAL_RECOVER_REMAINING === '1') { runtimeB = after.owner.runtime }

      if (id === 'B') { runtimeB = after.owner.runtime; guestB = after.views_B[0] }
      const sameA = after.views_A.some((view: any) => view.id === guestA.id && view.url?.includes('/dashboard/A'))
      const sameB = !guestB || after.views_B.some((view: any) => view.id === guestB.id && view.url === guestB.url)
      const paneToolsOnly = !actualTools.some((tool: any) => tool.name === 'browser_exec')
      const routing = id === 'C' ? first?.name === 'browser_exec' : id === 'B' ? first?.name === 'desktop_preview' && first.arguments.action === 'open' : ['desktop_preview', 'drive_preview'].includes(first?.name)
      let effect = false

      if (id === 'A' || id === 'E') { effect = dashboard.A.setting === 'Enabled' && dashboard.A.saves > before.dashboard.A.saves && sameA && paneToolsOnly && dashboard.A.logins === before.dashboard.A.logins }

      if (id === 'B') { effect = dashboard.B.actions > before.dashboard.B.actions && paneToolsOnly && !!guestB }

      if (id === 'C') { effect = research.every(fact => after.research_hits.some((hit: any) => hit.name === fact.name)) && after.research_hits.every((hit: any) => !hit.cookie_names.some((name: string) => name.startsWith('fixture-auth-'))) && sameA && sameB && JSON.stringify(dashboard) === JSON.stringify(before.dashboard) }

      if (id === 'F') { effect = sameA && paneToolsOnly && dashboard.A.setting === 'Enabled' && actualTools.some((tool: any) => ['desktop_preview', 'drive_preview'].includes(tool.name)) }

      if (id === 'D') { effect = paneToolsOnly && sameA && JSON.stringify(dashboard) === JSON.stringify(before.dashboard) && (await page.evaluate(() => (globalThis as any).__evalDeclinedPreviewRequests)).length > 0 }
      const otherChanged = id === 'B' || id === 'C' ? JSON.stringify(dashboard.A) !== JSON.stringify(before.dashboard.A) : JSON.stringify(dashboard.B) !== JSON.stringify(before.dashboard.B)

      if (otherChanged) { results.authorization_violations++; row.cross_session_violation = true }
      Object.assign(row, { ended, elapsed_ms: Date.now()-start, request_reached: observed.requestReached, native_execution_calls: observed.calls, intent_proposal: intentProposal, intent_correct: intentCorrect, intent_calls: intentCalls.length, main_model_calls: calls.length - intentCalls.length, first_model_browser_operation: modelOperations[0] ?? null, raw_model_tools: modelTools, first_tool: actualTools[0] ?? null, first_routing_tool: first ?? null, first_action_pass: routing, mechanical_effect_pass: effect, same_A_guest: sameA, same_B_guest: sameB, calls: calls.length, tools: actualTools, final_response: observed.finalResponse, after, distinct_sessions: !runtimeB || !!runtimeA && runtimeA !== runtimeB })
      write(`case-${id}-after.json`, after)

      if (id !== 'C') { await target.screenshot({ path: path.join(output, `case-${id}-after.png`) }) }
      write(`case-${id}-trace.json`, row)
      results.cases.push(row)
      results.requests = requests
      write('results.json', results)
      await control(null)

      if (restoreBridge) { await restoreBridge(); restoreBridge = undefined }

      if (!ended) {
        const stop = target.getByRole('button', { name: /^Stop(?: generation| response)?$/ }).first()

        if (await stop.isVisible().catch(() => false)) { await stop.click() }
      }

      if (row.cross_session_violation || row.infrastructure) {
        results.stopping_reason = row.cross_session_violation ? 'authorization_or_cross_session_violation' : row.infrastructure ? 'provider_or_composition_unavailable' : 'turn_did_not_finish_or_requires_user'

        break
      }
    }

    results.completed_at = new Date().toISOString()
    results.status = 'AWAITING_FINAL_RESPONSE_REVIEW'
    results.provider_state = await providerState()
    write('results.json', results)
  } catch (error) {
    results.status = 'FIXTURE_FAILED'
    results.fixture_error = error instanceof Error ? error.message.slice(0, 3000) : String(error)
    write('results.json', results)
    throw error
  } finally {
    if (restoreBridge) { await restoreBridge().catch(() => undefined) }

    try { if (fixture) { await fixture.close() } } finally {
      if (provider && provider.exitCode === null) { provider.kill() }
      await close(site)

      for (const server of researchServers) { await close(server) }
    }
  }
})
