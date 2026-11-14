import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { setEnvVar } from '@/api/config'
import type { HermesApiRequest } from '@/global'
import { makeOAuthProvider } from '@/test/oauth-provider'

import { $connectionsRegistry } from './connection-registry-state'
import {
  $desktopOnboarding,
  closeManualOnboarding,
  confirmOnboardingModel,
  type OnboardingContext,
  reopenCloudVerification,
  retryCloudModelDiscovery,
  saveOnboardingLocalEndpoint,
  setOnboardingCode,
  setOnboardingModel,
  startManualProviderOAuth,
  startProviderOAuth,
  submitOnboardingCode
} from './onboarding'
import { captureOnboardingScope, requestOnboardingGateway } from './onboarding-scope'
import { $previewTabs, closeRightRail, openPreview } from './preview'
import { $activeSessionId } from './session'

const gatewayMocks = vi.hoisted(() => ({ requestGatewayForAgent: vi.fn(async () => ({ ok: true })) }))

vi.mock('@/store/gateway', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestGatewayForAgent: gatewayMocks.requestGatewayForAgent
}))

const owner = { connectionId: 'athena', profile: 'leverage-ai' }
const requests: HermesApiRequest[] = []
let authFlow: 'device_code' | 'pkce'
let deferStart: (() => Promise<void>) | undefined

beforeEach(() => {
  vi.useFakeTimers()
  $connectionsRegistry.set(null)
  authFlow = 'device_code'
  deferStart = undefined
  closeRightRail()
  requests.length = 0
  setApiRequestConnection(owner.connectionId)
  setApiRequestProfile(owner.profile)
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      openExternal: vi.fn(async () => undefined),
      api: vi.fn(async (request: HermesApiRequest) => {
        requests.push(request)
        const { path } = request

        if (path === '/api/providers/oauth') {
          return { providers: [makeOAuthProvider('openai-codex')] }
        }

        if (path.endsWith('/start')) {
          await deferStart?.()

          return {
            flow: authFlow,
            session_id: 'fixture-session',
            auth_url: 'https://example.invalid/authorize',
            verification_url: 'https://example.invalid/device',
            user_code: 'fixture-code',
            expires_in: 600
          }
        }

        if (path.includes('/poll/') || path.endsWith('/submit')) {
          return { ok: true, status: 'approved' }
        }

        if (path.startsWith('/api/model/options')) {
          return { providers: [{ slug: 'openai-codex', models: ['fixture-model'] }] }
        }

        if (path.startsWith('/api/model/recommended-default')) {
          return { model: 'fixture-model' }
        }

        if (path === '/api/providers/validate') {
          return { ok: true, reachable: true, models: ['endpoint-model'] }
        }

        return { ok: true }
      })
    }
  })
})

afterEach(() => {
  closeManualOnboarding()
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.useRealTimers()
  delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
})

function beginContext(): OnboardingContext {
  startManualProviderOAuth('openai-codex', 'leverage-ai')

  return {
    scope: $desktopOnboarding.get().targetScope,
    requestGateway: vi.fn(
      async method => (method === 'setup.status' ? { provider_configured: true } : { ok: true }) as never
    )
  }
}

function switchForeground() {
  setApiRequestConnection('local')
  setApiRequestProfile('default')
}

function expectOwnerRequests() {
  expect(requests.length).toBeGreaterThan(0)

  for (const request of requests) {
    expect(request).toMatchObject(owner)
  }
}

it.each(['device_code', 'pkce'] as const)(
  'pins %s completion and model edits after a foreground switch',
  async flow => {
    authFlow = flow
    const ctx = beginContext()
    await startProviderOAuth({ ...makeOAuthProvider('openai-codex'), flow }, ctx)
    switchForeground()

    if (flow === 'pkce') {
      setOnboardingCode('fixture-code')
      await submitOnboardingCode(ctx)
    } else {
      await vi.advanceTimersByTimeAsync(3000)
    }

    expect($desktopOnboarding.get().flow.status).toBe('confirming_model')
    expect(ctx.requestGateway).not.toHaveBeenCalledWith('reload.env')
    expect(ctx.requestGateway).toHaveBeenCalledWith('setup.runtime_check', { provider: 'openai-codex' })
    await setOnboardingModel('other-fixture-model', 'openai-codex')
    // Manual flows defer the global-default flip to the confirm card (#102829):
    // completeWithModelConfirm skips its early persist, so the only /api/model/set
    // is the explicit Change pick (persisted on Begin via confirmOnboardingModel
    // when nothing changes). Owner pinning is what this test guards: every write
    // still carries the flow's initiating owner below.
    expect(requests.filter(request => request.path === '/api/model/set')).toHaveLength(1)
    expect(requests.some(request => request.path.includes(flow === 'pkce' ? '/submit' : '/poll/'))).toBe(true)
    expectOwnerRequests()
  }
)

it('cancels on the original owner after the foreground changes', async () => {
  const ctx = beginContext()
  await startProviderOAuth({ ...makeOAuthProvider('openai-codex'), flow: 'device_code' }, ctx)
  switchForeground()
  closeManualOnboarding()
  expect(requests.at(-1)).toMatchObject({ method: 'DELETE', path: '/api/providers/oauth/sessions/fixture-session' })
  expectOwnerRequests()
})

it('cancels a late OAuth start on its retired owner, not a newly opened flow', async () => {
  let release!: () => void
  deferStart = () =>
    new Promise<void>(resolve => {
      release = resolve
    })
  const ctx = beginContext()
  const pending = startProviderOAuth(makeOAuthProvider('openai-codex'), ctx)
  expect(release).toBeTypeOf('function')
  closeManualOnboarding()
  switchForeground()
  startManualProviderOAuth('openai-codex', 'default')
  release()
  await pending
  expect(requests.at(-1)).toMatchObject({ ...owner, method: 'DELETE' })
  expect($desktopOnboarding.get().targetScope).toEqual({ connectionId: 'local', profile: 'default' })
  expect($desktopOnboarding.get().flow.status).toBe('idle')
})

it('pins local-endpoint validation and persistence to the setup owner', async () => {
  const ctx = beginContext()
  switchForeground()
  await saveOnboardingLocalEndpoint('http://127.0.0.1:11434/v1', '', ctx)
  expect(requests.some(request => request.path === '/api/providers/validate')).toBe(true)
  expect(requests.some(request => request.path === '/api/model/set')).toBe(true)
  expectOwnerRequests()
})

it('preserves explicit local and untagged legacy routes instead of filling ambient halves', () => {
  expect(captureOnboardingScope({ connectionId: 'local', profile: 'default' })).toEqual({
    connectionId: 'local',
    profile: 'default'
  })
  expect(captureOnboardingScope({ profile: 'legacy' })).toEqual({ connectionId: null, profile: 'legacy' })
  setApiRequestConnection(null)
  const legacy = captureOnboardingScope('legacy')
  setApiRequestConnection('athena')
  expect(legacy).toEqual({ connectionId: null, profile: 'legacy' })
})

// A setup owner without a profile writes REST settings to the backend's launch
// home (no ?profile=). Readiness on a shared backend must target that same home,
// not a hard-coded `default` profile that may be a different one.
it('keeps readiness on the same profile as REST writes when the owner has none', async () => {
  const getConnectionFor = vi.fn(async () => ({ sharedRemote: true }))
  Object.assign(window.hermesDesktop, { getConnection: vi.fn(), getConnectionFor })
  setApiRequestProfile(null)
  const scope = captureOnboardingScope()

  expect(scope).toEqual({ connectionId: 'athena', profile: null })
  expect(captureOnboardingScope({ connectionId: 'athena', profile: '  ' })).toEqual(scope)

  await setEnvVar('OPENAI_API_KEY', 'fixture', scope)
  expect(requests.at(-1)).toMatchObject({ connectionId: 'athena' })
  expect(requests.at(-1)).not.toHaveProperty('profile')

  await requestOnboardingGateway(scope, 'setup.runtime_check', { provider: 'custom' })
  expect(gatewayMocks.requestGatewayForAgent).toHaveBeenLastCalledWith('athena', 'default', 'setup.runtime_check', {
    provider: 'custom'
  })

  await requestOnboardingGateway(owner, 'setup.runtime_check', {})
  expect(gatewayMocks.requestGatewayForAgent).toHaveBeenLastCalledWith('athena', 'leverage-ai', 'setup.runtime_check', {
    profile: 'leverage-ai'
  })
})

it('defers Cloud installation polls while a different runtime/profile is foreground', async () => {
  const ctx = beginContext()
  await startProviderOAuth({ ...makeOAuthProvider('semantier-cloud'), flow: 'device_code' }, ctx)
  switchForeground()
  await vi.advanceTimersByTimeAsync(6000)
  expect(requests.some(request => request.path.includes('/poll/'))).toBe(false)
  setApiRequestConnection(owner.connectionId)
  setApiRequestProfile(owner.profile)
  await vi.advanceTimersByTimeAsync(3000)
  expect(requests.some(request => request.path.includes('/poll/'))).toBe(true)
  expectOwnerRequests()
})

it('confirms a manual default only on the initiating owner', async () => {
  const ctx = beginContext()
  await startProviderOAuth(makeOAuthProvider('openai-codex'), ctx)
  await vi.advanceTimersByTimeAsync(3000)
  expect($desktopOnboarding.get().flow.status).toBe('confirming_model')
  requests.length = 0
  switchForeground()
  await confirmOnboardingModel(ctx)
  expectOwnerRequests()
})

it('defers Cloud confirmation until its original foreground scope returns', async () => {
  const ctx = beginContext()
  $desktopOnboarding.set({
    ...$desktopOnboarding.get(),
    flow: {
      status: 'confirming_model',
      providerSlug: 'semantier-cloud',
      currentModel: 'fixture-model',
      label: 'Semantier Cloud',
      saving: false
    }
  })
  requests.length = 0
  switchForeground()
  await confirmOnboardingModel(ctx)
  expect(requests).toHaveLength(0)
  expect($desktopOnboarding.get().flow.status).toBe('confirming_model')
  setApiRequestConnection(owner.connectionId)
  setApiRequestProfile(owner.profile)
  await confirmOnboardingModel(ctx)
  expectOwnerRequests()
})

it('observes the same native operation after a lost poll reply without starting another install', async () => {
  const ctx = beginContext()
  const api = window.hermesDesktop.api
  let originalPath = ''
  let observations = 0
  window.hermesDesktop.api = vi.fn(async request => {
    if (request.path.includes('/poll/')) {
      if (!originalPath) {
        originalPath = request.path
        throw new Error('local response lost while native worker continues')
      }

      observations++
      expect(request.path).toBe(originalPath)
      expect(request).toMatchObject(owner)

      return { status: 'pending' }
    }

    return api(request)
  }) as typeof api
  await startProviderOAuth({ ...makeOAuthProvider('semantier-cloud'), flow: 'device_code' }, ctx)
  await vi.advanceTimersByTimeAsync(3000)
  expect(observations).toBe(1)
})

it('deduplicates concurrent Cloud starts on the same owning profile', async () => {
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  deferStart = () => gate
  const ctx = beginContext()
  const provider = { ...makeOAuthProvider('semantier-cloud'), flow: 'device_code' as const }
  const first = startProviderOAuth(provider, ctx)
  const second = startProviderOAuth(provider, ctx)
  expect(requests.filter(request => request.path.endsWith('/start'))).toHaveLength(1)
  release()
  await Promise.all([first, second])
  expect(window.hermesDesktop.openExternal).not.toHaveBeenCalled()
  expect($previewTabs.get().map(tab => tab.target.url)).toContain('https://example.invalid/device?entry=windows')
  expectOwnerRequests()
})

it('shows Cloud recovery instead of the raw Electron transport error', async () => {
  const ctx = beginContext()

  deferStart = async () => { throw new Error('Error invoking remote method \'hermes:api\': Error: 503: {"detail":{"code":"cloud_http3_unavailable","message":"PRIVATE ENDPOINT","profile":"leverage-ai"}}') }
  await startProviderOAuth({ ...makeOAuthProvider('semantier-cloud'), flow: 'device_code' }, ctx)
  const flow = $desktopOnboarding.get().flow
  expect(flow.status).toBe('error')

  if (flow.status !== 'error') { throw new Error('Expected Cloud failure') }
  expect(flow.message).toContain('HTTP/3')
  expect(flow.detail).toContain('cloud_http3_unavailable')
  expect(JSON.stringify(flow)).not.toContain('PRIVATE ENDPOINT')
})

it('starts a reopened same-owner Cloud flow after the retired start is cancelled', async () => {
  let release!: () => void
  deferStart = () => new Promise<void>(resolve => { release = resolve })
  const ctx = beginContext()
  const provider = { ...makeOAuthProvider('semantier-cloud'), flow: 'device_code' as const }
  const retired = startProviderOAuth(provider, ctx)
  closeManualOnboarding()
  startManualProviderOAuth('semantier-cloud', ctx.scope)
  const reopened = startProviderOAuth(provider, ctx)
  expect(requests.filter(request => request.path.endsWith('/start'))).toHaveLength(1)
  deferStart = undefined
  release()
  await Promise.all([retired, reopened])
  expect(requests.filter(request => request.path.endsWith('/start'))).toHaveLength(2)
  expect(requests.filter(request => request.method === 'DELETE')).toHaveLength(1)
  expect($desktopOnboarding.get().flow.status).toBe('polling')
  expectOwnerRequests()
})

it('opens Cloud Billing separately and preserves an authenticated-style tab', async () => {
  const ctx = beginContext()
  const original = openPreview({ kind: 'url', url: 'https://bank.invalid/session', source: 'https://bank.invalid/session', label: 'Existing session' })
  await startProviderOAuth({ ...makeOAuthProvider('semantier-cloud'), flow: 'device_code' }, ctx)
  expect(window.hermesDesktop.openExternal).not.toHaveBeenCalled()
  expect($previewTabs.get()).toHaveLength(2)
  expect($previewTabs.get().find(tab => tab.id === original)?.target.url).toBe('https://bank.invalid/session')
  expect($previewTabs.get().some(tab => tab.target.url === 'https://example.invalid/device?entry=windows')).toBe(true)
})

it('refuses a foreign Cloud open before creating an intent', async () => {
  const ctx = beginContext()
  switchForeground()
  await startProviderOAuth({ ...makeOAuthProvider('semantier-cloud'), flow: 'device_code' }, ctx)
  expect(requests.filter(request => request.path.endsWith('/start'))).toHaveLength(0)
  expect($previewTabs.get()).toHaveLength(0)
  expect(window.hermesDesktop.openExternal).not.toHaveBeenCalled()
})

it('retries installed Cloud model discovery without starting another authorization or selecting another provider', async () => {
  const ctx = beginContext()
  const provider = { ...makeOAuthProvider('semantier-cloud'), flow: 'device_code' as const }
  const task = retryCloudModelDiscovery(ctx, provider)
  await vi.advanceTimersByTimeAsync(2000)
  await task
  expect($desktopOnboarding.get().flow).toMatchObject({ status: 'error', installed: true })
  expect(requests.filter(request => request.path.endsWith('/start'))).toHaveLength(0)
  expect(requests.filter(request => request.path === '/api/model/set')).toHaveLength(0)
})

it('keeps Cloud reopen bound to its initiating chat runtime', async () => {
  const ctx = beginContext()
  const originalRuntime = $activeSessionId.get()

  try {
    await startProviderOAuth({ ...makeOAuthProvider('semantier-cloud'), flow: 'device_code' }, ctx)
    closeRightRail()
    $activeSessionId.set('different-chat-runtime')
    expect(reopenCloudVerification()).toBe(false)
    expect($previewTabs.get()).toHaveLength(0)
    $activeSessionId.set(originalRuntime)
    expect(reopenCloudVerification()).toBe(true)
    expect($previewTabs.get()).toHaveLength(1)
  } finally { $activeSessionId.set(originalRuntime) }
})

it.each([
  ['cloud_server_error', 'Model catalog returned a server error'],
  ['cloud_authorization_denied', 'Model access is denied for this account']
])('keeps installed catalog diagnosis and read recovery for %s', async (code, message) => {
  const ctx = beginContext()
  const original = vi.mocked(window.hermesDesktop.api).getMockImplementation()!
  vi.mocked(window.hermesDesktop.api).mockImplementation(async request => {
    if (request.path.startsWith('/api/model/options')) {
      requests.push(request)

      return { providers: [{ slug: 'semantier-cloud', models: [], discovery_state: 'error', discovery_status: code }] } as never
    }

    return original(request) as never
  })
  const task = retryCloudModelDiscovery(ctx, { ...makeOAuthProvider('semantier-cloud'), flow: 'device_code' })
  await vi.advanceTimersByTimeAsync(2000)
  await task
  expect($desktopOnboarding.get().flow).toMatchObject({ status: 'error', installed: true, message: expect.stringContaining(message) })
  expect(requests.filter(request => request.path.startsWith('/api/model/options')).every(request => new URL(request.path, 'https://fixture.test').searchParams.get('query_scope') === owner.connectionId)).toBe(true)
  expect(requests.filter(request => request.path.endsWith('/start'))).toHaveLength(0)
})


it('opens the selected Cloud login inside Browser Pane even when provider discovery is unavailable', async () => {
  $connectionsRegistry.set({version:2,primary:owner.connectionId,launchMode:'primary',lastUsed:owner.connectionId,connections:[
    {id:owner.connectionId,kind:'cloud',label:'Local Docker',url:'https://127.0.0.1',transport:'semantier-workspace-http'}
  ]} as never)
  window.hermesDesktop.api = vi.fn().mockRejectedValue(new Error('Provider discovery unavailable'))
  startManualProviderOAuth('semantier-cloud', owner)
  await vi.waitFor(() => expect($previewTabs.get().map(tab => tab.target.url)).toContain('https://127.0.0.1/billing?entry=windows'))
  expect(window.hermesDesktop.openExternal).not.toHaveBeenCalled()
  expect(window.hermesDesktop.api).not.toHaveBeenCalled()
})

it('opens the configured Cloud portal before attempting native provider discovery', async () => {
  const api = window.hermesDesktop.api
  window.hermesDesktop.api = vi.fn(async request => {
    if (request.path.startsWith('/api/config')) {
      return {providers:{'semantier-cloud':{base_url:'https://inference.example/v1',portal_origin:'https://accounts.example'}}}
    }

    return api(request)
  }) as typeof api
  startManualProviderOAuth('semantier-cloud', owner)
  await vi.waitFor(() => expect($previewTabs.get().map(tab => tab.target.url)).toContain('https://accounts.example/billing?entry=windows'))
  expectOwnerRequests()
  expect(window.hermesDesktop.openExternal).not.toHaveBeenCalled()
})


it('keeps the public Cloud login available when native settings cannot be read', async () => {
  window.hermesDesktop.api = vi.fn().mockRejectedValue(new Error('Native runtime unavailable'))
  startManualProviderOAuth('semantier-cloud', owner)
  await vi.waitFor(() => expect($previewTabs.get().map(tab => tab.target.url)).toContain('https://cloud.semantier.cn/billing?entry=windows'))
  expect(window.hermesDesktop.openExternal).not.toHaveBeenCalled()
})

it('does not open a late Cloud login after the foreground owner changes', async () => {
  let release!: (config: unknown) => void
  window.hermesDesktop.api = vi.fn(() => new Promise(resolve => { release = resolve })) as typeof window.hermesDesktop.api
  const { openCloudSignIn } = await import('./cloud-sign-in')
  const opening = openCloudSignIn(owner)
  switchForeground()
  release({providers:{'semantier-cloud':{portal_origin:'https://accounts.example'}}})
  await opening
  expect($previewTabs.get()).toHaveLength(0)
})
