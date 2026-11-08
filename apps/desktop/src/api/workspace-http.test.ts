import { afterEach, expect, it, vi } from 'vitest'

import { HermesGateway, hermesApi, setApiRequestConnection, forgetWorkspaceReadScope } from './client'

const scope = { userId: 'a', workspaceId: 'wa', organizationId: 'org' }
const identity = { authenticated: true, user: { user_id: 'a', workspace_id: 'wa' }, workspace: { id: 'wa' }, organization_id: 'org' }

afterEach(() => vi.unstubAllGlobals())

it('opens HTTP with real identity checks and resumes exact server IDs without a WebSocket or model turn', async () => {
  const history = vi.fn().mockResolvedValue({ sessionId: 'original', messages: [{ messageId: '204', role: 'assistant', text: 'Reminder' }] })
  const send = vi.fn()
  const socket = vi.fn(() => { throw new Error('Cloud must not dial a socket') })
  vi.stubGlobal('WebSocket', socket)
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: { identity: vi.fn().mockResolvedValue(identity), history, send } } })
  const gateway = new HermesGateway()
  const ready = vi.fn()
  gateway.on('gateway.ready', ready)
  await gateway.connectWorkspace('https://cloud.example', scope)
  expect(gateway.connectionState).toBe('open')
  const result = await gateway.request('session.resume', { session_id: 'original' })
  expect(result).toMatchObject({ session_id: 'original', messages: [{ messageId: '204', row_id: 204 }] })
  expect(socket).not.toHaveBeenCalled()
  expect(ready).not.toHaveBeenCalled()
  expect(send).not.toHaveBeenCalled()
  await expect(gateway.request('session.rewind', { session_id: 'original' })).rejects.toMatchObject({ code: -32601 })
  gateway.close()
})

it('rejects a changed account before opening the HTTP connection', async () => {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: { identity: vi.fn().mockResolvedValue({ ...identity, organization_id: 'other' }) } } })
  const gateway = new HermesGateway()
  await expect(gateway.connectWorkspace('https://cloud.example', scope)).rejects.toThrow('changed')
  expect(gateway.connectionState).toBe('error')
  gateway.close()
})

it('pins REST reads to the identity that actually opened the Cloud connection', async () => {
  const api = vi.fn().mockResolvedValue({ sessions: [] })
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api, workspace: { identity: vi.fn().mockResolvedValue(identity) } } })
  const gateway = new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example', scope, 'cloud-a')
  setApiRequestConnection('cloud-a')
  await hermesApi({ path: '/api/sessions' })
  expect(api).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'cloud-a', workspaceScope: scope }))
  setApiRequestConnection(null)
  gateway.close()
})

it('uses server-owned Cloud admission for readiness and a real context read for ping', async () => {
  const read = vi.fn().mockResolvedValue({ ...identity, account_provisioning: { inference_ready: true } })
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: { identity: read } } })
  const gateway = new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example', scope)
  await expect(gateway.request('ping')).resolves.toEqual({ ok: true })
  await expect(gateway.request('setup.runtime_check')).resolves.toMatchObject({ ok: true })
  read.mockResolvedValue({ ...identity, account_provisioning: { inference_ready: false, reason: 'admission revoked' } })
  await expect(gateway.request('setup.runtime_check')).resolves.toEqual({ ok: false, error: 'admission revoked' })
  gateway.close()
})

it('reads the Native Winman roster through the explicit account-scoped HTTP owner', async () => {
  const roster = { profiles: [{ name: 'winman-a', runtime_bot_id: 'bot-a',
    canonical_session: { id: 'canonical-a' } }], bot_mode_protocol: true }
  const api = vi.fn().mockResolvedValue(roster)
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {
    api, workspace: { identity: vi.fn().mockResolvedValue(identity) }
  } })
  const gateway = new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example', scope, 'cloud-a')
  await expect(gateway.request('profiles.list', {})).resolves.toEqual(roster)
  expect(api).toHaveBeenCalledWith(expect.objectContaining({
    connectionId: 'cloud-a', workspaceScope: scope, profile: 'default', path: '/api/profiles'
  }))
  gateway.close()
})

it('resolves Bot Chat by its server title registry only for the selected Native profile', async () => {
  const api = vi.fn().mockResolvedValue({ profiles: [{ name: 'winman-a',
    canonical_session: { id: 'canonical-a', resolved_id: 'tip-a', title: 'Bot Chat', message_count: 7 } }] })
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {
    api, workspace: { identity: vi.fn().mockResolvedValue(identity) }
  } })
  const gateway = new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example', scope, 'cloud-a')
  await expect(gateway.request('session.list', { profile: 'winman-a', title: 'Bot Chat', include_hidden: true }))
    .resolves.toMatchObject({ sessions: [{ id: 'canonical-a', resolved_id: 'tip-a' }], scope: 'stored' })
  await expect(gateway.request('session.list', { profile: 'foreign', title: 'Bot Chat' })).rejects.toThrow('profile')
  await expect(gateway.request('session.list', { profile: 'winman-a', title: 'Other' })).rejects.toMatchObject({ code: -32601 })
  gateway.close()
})

it('resumes the explicit Native Cloud profile through its owner and never sends it to Main history', async () => {
  const snapshot = { session_id: 'canonical-a', messages: [{ row_id: 44, role: 'assistant', content: 'Native reply', timestamp: 10 }] }
  const nativeHistory = vi.fn().mockResolvedValue(snapshot)
  const history = vi.fn()
  const send = vi.fn()
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {
    workspace: { identity: vi.fn().mockResolvedValue(identity), nativeHistory, history, send }
  } })
  const gateway = new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example', scope, 'cloud-a', 'winman-a')
  await expect(gateway.request('session.resume', { session_id: 'canonical-a' })).resolves.toEqual(snapshot)
  expect(nativeHistory).toHaveBeenCalledWith('https://cloud.example', scope, 'canonical-a')
  expect(history).not.toHaveBeenCalled()
  await expect(gateway.request('prompt.submit', { session_id: 'canonical-a', text: 'Hello', submission_id: 'stable' })).rejects.toThrow('owner adapter')
  expect(send).not.toHaveBeenCalled()
  gateway.close()
})


it('rejects a late account response before it can enter renderer caches', async () => {
  let resolve!: (value: unknown) => void
  const api = vi.fn(() => new Promise(done => { resolve = done }))
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {
    api, workspace: { identity: vi.fn().mockResolvedValue(identity) }
  } })
  const gateway = new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example', scope, 'cloud-a')
  setApiRequestConnection('cloud-a')
  const pending = hermesApi({ path: '/api/sessions' })
  forgetWorkspaceReadScope('cloud-a')
  resolve({ sessions: [{ id: 'A-private' }] })
  await expect(pending).rejects.toThrow('changed during this read')
  setApiRequestConnection(null)
  gateway.close()
})

it('keeps concurrent reads valid when another profile opens for the same authenticated account', async () => {
  let resolve!: (value: unknown) => void
  const api = vi.fn(() => new Promise(done => { resolve = done }))
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {
    api, workspace: { identity: vi.fn().mockResolvedValue(identity) }
  } })
  const primary = new HermesGateway()
  const bot = new HermesGateway()
  await primary.connectWorkspace('https://cloud.example', scope, 'cloud-a')
  setApiRequestConnection('cloud-a')
  const pending = hermesApi({ path: '/api/sessions' })
  await bot.connectWorkspace('https://cloud.example', { ...scope }, 'cloud-a', 'winman-a')
  resolve({ sessions: [{ id: 'own-session' }] })
  await expect(pending).resolves.toEqual({ sessions: [{ id: 'own-session' }] })
  setApiRequestConnection(null)
  primary.close()
  bot.close()
})
