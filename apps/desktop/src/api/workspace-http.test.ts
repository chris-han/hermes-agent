import { afterEach, expect, it, vi } from 'vitest'

import { HermesGateway, hermesApi, setApiRequestConnection } from './client'

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
