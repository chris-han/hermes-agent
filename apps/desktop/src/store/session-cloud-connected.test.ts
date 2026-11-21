import { expect, it, vi } from 'vitest'

import { createGatewaySession, desktopSessionCreateParams } from '@/app/session/hooks/use-session-actions/session-create-request'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $connection, setCurrentModel, setCurrentModelSource, setCurrentProvider } from '@/store/session'

import { AccountModelCoordinator } from '../../electron/account-model-coordinator'
import { createSemantierWorkspaceClient } from '../../electron/semantier-workspace-client'

vi.mock('@/store/profile', async importOriginal => ({
  ...await importOriginal<Record<string, unknown>>(),
  ensureGatewayProfile: vi.fn(async () => undefined)
}))

const origin = process.env.ACCOUNT_MODEL_FIXTURE_ORIGIN
it.skipIf(!origin)('passes the connected account default to the standalone local session owner', async () => {
  const login = await fetch(origin + '/auth/password/login', { method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ login: 'coordinator-fixture', password: process.env.ACCOUNT_MODEL_FIXTURE_PASSWORD }) })
  expect(login.status).toBe(200)
  const cookie = login.headers.getSetCookie().map(value => value.split(';')[0]).join('; ')
  const client = createSemantierWorkspaceClient(origin!, (url, init) => fetch(url, {
    ...init, headers: { ...init?.headers, cookie }
  }))
  const identity = await client.identity()
  const account = { origin: origin!, scope: { userId: identity.user!.user_id,
    workspaceId: identity.workspace!.id, organizationId: identity.organization_id! } }
  const native = async (path: string, method?: string, body?: unknown) => {
    const response = await fetch(origin + '/__fixture/native', { method: 'POST',
      headers: { 'content-type': 'application/json', authorization: 'Bearer ' + process.env.ACCOUNT_MODEL_FIXTURE_KEY },
      body: JSON.stringify({ path, method, body }) })
    const value = await response.json()
    expect(response.status, JSON.stringify(value)).toBe(200)
    return value
  }
  const owner = new AccountModelCoordinator({ current: () => account, client: () => client,
    native, fence: value => native('fence', 'POST', value) })
  const original = Object.getOwnPropertyDescriptor(window, 'hermesDesktop')
  try {
    expect(await owner.prepare(account)).toBe('ready')
    const model = await owner.preferredModel(account)
    expect(model).toBe('deepseek-v4.1-flash-1017')
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: {
      modelDefault: async () => (await native('/api/providers/oauth/semantier-cloud/account-default-eligibility')).allowed
        ? { account, model } : null
    } } })
    $connection.set({ mode: 'local', connectionId: 'local', baseUrl: 'http://127.0.0.1' } as never)
    $connectionsRegistry.set({ connections: [], accountContext: account } as never)
    setCurrentModel('auto-local'); setCurrentProvider('local'); setCurrentModelSource('default')
    const params = await desktopSessionCreateParams('', null, 'default')
    expect(params).toMatchObject({ provider: 'semantier-cloud', model })
    let override: unknown
    const session = await createGatewaySession(null, params, async <T,>(method: string, values?: Record<string, unknown>) => {
      const result = await native('fixture-local-session', 'POST', { method, values })
      override = result.model_override
      return result.result as T
    })
    expect(session.session_id).toBeTruthy()
    expect(override).toMatchObject({ provider: 'semantier-cloud', model })
    const turn = await native('fixture-local-turn', 'POST', { session_id: session.session_id })
    expect(turn.completed).toBe(true)
    expect(turn.agent_error).toBeNull()
    expect(turn.history).toEqual(expect.arrayContaining([expect.objectContaining({
      role: 'assistant', content: 'fixture response'
    })]))
    expect(turn.title_started).toBe(true)
    expect(turn.title_finished).toBe(true)
    expect(turn.after_switch_completed).toBe(true)
    expect(turn.compression_bound).toBe(true)
    expect(turn.compression_refused).toBe(true)
    expect(turn.vision_refused).toBe(true)
    expect(turn.after_switch_history.filter((item: { role: string; content: string }) =>
      item.role === 'assistant' && item.content === 'fixture response')).toHaveLength(1)
  } finally {
    await owner.suspend()
    $connection.set(null); $connectionsRegistry.set(null)
    setCurrentModel(''); setCurrentProvider(''); setCurrentModelSource('')
    if (original) {Object.defineProperty(window, 'hermesDesktop', original)} else {delete (window as { hermesDesktop?: unknown }).hermesDesktop}
  }
}, 75000)
