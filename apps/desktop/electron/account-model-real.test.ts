import { expect, it } from 'vitest'

import { AccountModelCoordinator } from './account-model-coordinator'
import { createSemantierWorkspaceClient } from './semantier-workspace-client'

const origin = process.env.ACCOUNT_MODEL_FIXTURE_ORIGIN
it.skipIf(!origin)('prepares the native device using one authenticated account session', async () => {
  const login = await fetch(origin + '/auth/password/login', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ login: 'coordinator-fixture', password: process.env.ACCOUNT_MODEL_FIXTURE_PASSWORD })
  })
  expect(login.status).toBe(200)
  const cookie = login.headers.getSetCookie().map(value => value.split(';')[0]).join('; ')
  let approvals = 0
  const client = createSemantierWorkspaceClient(origin!, (url, init) => {
    if (url.endsWith('/approve')) {approvals++}
    return fetch(url, { ...init, headers: { ...init?.headers, cookie } })
  })
  const identity = await client.identity()
  expect(identity.authenticated).toBe(true)
  const account = { origin: origin!, scope: { userId: identity.user!.user_id,
    workspaceId: identity.workspace!.id, organizationId: identity.organization_id! } }
  const native = async (path: string, method?: string, body?: unknown) => {
    const response = await fetch(origin + '/__fixture/native', { method: 'POST',
      headers: { 'content-type': 'application/json', authorization: 'Bearer ' + process.env.ACCOUNT_MODEL_FIXTURE_KEY },
      body: JSON.stringify({ path, method, body }) })
    expect(response.status).toBe(200)
    return response.json()
  }
  const owner = new AccountModelCoordinator({ current: () => account, client: () => client,
    native, fence: account => native('fence', 'POST', account) })
  // No model provider is configured in this fixture, so installation must not fabricate readiness.
  expect(await owner.prepare(account)).toBe('unavailable')
  expect(approvals).toBe(1)
  expect((await native('/api/providers/oauth')).providers[0].status.logged_in).toBe(true)
  expect(await owner.preferredModel(account)).toBeNull()
  await owner.suspend()
  expect((await native('/api/providers/oauth')).providers[0].status.logged_in).toBe(false)
}, 30000)
