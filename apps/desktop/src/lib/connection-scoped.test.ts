import { describe, expect, it } from 'vitest'

import { connectionScopeSuffix, connectionScopedAtom, rescopeConnectionScopedStores } from './connection-scoped'

const remote = (profile: string, baseUrl = 'https://gw.example:8443') => ({
  baseUrl,
  mode: 'remote' as const,
  profile
})

describe('connectionScopeSuffix', () => {
  it('is empty for a local connection', () => {
    expect(connectionScopeSuffix({ baseUrl: 'http://127.0.0.1:8000', mode: 'local', profile: 'default' })).toBe('')
  })

  it('includes the profile by default so profile-local lists stay apart', () => {
    expect(connectionScopeSuffix(remote('default'))).toBe(
      `.remote.${encodeURIComponent('https://gw.example:8443')}.default`
    )
    expect(connectionScopeSuffix(remote('k9'))).not.toBe(connectionScopeSuffix(remote('default')))
  })

  it('is stable across profile switch when includeProfile is false', () => {
    const a = connectionScopeSuffix(remote('default'), false)
    const b = connectionScopeSuffix(remote('k9'), false)

    expect(a).toBe(`.remote.${encodeURIComponent('https://gw.example:8443')}`)
    expect(a).toBe(b)
  })

  it('still isolates different gateways when includeProfile is false', () => {
    expect(connectionScopeSuffix(remote('default', 'https://gw-a.example'), false)).not.toBe(
      connectionScopeSuffix(remote('default', 'https://gw-b.example'), false)
    )
  })
})

it('separates accounts and signed-out state on the same Semantier Workspace origin', () => {
  const a = { ...remote('default'), transport: 'semantier-workspace-http' as const,
    workspaceScope: { userId: 'a', workspaceId: 'wa', organizationId: 'org' } }
  const b = { ...a, workspaceScope: { userId: 'b', workspaceId: 'wb', organizationId: 'org' } }
  const signedOut = { ...a, workspaceScope: undefined }
  expect(connectionScopeSuffix(a)).not.toBe(connectionScopeSuffix(b))
  expect(connectionScopeSuffix(a, false)).not.toBe(connectionScopeSuffix(b, false))
  expect(connectionScopeSuffix(signedOut)).not.toBe(connectionScopeSuffix(a))
  expect(connectionScopeSuffix(signedOut)).not.toBe(connectionScopeSuffix(remote('default')))
})


it('loads account-owned persisted values only after that account is authenticated again', () => {
  const a = { ...remote('default'), transport: 'semantier-workspace-http' as const,
    workspaceScope: { userId: 'a', workspaceId: 'wa', organizationId: 'org' } }
  const b = { ...a, workspaceScope: { userId: 'b', workspaceId: 'wb', organizationId: 'org' } }
  const value = connectionScopedAtom<string[]>('account-isolation-fixture', [])
  rescopeConnectionScopedStores(a)
  value.set(['A-private'])
  rescopeConnectionScopedStores(b)
  expect(value.get()).toEqual([])
  value.set(['B-private'])
  rescopeConnectionScopedStores({ ...a, workspaceScope: undefined })
  expect(value.get()).toEqual([])
  rescopeConnectionScopedStores(a)
  expect(value.get()).toEqual(['A-private'])
})
