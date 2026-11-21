import { afterEach, expect, it, vi } from 'vitest'

import { desktopSessionCreateParams } from '@/app/session/hooks/use-session-actions/session-create-request'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import {
  $connection,
  $currentModel,
  resetCloudComposerForFreshDraft,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider
} from '@/store/session'

vi.mock('@/store/profile', async importOriginal => ({
  ...await importOriginal<Record<string, unknown>>() ,
  ensureGatewayProfile: vi.fn(async () => undefined)
}))

afterEach(() => {
  setCurrentModel('')
  setCurrentProvider('')
  setCurrentModelSource('')
  $connection.set(null)
  $connectionsRegistry.set(null)
})

it('uses the currently selected authorized account default only for a fresh local draft', async () => {
  const original = Object.getOwnPropertyDescriptor(window, 'hermesDesktop')
  const account = { origin: 'https://cloud.example', scope: { userId: 'a', workspaceId: 'wa', organizationId: 'oa' } }
  const modelDefault = vi.fn().mockResolvedValue({ account, model: 'eligible-model' })
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: {
    modelDefault
  } } })
  try {
    $connection.set({ mode: 'local', connectionId: 'local', baseUrl: 'http://127.0.0.1' } as never)
    $connectionsRegistry.set({ connections: [], accountContext: account } as never)
    setCurrentModel(''); setCurrentProvider(''); setCurrentModelSource('default')
    expect(await desktopSessionCreateParams('', null, 'default')).toMatchObject({ provider: 'semantier-cloud', model: 'eligible-model' })
    setCurrentModel('my-local-model'); setCurrentProvider('custom:byok'); setCurrentModelSource('manual')
    expect(await desktopSessionCreateParams('', null, 'default')).toMatchObject({ provider: 'custom:byok', model: 'my-local-model' })
    setCurrentModelSource('default')
    modelDefault.mockResolvedValueOnce(null) // The native owner refuses a saved BYOK model.
    expect(await desktopSessionCreateParams('', null, 'default')).not.toHaveProperty('model')
    setCurrentModel(''); setCurrentProvider(''); setCurrentModelSource('default')
    $connectionsRegistry.set({ connections: [], accountContext: { ...account, scope: { ...account.scope, userId: 'other' } } } as never)
    expect(await desktopSessionCreateParams('', null, 'default')).not.toHaveProperty('model')
    $connection.set({ mode: 'remote', connectionId: 'remote', baseUrl: 'https://remote.example' } as never)
    $connectionsRegistry.set({ connections: [], accountContext: account } as never)
    expect(await desktopSessionCreateParams('', null, 'default')).not.toHaveProperty('model')
  } finally {
    if (original) {Object.defineProperty(window, 'hermesDesktop', original)} else {delete (window as { hermesDesktop?: unknown }).hermesDesktop}
  }
})

it('uses an eligible account default when the displayed provider was auto-detected', async () => {
  const original = Object.getOwnPropertyDescriptor(window, 'hermesDesktop')
  const account = { origin: 'https://cloud.example', scope: { userId: 'a', workspaceId: 'wa', organizationId: 'oa' } }
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: {
    modelDefault: vi.fn().mockResolvedValue({ account, model: 'eligible-model' })
  } } })
  try {
    $connection.set({ mode: 'local', connectionId: 'local', baseUrl: 'http://127.0.0.1' } as never)
    $connectionsRegistry.set({ connections: [], accountContext: account } as never)
    setCurrentModel('auto-detected-model'); setCurrentProvider('local'); setCurrentModelSource('default')
    expect(await desktopSessionCreateParams('', null, 'default')).toMatchObject({ provider: 'semantier-cloud', model: 'eligible-model' })
  } finally {
    if (original) {Object.defineProperty(window, 'hermesDesktop', original)} else {delete (window as { hermesDesktop?: unknown }).hermesDesktop}
  }
})

it('does not apply a default if the account changes while the draft is prepared', async () => {
  const original = Object.getOwnPropertyDescriptor(window, 'hermesDesktop')
  const account = { origin: 'https://cloud.example', scope: { userId: 'a', workspaceId: 'wa', organizationId: 'oa' } }
  let release!: (value: unknown) => void
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: {
    modelDefault: vi.fn(() => new Promise(resolve => {release = resolve}))
  } } })
  try {
    $connection.set({ mode: 'local', connectionId: 'local', baseUrl: 'http://127.0.0.1' } as never)
    $connectionsRegistry.set({ connections: [], accountContext: account } as never)
    setCurrentModel(''); setCurrentProvider(''); setCurrentModelSource('default')
    const pending = desktopSessionCreateParams('', null, 'default')
    $connectionsRegistry.set({ connections: [], accountContext: { ...account, scope: { ...account.scope, userId: 'b' } } } as never)
    release({ account, model: 'eligible-model' })
    expect(await pending).not.toHaveProperty('model')
  } finally {
    if (original) {Object.defineProperty(window, 'hermesDesktop', original)} else {delete (window as { hermesDesktop?: unknown }).hermesDesktop}
  }
})

it.each([
  ['same local connection refresh', { mode: 'local', connectionId: 'local', baseUrl: 'http://127.0.0.1', wsUrl: 'ws://127.0.0.1' }, true],
  ['different connection', { mode: 'remote', connectionId: 'other', baseUrl: 'https://other', wsUrl: 'wss://other' }, false],
  ['different local endpoint', { mode: 'local', connectionId: 'local', baseUrl: 'http://127.0.0.2', wsUrl: 'ws://127.0.0.2' }, false]
])('handles %s while the default is loading', async (_label, next, accepted) => {
  const original = Object.getOwnPropertyDescriptor(window, 'hermesDesktop')
  const account = { origin: 'https://cloud.example', scope: { userId: 'a', workspaceId: 'wa', organizationId: 'oa' } }
  let release!: (value: unknown) => void
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: {
    modelDefault: vi.fn(() => new Promise(resolve => {release = resolve}))
  } } })
  try {
    $connection.set({ mode: 'local', connectionId: 'local', baseUrl: 'http://127.0.0.1', wsUrl: 'ws://127.0.0.1' } as never)
    $connectionsRegistry.set({ connections: [], accountContext: account } as never)
    setCurrentModelSource('default')
    const pending = desktopSessionCreateParams('', null, 'default')
    $connection.set(next as never)
    release({ account, model: 'eligible-model' })
    const params = await pending
    if (accepted) {expect(params).toMatchObject({ provider: 'semantier-cloud', model: 'eligible-model' })}
    else {expect(params).not.toHaveProperty('model')}
  } finally {
    if (original) {Object.defineProperty(window, 'hermesDesktop', original)} else {delete (window as { hermesDesktop?: unknown }).hermesDesktop}
  }
})

it.each(['semantier-cloud', 'custom:byok'])('fresh draft has the correct model policy for %s', async provider => {
  setCurrentModel('explicit-session-model')
  setCurrentProvider(provider)
  setCurrentModelSource('manual')
  resetCloudComposerForFreshDraft()
  const params = await desktopSessionCreateParams('', null, 'default')

  if (provider === 'semantier-cloud') {
    expect($currentModel.get()).toBe('')
    expect(params).not.toHaveProperty('model')
    expect(params).not.toHaveProperty('provider')
  } else {
    expect($currentModel.get()).toBe('explicit-session-model')
    expect(params).toMatchObject({model:'explicit-session-model',provider})
  }
})
