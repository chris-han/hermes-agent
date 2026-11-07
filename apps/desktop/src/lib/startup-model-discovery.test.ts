import { beforeEach, describe, expect, it, vi } from 'vitest'

const requestModelOptions = vi.fn()
const requestGatewayForAgent = vi.fn()
const activeGatewayConnectionId = vi.fn(() => 'source-b')
const invalidateQueries = vi.fn()

vi.mock('@/lib/model-options', () => ({ requestModelOptions: (...args: unknown[]) => requestModelOptions(...args) }))
vi.mock('@/store/gateway', () => ({
  activeGatewayConnectionId: () => activeGatewayConnectionId(),
  requestGatewayForAgent: (...args: unknown[]) => requestGatewayForAgent(...args)
}))
vi.mock('@/lib/query-client', () => ({ queryClient: { invalidateQueries: (...args: unknown[]) => invalidateQueries(...args) } }))
vi.mock('@/store/session', () => ({ $connection: { get: () => null } }))

beforeEach(() => {
  vi.resetModules()
  vi.clearAllMocks()
  requestModelOptions.mockResolvedValue({ discovery_contract: 'provider-query-v1', providers: [] })
  invalidateQueries.mockResolvedValue(undefined)
})

describe('connection-ready model discovery', () => {
  it('refreshes each connection and profile only once without blocking readiness', async () => {
    let release!: (value: unknown) => void
    requestModelOptions.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
    const { refreshModelsOnConnectionReady } = await import('./startup-model-discovery')
    const { $refreshingModelSources, modelDiscoveryRefreshKey } = await import('@/store/model-discovery')
    const first = refreshModelsOnConnectionReady('source-a', 'profile-a')
    await Promise.resolve()
    expect($refreshingModelSources.get().has(modelDiscoveryRefreshKey('source-a', 'profile-a'))).toBe(true)
    await refreshModelsOnConnectionReady('source-a', 'profile-a')
    expect(requestModelOptions).toHaveBeenCalledTimes(1)
    expect(requestModelOptions.mock.calls[0][0]).toMatchObject({ queryOnly: true, queryScope: 'source-a', refresh: true, profile: 'profile-a' })
    await refreshModelsOnConnectionReady('source-b', 'profile-a')
    await refreshModelsOnConnectionReady('source-a', 'profile-b')
    expect(requestModelOptions).toHaveBeenCalledTimes(3)
    release({ discovery_contract: 'provider-query-v1', providers: [] })
    await first
    expect($refreshingModelSources.get().has(modelDiscoveryRefreshKey('source-a', 'profile-a'))).toBe(false)
    expect(invalidateQueries).toHaveBeenCalledTimes(3)
  })

  it('keeps failure nonblocking and does not repeat on reconnect', async () => {
    requestModelOptions.mockRejectedValueOnce(new Error('offline'))
    const { refreshModelsOnConnectionReady } = await import('./startup-model-discovery')
    await expect(refreshModelsOnConnectionReady('source-a', 'profile-a')).resolves.toBeUndefined()
    await refreshModelsOnConnectionReady('source-a', 'profile-a')
    expect(requestModelOptions).toHaveBeenCalledTimes(1)
  })

  it('invalidates only this connection and profile after an old refresh completes', async () => {
    const { refreshModelsOnConnectionReady } = await import('./startup-model-discovery')
    await refreshModelsOnConnectionReady('source-a', 'profile-a')
    const predicate = invalidateQueries.mock.calls[0][0].predicate
    expect(predicate({ queryKey: ['model-options', 'profile-a', 'global', 'owner', 'source-a'] })).toBe(true)
    expect(predicate({ queryKey: ['model-options', 'profile-a', 'global', 'owner', 'source-b'] })).toBe(false)
    expect(predicate({ queryKey: ['model-options', 'profile-a', 'global'] })).toBe(false)
    expect(predicate({ queryKey: ['model-options', 'profile-b', 'global', 'owner', 'source-a'] })).toBe(false)
  })
})
