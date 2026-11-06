import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'
import { $settingsScopeOverride } from '@/store/settings-scope'

import { useSettingsSearchCatalog } from './use-settings-search'

const getOptions = vi.fn()
vi.mock('@/hermes', () => ({
  getEnvVars: async () => ({}),
  getHermesConfigSchema: async () => ({ fields: {} }),
  getGlobalModelOptions: (...args: unknown[]) => getOptions(...args),
  setApiRequestProfile: vi.fn()
}))
vi.mock('../hooks/use-config-record', () => ({
  useHermesConfigRecord: () => ({ data: {}, isFetching: false, isError: false })
}))
vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
  useGatewayRequest: () => ({ requestGateway: vi.fn() })
}))

beforeEach(() => {
  $connection.set({ mode: 'local', connectionId: 'source-a' } as never)
  $settingsScopeOverride.set('profile-a')
})
afterEach(() => {
  cleanup()
  $settingsScopeOverride.set(null)
  $connection.set(null)
  vi.clearAllMocks()
})

describe('model search catalog scope', () => {
  it('ignores a late old-source catalog after switching connections', async () => {
    let finishA!: (value: unknown) => void
    getOptions.mockReturnValueOnce(new Promise(resolve => { finishA = resolve }))
      .mockResolvedValueOnce({ providers: [{ slug: 'b', name: 'B', models: ['b-model'], eligible_models: ['b-model'] }] })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
    const { result } = renderHook(() => useSettingsSearchCatalog(true), { wrapper })
    await waitFor(() => expect(getOptions).toHaveBeenCalledTimes(1))
    act(() => $connection.set({ mode: 'local', connectionId: 'source-b' } as never))
    await waitFor(() => expect(result.current.modelEntries.some(entry => entry.label === 'b-model')).toBe(true))
    await act(async () => { finishA({ providers: [{ slug: 'a', name: 'A', models: ['a-model'], eligible_models: ['a-model'] }] }) })
    expect(result.current.modelEntries.some(entry => entry.label === 'a-model')).toBe(false)
    expect(result.current.modelEntries[0].target.modelConnection).toBe('source-b')
  })

  it('indexes an untagged remote and drops its late results after the connection changes', async () => {
    let finishA!: (value: unknown) => void
    $connection.set({ mode: 'remote', baseUrl: 'https://first.example', token: 'secret-a' } as never)
    getOptions.mockReturnValueOnce(new Promise(resolve => { finishA = resolve }))
      .mockResolvedValueOnce({ providers: [{ slug: 'b', name: 'B', models: ['b-model'], eligible_models: ['b-model'] }] })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
    const { result } = renderHook(() => useSettingsSearchCatalog(true), { wrapper })
    await waitFor(() => expect(getOptions).toHaveBeenCalledTimes(1))
    act(() => $connection.set({ mode: 'remote', baseUrl: 'https://second.example', token: 'secret-b' } as never))
    await waitFor(() => expect(result.current.modelEntries.some(entry => entry.label === 'b-model')).toBe(true))
    const target = result.current.modelEntries[0].target.modelConnection!
    expect(target).not.toContain('second.example')
    expect(target).not.toContain('secret-b')
    await act(async () => { finishA({ providers: [{ slug: 'a', name: 'A', models: ['a-model'], eligible_models: ['a-model'] }] }) })
    expect(result.current.modelEntries.some(entry => entry.label === 'a-model')).toBe(false)
  })

  it('drops old profile results immediately and indexes the new scoped response', async () => {
    let finishB!: (value: unknown) => void
    getOptions.mockResolvedValueOnce({ providers: [{ slug: 'a', name: 'A', models: ['a-model'], eligible_models: ['a-model'] }] })
      .mockReturnValueOnce(new Promise(resolve => { finishB = resolve }))
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
    const { result } = renderHook(() => useSettingsSearchCatalog(true), { wrapper })
    await waitFor(() => expect(result.current.modelEntries.some(entry => entry.label === 'a-model')).toBe(true))
    act(() => $settingsScopeOverride.set('profile-b'))
    expect(result.current.modelEntries).toEqual([])
    await act(async () => { finishB({ providers: [{ slug: 'b', name: 'B', models: ['b-model'], eligible_models: ['b-model'] }] }) })
    await waitFor(() => expect(result.current.modelEntries.some(entry => entry.label === 'b-model')).toBe(true))
    expect(result.current.modelEntries.some(entry => entry.label === 'a-model')).toBe(false)
    expect(getOptions).toHaveBeenLastCalledWith(undefined, 'profile-b')
  })
})
