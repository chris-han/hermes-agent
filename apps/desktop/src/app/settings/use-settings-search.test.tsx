import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

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

afterEach(() => { cleanup(); vi.clearAllMocks() })

describe('settings search catalog', () => {
  it('does not query or expose individual providers and models', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
    const { result } = renderHook(() => useSettingsSearchCatalog(true), { wrapper })
    expect(getOptions).not.toHaveBeenCalled()
    expect(result.current).not.toHaveProperty('modelEntries')
  })
})
