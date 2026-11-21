import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { expect, it, vi } from 'vitest'

import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $connection, $currentModelSource } from '@/store/session'

import { useChatBarState } from './use-chat-bar-state'

it('shows a bound default for a fresh local draft without replacing live or manual choices', async () => {
  const original = Object.getOwnPropertyDescriptor(window, 'hermesDesktop')
  const account = { origin: 'https://cloud.example', scope: { userId: 'a', workspaceId: 'wa', organizationId: 'oa' } }
  const registry = { connections: [], accountContext: account }
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {
    connections: { list: vi.fn().mockResolvedValue(registry) }, workspace: {
      modelDefault: vi.fn().mockResolvedValue({ model: 'eligible-model', account }),
      onAccountChanged: () => () => {}, onModelStatus: () => () => {}
    }
  } })
  $connectionsRegistry.set(registry as never)
  $connection.set({ mode: 'local', connectionId: 'local', baseUrl: 'http://127.0.0.1' } as never)
  $currentModelSource.set('default')
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  const base = { activeGatewayProfile: 'default', activeSessionId: null, contextSuggestions: [], currentModel: 'auto-local',
    currentProvider: 'local', gateway: null, gatewayOpen: false }
  const hook = renderHook(({ sessionId }: { sessionId: string | null }) => useChatBarState({ ...base, activeSessionId: sessionId }),
    { initialProps: { sessionId: null as string | null }, wrapper })
  try {
    await waitFor(() => expect(hook.result.current.model).toMatchObject({ model: 'eligible-model', provider: 'semantier-cloud' }))
    act(() => $currentModelSource.set('manual'))
    expect(hook.result.current.model.model).toBe('auto-local')
    act(() => $currentModelSource.set('default'))
    hook.rerender({ sessionId: 'live' })
    expect(hook.result.current.model.model).toBe('auto-local')
  } finally {
    hook.unmount(); client.clear(); $connection.set(null); $connectionsRegistry.set(null); $currentModelSource.set('')
    if (original) {Object.defineProperty(window, 'hermesDesktop', original)} else {delete (window as { hermesDesktop?: unknown }).hermesDesktop}
  }
})
