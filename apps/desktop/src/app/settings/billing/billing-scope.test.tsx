import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { $activeGatewayRoute } from '@/store/gateway'
import { $connection } from '@/store/session'

import { BillingApiProvider } from './api'
import { billingDevFixtures } from './dev-fixtures'
import { createSimulatedBillingApi } from './simulated-api'
import { useBillingState } from './use-billing-state'

const initialConnection = $connection.get()
const initialProfile = $activeGatewayRoute.get()

afterEach(() => {
  cleanup()
  $connection.set(initialConnection)
  $activeGatewayRoute.set(initialProfile)
})

it.each(['connection', 'profile'])('discards an old in-flight account when the %s changes on the same socket', async lane => {
  let finishOld!: (value: unknown) => void
  const old = new Promise(resolve => { finishOld = resolve })
  const fixture = Object.values(billingDevFixtures)[0]
  const api = createSimulatedBillingApi(fixture)
  api.fetchBillingState = vi.fn().mockReturnValueOnce(old).mockResolvedValueOnce({ ok: true, data: { org_name: 'new-org' } })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>
      <BillingApiProvider value={api}>{children}</BillingApiProvider>
    </QueryClientProvider>
  )

  act(() => {
    $connection.set({ connectionId: 'old' } as HermesConnection)
    $activeGatewayRoute.set('profile-a')
  })
  const { result } = renderHook(() => useBillingState(), { wrapper })
  await waitFor(() => expect(api.fetchBillingState).toHaveBeenCalledTimes(1))
  act(() => {
    if (lane === 'profile') {$activeGatewayRoute.set('profile-b')}
    else {$connection.set({ connectionId: 'new' } as HermesConnection)}
  })
  await waitFor(() => expect(result.current.data).toEqual({ ok: true, data: { org_name: 'new-org' } }))
  await act(async () => finishOld({ ok: true, data: { org_name: 'old-org' } }))
  expect(result.current.data).toEqual({ ok: true, data: { org_name: 'new-org' } })
  client.clear()
})

it('refreshes on native window return without replacing the current connection', async () => {
  const fixture = Object.values(billingDevFixtures)[0]
  const api = createSimulatedBillingApi(fixture)
  const fetch = vi.spyOn(api, 'fetchBillingState')
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}><BillingApiProvider value={api}>{children}</BillingApiProvider></QueryClientProvider>
  )

  const connection = $connection.get()
  const { result } = renderHook(() => useBillingState(), { wrapper })
  await waitFor(() => expect(result.current.isSuccess).toBe(true))
  const before = fetch.mock.calls.length
  act(() => window.dispatchEvent(new Event('focus')))
  await waitFor(() => expect(fetch.mock.calls.length).toBe(before + 1))
  expect($connection.get()).toBe(connection)
  client.clear()
})
