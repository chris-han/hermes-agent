import { expect, it, vi } from 'vitest'

import { fetchBillingWithRetry } from './use-billing-state'

const unavailable = {
  ok: false as const,
  refusal: { backend: 'semantier', kind: 'billing_unavailable', message: 'Retry' }
}

it('bounds return retries and cancels pending recovery when a profile leaves', async () => {
  vi.useFakeTimers()

  try {
    const fetch = vi.fn().mockResolvedValue(unavailable)
    const controller = new AbortController()
    const pending = fetchBillingWithRetry(fetch, controller.signal)
    await vi.runAllTimersAsync()
    await pending
    expect(fetch).toHaveBeenCalledTimes(3)
    fetch.mockClear()
    const canceled = fetchBillingWithRetry(fetch, controller.signal)
    await Promise.resolve()
    controller.abort()
    await canceled
    expect(fetch).toHaveBeenCalledTimes(1)
  } finally {
    vi.useRealTimers()
  }
})
