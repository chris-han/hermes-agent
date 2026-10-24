import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $desktopOnboarding } from '@/store/onboarding'
import { makeOAuthProvider } from '@/test/oauth-provider'

import { Picker } from '.'

afterEach(cleanup)

it('features the real Cloud provider and sends its scoped start rather than opening a portal link', async () => {
  const api = vi.fn(() => new Promise(() => undefined))
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
  $desktopOnboarding.set({
    configured: false,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: [makeOAuthProvider('nous', 'Nous Portal'), makeOAuthProvider('semantier-cloud', 'Semantier Cloud')],
    reason: null,
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false,
    freeTierReady: false
  })
  render(<Picker ctx={{ requestGateway: async () => undefined as never }} />)
  fireEvent.click(screen.getByRole('button', { name: /Semantier Cloud/ }))
  await waitFor(() =>
    expect(api).toHaveBeenCalledWith(
      expect.objectContaining({
        path: '/api/providers/oauth/semantier-cloud/start',
        method: 'POST'
      })
    )
  )
})
