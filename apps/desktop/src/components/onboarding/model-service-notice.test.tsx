// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, test, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

const { signIn, refresh } = vi.hoisted(() => ({ signIn: vi.fn(), refresh: vi.fn() }))
vi.mock('@/store/boot', () => ({ $desktopBoot: atom({ running: false, error: null }) }))
vi.mock('@/store/onboarding', () => ({
  $desktopOnboarding: atom({ readiness: null, readinessScope: null }),
  startManualProviderOAuth: signIn,
  refreshOnboarding: refresh
}))
vi.mock('@/store/onboarding-scope', () => ({
  captureOnboardingScope: (profile: string) => ({ connectionId: null, profile })
}))
vi.mock('@/store/settings-scope', () => ({ setSettingsScope: vi.fn() }))
const { $desktopOnboarding } = await import('@/store/onboarding')
const { $desktopBoot } = await import('@/store/boot')
const { ModelServiceNotice } = await import('./model-service-notice')

function show(state: 'plan' | 'unavailable' | 'reauth', profile = 'work') {
  $desktopOnboarding.set({
    ...$desktopOnboarding.get(),
    readiness: {
      ready: false,
      source: 'runtime',
      modelService: { state }
    } as never,
    readinessScope: { connectionId: null, profile }
  })

  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <ModelServiceNotice profile="work" requestGateway={vi.fn()} />
      </I18nProvider>
    </MemoryRouter>
  )
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $desktopBoot.set({ ...$desktopBoot.get(), running: false })
})

test('background readiness is quiet and a failed transport offers explicit retry, not sign-in', () => {
  show('unavailable')
  expect(screen.getAllByRole('status')).toHaveLength(1)
  expect(signIn).not.toHaveBeenCalled()
  expect(refresh).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  expect(refresh).toHaveBeenCalledTimes(1)
  expect(signIn).not.toHaveBeenCalled()
})

test('credential recovery starts only after user action and keeps the captured profile', () => {
  show('reauth')
  expect(signIn).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: /Sign in/ }))
  expect(signIn).toHaveBeenCalledWith('semantier-cloud', { connectionId: null, profile: 'work' })
})

test('late readiness for a different profile cannot present a recovery action', () => {
  show('plan', 'previous-work')
  expect(screen.queryByRole('status')).toBeNull()
  expect(signIn).not.toHaveBeenCalled()
})
