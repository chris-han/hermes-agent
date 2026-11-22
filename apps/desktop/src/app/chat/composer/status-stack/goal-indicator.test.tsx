import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $goalsBySession, type SessionGoal } from '@/store/goals'
import { $sharedMetricsConsent } from '@/store/shared-metrics'

import { ComposerStatusStack } from './index'

// The stack measures itself into a surface var — jsdom has no ResizeObserver.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', ResizeObserverStub)

const SID = 'sess-goal-1'

const goal = (status: SessionGoal['status'], title = 'ship the feature', detail?: string): SessionGoal => ({
  detail,
  status,
  title,
  updatedAt: Date.now()
})

function renderStack(sessionId: null | string = SID) {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="en">
        <ComposerStatusStack queue={null} sessionId={sessionId} />
      </I18nProvider>
    </MemoryRouter>
  )
}

describe('ComposerStatusStack goal indicator', () => {
  beforeEach(() => {
    $goalsBySession.set({})
  })

  afterEach(() => {
    cleanup()
    $goalsBySession.set({})
    $sharedMetricsConsent.set(null)
  })

  it.each([false, true])('hides the shared-metrics offer by default, including recipient re-ask=%s', reask => {
    $sharedMetricsConsent.set({ enabled: false, send: false, decided: false, reask })

    const view = renderStack()

    expect(screen.queryByText('Share usage stats?')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Share with Semantier' })).toBeNull()
    expect(view.container.firstChild).toBeNull()
    // Dismissing the banner must not mutate consent or silently opt in.
    expect($sharedMetricsConsent.get()).toMatchObject({ enabled: false, send: false, decided: false })
  })

  it('shows an active goal with its title', () => {
    $goalsBySession.set({ [SID]: goal('active') })

    renderStack()

    expect(screen.getByText('Goal active')).toBeTruthy()
    expect(screen.queryByText('ship the feature')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Goal (active|paused)/ }))
    expect(screen.getByText('ship the feature')).toBeTruthy()
  })

  it('labels a paused goal as paused', () => {
    $goalsBySession.set({ [SID]: goal('paused') })

    renderStack()

    expect(screen.getByText('Goal paused')).toBeTruthy()
    expect(screen.queryByText('ship the feature')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: /Goal (active|paused)/ }))
    expect(screen.getByText('ship the feature')).toBeTruthy()
  })

  it('scopes the indicator to the goal-owning session', () => {
    $goalsBySession.set({ 'other-session': goal('active') })

    const view = renderStack()

    expect(view.container.firstChild).toBeNull()
  })
})
