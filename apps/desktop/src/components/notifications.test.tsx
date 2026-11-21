import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'

import { clearNotifications, notify } from '@/store/notifications'
import { stubResizeObserver } from '@/test/jsdom'

import { NotificationStack } from './notifications'

beforeAll(stubResizeObserver)

describe('toast titles', () => {
  beforeEach(() => {
    clearNotifications()
  })

  afterEach(() => {
    cleanup()
    clearNotifications()
  })

  it('keeps long resolver diagnostics inside a scrollable toast detail region', () => {
    const detail = 'Resolver: D:/repo/deliverables/work/very-long-build-root/'.repeat(25)
    notify({
      id: 'resolver-error',
      kind: 'error',
      title: 'Cannot enable disk-cleanup',
      message: 'Check package index',
      detail,
      durationMs: 0
    })
    render(<NotificationStack />)
    fireEvent.click(screen.getByText('Details'))
    const log = screen.getByText(detail)
    expect(log.tagName).toBe('PRE')
    expect(log.className).toContain('overflow-y-auto')
    expect(log.className).toContain('overflow-x-auto')
    expect(log.className).toContain('[overflow-wrap:anywhere]')
  })

  it.each(['default', 'bottom-right'] as const)(
    'caps the %s toast stack at one back edge and keeps older notifications reachable',
    async placement => {
      for (let index = 0; index < 7; index++) {
        notify({ id: `notice-${index}`, message: `Notice ${index}`, placement, durationMs: 0 })
      }

      const view = render(<NotificationStack />)

      expect(screen.getAllByRole('status')).toHaveLength(1)
      expect(view.baseElement.querySelectorAll('[data-slot="card-stack-edge"]')).toHaveLength(1)
      fireEvent.click(screen.getByRole('button', { name: /Show.*6/ }))
      expect(screen.getByText('Notice 0')).toBeTruthy()
      expect(screen.getAllByRole('status')).toHaveLength(7)
      fireEvent.click(screen.getAllByRole('button', { name: /Dismiss/ })[0])
      await waitFor(() => expect(screen.queryByText('Notice 6')).toBeNull())
    }
  )
})
