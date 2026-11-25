import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { PersistentFollowupPanel } from './persistent-followup-panel'

vi.mock('@/i18n', () => ({ useI18n: () => ({ locale: 'en' }) }))
vi.mock('@/store/connections', async () => {
  const { atom } = await import('nanostores')

  return {
    $activeConnectionId: atom('cloud'),
    $connectionsRegistry: atom({
      connections: [{ id: 'cloud', transport: 'semantier-workspace-http', url: 'https://cloud.example' }]
    })
  }
})

const origin = 'https://cloud.example'
const scope = { userId: 'owner', workspaceId: 'workspace', organizationId: 'org' }

function workspaceApi() {
  const identity = vi.fn().mockResolvedValue({
    authenticated: true,
    user: { user_id: scope.userId },
    workspace: { id: scope.workspaceId },
    organization_id: scope.organizationId
  })

  const winmanCapability = vi.fn()
  const persistentFollowups = vi.fn()
  vi.stubGlobal('hermesDesktop', { workspace: { identity, winmanCapability, persistentFollowups } })

  return { winmanCapability, persistentFollowups }
}

afterEach(() => vi.unstubAllGlobals())

it('allows signed-in opt-in before protected work can be read', async () => {
  const api = workspaceApi()
  api.winmanCapability
    .mockResolvedValueOnce({ desired: { enabled: false, revision: 1 }, execution: 'disabled' })
    .mockResolvedValueOnce({ desired: { enabled: true, revision: 2 }, execution: 'ready' })
    .mockResolvedValueOnce({ desired: { enabled: false, revision: 3 }, execution: 'disabled' })
  api.persistentFollowups.mockImplementation(async () => {
    if (api.winmanCapability.mock.calls.length === 1) {
      throw new Error('403 disabled')
    }

    return { items: [{ id: 'one', title: 'Owned reminder', scheduled_at: '2026-11-24T12:00:00Z' }] }
  })

  render(<PersistentFollowupPanel />)
  const toggle = await screen.findByRole('switch', { name: 'Persistent follow-up' })
  await waitFor(() => expect(toggle.hasAttribute('disabled')).toBe(false))
  expect(toggle.getAttribute('aria-checked')).toBe('false')
  expect(api.persistentFollowups).not.toHaveBeenCalled()
  fireEvent.click(toggle)
  expect(await screen.findByText('Owned reminder')).toBeTruthy()
  expect(api.winmanCapability).toHaveBeenCalledWith(origin, scope, true)
  expect(api.persistentFollowups).toHaveBeenCalledExactlyOnceWith(origin, scope)
  fireEvent.click(toggle)
  await waitFor(() => expect(toggle.getAttribute('aria-checked')).toBe('false'))
  expect(screen.queryByText('Owned reminder')).toBeNull()
  expect(api.persistentFollowups).toHaveBeenCalledTimes(1)
})

it('keeps capability visible when the separate work read is denied', async () => {
  const api = workspaceApi()
  api.winmanCapability.mockResolvedValue({ desired: { enabled: true, revision: 1 }, execution: 'ready' })
  api.persistentFollowups.mockRejectedValue(new Error('403 work denied'))

  render(<PersistentFollowupPanel />)
  expect(await screen.findByRole('button', { name: 'Retry' })).toBeTruthy()
  const toggle = screen.getByRole('switch', { name: 'Persistent follow-up' })
  expect(toggle.getAttribute('aria-checked')).toBe('true')
  expect(toggle.hasAttribute('disabled')).toBe(false)
})

it('shows explicit scheduled Winman arming and readback commands in the ready panel', async () => {
  const api = workspaceApi()
  api.winmanCapability.mockResolvedValue({ desired: { enabled: true, revision: 1 }, execution: 'ready' })
  api.persistentFollowups.mockResolvedValue({ items: [] })

  render(<PersistentFollowupPanel />)
  expect(await screen.findByText('On · Ready')).toBeTruthy()
  fireEvent.click(screen.getByText('How to schedule a follow-up'))
  expect(screen.getByText('/followup arm <SOURCE_SHA256> <ISO_8601_WITH_TZ> CNY0.01 <EVENT>')).toBeTruthy()
  expect(screen.getByText('/followup read <WORK_ID>')).toBeTruthy()
  expect(screen.getByText('/followup create selected <ISO_8601_WITH_TZ>')).toBeTruthy()
})
