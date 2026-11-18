import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { FileSyncSettings } from './file-sync-settings'
const original = window.hermesDesktop
const connection = $connection.get()
afterEach(() => {
  cleanup()
  vi.useRealTimers()
  window.hermesDesktop = original
  $connection.set(connection)
})

function connect(userId: string) {
  $connection.set({
    mode: 'remote',
    baseUrl: 'https://cloud.example',
    connectionId: 'cloud',
    transport: 'semantier-workspace-http',
    workspaceScope: { userId, workspaceId: 'w-' + userId, organizationId: 'org' }
  } as never)
}

it('shows a recoverable state without a connected personal Workspace', () => {
  $connection.set(null as never)
  render(<FileSyncSettings />)
  expect(screen.getByText('Connect this window to your Semantier Workspace first.')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Choose and enable' })).toBeNull()
})
it('ignores a completed folder operation from the previous account', async () => {
  let finish: (value: any) => void = () => {}

  const choose = vi.fn(
    () =>
      new Promise<any>(resolve => {
        finish = resolve
      })
  )

  window.hermesDesktop = {
    ...original,
    fileSync: {
      status: async () => ({ state: 'Paused' }),
      choose,
      pause: async () => ({ state: 'Paused' }),
      resume: async () => ({ state: 'Paused' })
    }
  }
  connect('a')
  render(<FileSyncSettings />)
  fireEvent.click(screen.getByRole('button', { name: 'Choose and enable' }))
  expect(choose).toHaveBeenCalledTimes(1)
  act(() => connect('b'))
  await act(async () => finish({ state: 'Synced', folder: 'A-private-folder' }))
  await waitFor(() => expect(screen.getByText('No folder selected')).toBeTruthy())
  expect(screen.queryByText('A-private-folder')).toBeNull()
})

it('clears a failed status poll after recovery and retains current engine errors', async () => {
  vi.useFakeTimers()

  const status = vi.fn()
    .mockRejectedValueOnce(new Error('identity timeout'))
    .mockResolvedValueOnce({ state: 'Synced', folder: 'pilot' })
    .mockResolvedValue({ state: 'Error', folder: 'pilot', message: 'Engine unavailable' })

  window.hermesDesktop = {
    ...original,
    fileSync: { status, choose: vi.fn(), pause: vi.fn(), resume: vi.fn() }
  }
  connect('a')
  render(<FileSyncSettings />)
  await act(async () => {})
  expect(screen.getByRole('alert').textContent).toContain('Reconnect your Workspace')
  await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
  expect(screen.getByText('Synced')).toBeTruthy()
  expect(screen.queryByRole('alert')).toBeNull()
  await act(async () => { await vi.advanceTimersByTimeAsync(2000) })
  expect(screen.getByText('Error')).toBeTruthy()
  expect(screen.getByRole('alert').textContent).toBe('Engine unavailable')
})

it('does not let a previous account poll clear the current account error', async () => {
  vi.useFakeTimers()

  let finish: (value: any) => void = () => {}

  const status = vi.fn()
    .mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
    .mockRejectedValue(new Error('current owner offline'))

  window.hermesDesktop = {
    ...original,
    fileSync: { status, choose: vi.fn(), pause: vi.fn(), resume: vi.fn() }
  }
  connect('a')
  render(<FileSyncSettings />)
  await act(async () => { connect('b') })
  expect(screen.getByRole('alert').textContent).toContain('Reconnect your Workspace')
  await act(async () => { finish({ state: 'Synced', folder: 'A-private-folder' }) })
  expect(screen.getByRole('alert').textContent).toContain('Reconnect your Workspace')
  expect(screen.queryByText('A-private-folder')).toBeNull()
})
