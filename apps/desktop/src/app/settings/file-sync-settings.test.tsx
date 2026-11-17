import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { FileSyncSettings } from './file-sync-settings'
const original = window.hermesDesktop
const connection = $connection.get()
afterEach(() => {
  cleanup()
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
