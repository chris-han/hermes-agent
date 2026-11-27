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

  const status = vi
    .fn()
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
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2000)
  })
  expect(screen.getByText('Synced')).toBeTruthy()
  expect(screen.queryByRole('alert')).toBeNull()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2000)
  })
  expect(screen.getByText('Error')).toBeTruthy()
  expect(screen.getByRole('alert').textContent).toBe('Engine unavailable')
})

it('does not let a previous account poll clear the current account error', async () => {
  vi.useFakeTimers()

  let finish: (value: any) => void = () => {}

  const status = vi
    .fn()
    .mockImplementationOnce(
      () =>
        new Promise(resolve => {
          finish = resolve
        })
    )
    .mockRejectedValue(new Error('current owner offline'))

  window.hermesDesktop = {
    ...original,
    fileSync: { status, choose: vi.fn(), pause: vi.fn(), resume: vi.fn() }
  }
  connect('a')
  render(<FileSyncSettings />)
  await act(async () => {
    connect('b')
  })
  expect(screen.getByRole('alert').textContent).toContain('Reconnect your Workspace')
  await act(async () => {
    finish({ state: 'Synced', folder: 'A-private-folder' })
  })
  expect(screen.getByRole('alert').textContent).toContain('Reconnect your Workspace')
  expect(screen.queryByText('A-private-folder')).toBeNull()
})

it('keeps the current account poll when earlier success and failure settle late', async () => {
  vi.useFakeTimers()
  let oldSuccess!: (value: any) => void
  let oldFailure!: (reason: Error) => void

  const status = vi
    .fn()
    .mockImplementationOnce(
      () =>
        new Promise(resolve => {
          oldSuccess = resolve
        })
    )
    .mockImplementationOnce(
      () =>
        new Promise((_, reject) => {
          oldFailure = reject
        })
    )
    .mockResolvedValue({ state: 'Conflict', folder: 'current-folder' })

  window.hermesDesktop = { ...original, fileSync: { status, choose: vi.fn(), pause: vi.fn(), resume: vi.fn() } }
  connect('a')
  render(<FileSyncSettings />)
  await act(async () => {
    connect('b')
  })
  await act(async () => {
    connect('c')
  })
  expect(screen.getByText('Conflict')).toBeTruthy()
  await act(async () => {
    oldSuccess({ state: 'Paused', folder: 'old-folder' })
    oldFailure(new Error('obsolete'))
  })
  expect(screen.getByText('Conflict')).toBeTruthy()
  expect(screen.getByText('current-folder')).toBeTruthy()
  expect(screen.queryByText('old-folder')).toBeNull()
  expect(screen.queryByRole('alert')).toBeNull()
})

it('does not let a pre-resume poll overwrite the resumed folder', async () => {
  vi.useFakeTimers()
  let finish!: (value: any) => void

  const status = vi
    .fn()
    .mockResolvedValueOnce({ state: 'Paused', folder: 'current-folder' })
    .mockImplementation(
      () =>
        new Promise(resolve => {
          finish = resolve
        })
    )

  window.hermesDesktop = {
    ...original,
    fileSync: {
      status,
      choose: vi.fn(),
      pause: vi.fn(),
      resume: async () => ({ state: 'Synced', folder: 'current-folder' })
    }
  }
  connect('a')
  render(<FileSyncSettings />)
  await act(async () => {})
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2000)
  })
  fireEvent.click(screen.getByRole('button', { name: 'Resume' }))
  await act(async () => {})
  await act(async () => {
    finish({ state: 'Paused', folder: 'old-folder' })
  })
  expect(screen.getByText('Synced')).toBeTruthy()
  expect(screen.queryByText('old-folder')).toBeNull()
})

it('rejects late A results after A to B to A and displays current Error and Paused', async () => {
  let finish!: (value: any) => void

  const status = vi
    .fn()
    .mockImplementationOnce(
      () =>
        new Promise(resolve => {
          finish = resolve
        })
    )
    .mockResolvedValueOnce({ state: 'Paused', folder: 'B-folder' })
    .mockResolvedValue({ state: 'Error', folder: 'A-current', message: 'Actual engine error' })

  window.hermesDesktop = { ...original, fileSync: { status, choose: vi.fn(), pause: vi.fn(), resume: vi.fn() } }
  connect('a')
  render(<FileSyncSettings />)
  await act(async () => {
    connect('b')
  })
  expect(screen.getByText('Paused')).toBeTruthy()
  await act(async () => {
    connect('a')
  })
  await act(async () => {
    finish({ state: 'Synced', folder: 'A-old' })
  })
  expect(screen.getByText('Error')).toBeTruthy()
  expect(screen.getByRole('alert').textContent).toBe('Actual engine error')
  expect(screen.getByText('A-current')).toBeTruthy()
  expect(screen.queryByText('A-old')).toBeNull()
  expect(screen.queryByText('B-folder')).toBeNull()
})

it('publishes a slow settled poll so the selected folder can be resumed', async () => {
  vi.useFakeTimers()
  let finish!: (value: any) => void

  const status = vi
    .fn()
    .mockImplementationOnce(
      () =>
        new Promise(resolve => {
          finish = resolve
        })
    )
    .mockImplementation(() => new Promise(() => {}))

  window.hermesDesktop = { ...original, fileSync: { status, choose: vi.fn(), pause: vi.fn(), resume: vi.fn() } }
  connect('a')
  render(<FileSyncSettings />)
  await act(async () => {
    await vi.advanceTimersByTimeAsync(8000)
  })
  await act(async () => {
    finish({ state: 'Paused', folder: 'current-folder' })
  })
  expect(screen.getByText('current-folder')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Resume' }).hasAttribute('disabled')).toBe(false)
  expect(status).toHaveBeenCalledTimes(1)

  status.mockResolvedValue({ state: 'Synced', folder: 'current-folder' })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2000)
  })
  expect(screen.getByText('Synced')).toBeTruthy()
})

it('surfaces a slow failed poll and recovers after it settles', async () => {
  vi.useFakeTimers()
  let fail!: (reason: Error) => void

  const status = vi
    .fn()
    .mockImplementationOnce(
      () =>
        new Promise((_, reject) => {
          fail = reject
        })
    )
    .mockImplementation(() => new Promise(() => {}))

  window.hermesDesktop = { ...original, fileSync: { status, choose: vi.fn(), pause: vi.fn(), resume: vi.fn() } }
  connect('a')
  render(<FileSyncSettings />)
  await act(async () => {
    await vi.advanceTimersByTimeAsync(8000)
  })
  await act(async () => {
    fail(new Error('slow identity timeout'))
  })
  expect(screen.getByRole('alert').textContent).toContain('Reconnect your Workspace')
  expect(status).toHaveBeenCalledTimes(1)

  status.mockResolvedValue({ state: 'Synced', folder: 'current-folder' })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2000)
  })
  expect(screen.getByText('Synced')).toBeTruthy()
  expect(screen.queryByRole('alert')).toBeNull()
})
