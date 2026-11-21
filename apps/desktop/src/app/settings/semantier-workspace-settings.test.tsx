import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { SemantierWorkspaceSettings } from './semantier-workspace-settings'

vi.mock('@/i18n', () => ({ useI18n: () => ({ locale: 'en' }) }))
vi.mock('@/store/connections', () => ({ refreshConnectionsRegistry: vi.fn() }))
afterEach(() => vi.restoreAllMocks())

it('signs in and opens the existing server workspace without creating another connection', async () => {
  const identity = vi.fn().mockResolvedValue({ authenticated: false })

  const login = vi
    .fn()
    .mockResolvedValue({
      authenticated: true,
      user: { user_id: 'a', name: 'Alice' },
      workspace: { id: 'wa' },
      organization_id: 'org'
    })

  const save = vi.fn()
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { workspace: { identity, login }, connections: { save } }
  })
  const onConnect = vi.fn()
  render(
    <SemantierWorkspaceSettings
      connections={[
        {
          id: 'own-cloud',
          kind: 'cloud',
          transport: 'semantier-workspace-http',
          label: 'My workspace',
          url: 'https://cloud.semantier.cn',
          tokenSet: false,
          tokenPreview: null
        }
      ]}
      onConnect={onConnect}
    />
  )
  fireEvent.click(await screen.findByRole('button', { name: 'Sign in and open workspace' }))
  await waitFor(() => expect(onConnect).toHaveBeenCalledWith('own-cloud'))
  expect(login).toHaveBeenCalledWith('https://cloud.semantier.cn')
  expect(save).not.toHaveBeenCalled()
})

it('creates an explicit HTTP connection only after authenticated sign-in', async () => {
  const identity = vi.fn().mockResolvedValue({ authenticated: false })

  const login = vi
    .fn()
    .mockResolvedValue({ authenticated: true, user: { user_id: 'a' }, workspace: { id: 'wa' }, organization_id: 'org' })

  const save = vi.fn().mockResolvedValue({ connection: { id: 'new-cloud' } })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { workspace: { identity, login }, connections: { save } }
  })
  const onConnect = vi.fn()
  render(<SemantierWorkspaceSettings connections={[]} onConnect={onConnect} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Sign in and open workspace' }))
  await waitFor(() => expect(onConnect).toHaveBeenCalledWith('new-cloud'))
  expect(save).toHaveBeenCalledWith(
    expect.objectContaining({ kind: 'cloud', transport: 'semantier-workspace-http', url: 'https://cloud.semantier.cn' })
  )
})

it('does not save or open a workspace when hosted sign-in is cancelled', async () => {
  const save = vi.fn(),
    onConnect = vi.fn()

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      workspace: {
        identity: vi.fn().mockResolvedValue({ authenticated: false }),
        login: vi.fn().mockRejectedValue(new Error('closed'))
      },
      connections: { save }
    }
  })
  render(<SemantierWorkspaceSettings connections={[]} onConnect={onConnect} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Sign in and open workspace' }))
  expect((await screen.findByRole('alert')).textContent).toContain('Could not open the workspace')
  expect(save).not.toHaveBeenCalled()
  expect(onConnect).not.toHaveBeenCalled()
  expect((screen.getByRole('textbox', { name: 'Workspace address' }) as HTMLInputElement).value).toBe(
    'https://cloud.semantier.cn'
  )
})

it('keeps identity outage distinct from sign-out and retries without opening login', async () => {
  const identity = vi.fn().mockRejectedValue(new Error('TimeoutError'))
  const login = vi.fn()
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true, value: { workspace: { identity, login }, connections: { save: vi.fn() } }
  })
  render(<SemantierWorkspaceSettings connections={[]} onConnect={vi.fn()} />)
  expect(await screen.findByText('Workspace identity is unavailable. Try again.')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(identity).toHaveBeenCalledTimes(2))
  expect(login).not.toHaveBeenCalled()
  expect((screen.getByRole('textbox', { name: 'Workspace address' }) as HTMLInputElement).value).toBe('https://cloud.semantier.cn')
})

it('shows forbidden workspace access without proposing sign-in', async () => {
  const identity = vi.fn().mockRejectedValue(new Error('Semantier workspace request failed (403).'))
  const login = vi.fn()
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true, value: { workspace: { identity, login }, connections: { save: vi.fn() } }
  })
  render(<SemantierWorkspaceSettings connections={[]} onConnect={vi.fn()} />)
  expect(await screen.findByText('Workspace access is denied for this account or organization.')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Sign in and open workspace' })).toBeNull()
  expect(login).not.toHaveBeenCalled()
})

it('does not start login for a stale origin after the address changes', async () => {
  let resolveOld!: (value: { authenticated: boolean }) => void

  const identity = vi.fn()
    .mockResolvedValueOnce({ authenticated: false })
    .mockImplementationOnce(() => new Promise(resolve => { resolveOld = resolve }))
    .mockResolvedValue({ authenticated: false })

  const login = vi.fn()
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true, value: { workspace: { identity, login }, connections: { save: vi.fn() } }
  })
  render(<SemantierWorkspaceSettings connections={[]} onConnect={vi.fn()} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Sign in and open workspace' }))
  fireEvent.change(screen.getByRole('textbox', { name: 'Workspace address' }), { target: { value: 'https://other.example' } })
  resolveOld({ authenticated: false })
  await waitFor(() => expect(identity).toHaveBeenCalledWith('https://other.example'))
  expect(login).not.toHaveBeenCalled()
})
