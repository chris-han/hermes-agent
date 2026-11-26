import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { SemantierWorkspaceSettings } from './semantier-workspace-settings'

vi.mock('@/i18n', () => ({ useI18n: () => ({ locale: 'en' }) }))
vi.mock('@/store/connections', () => ({ refreshConnectionsRegistry: vi.fn() }))
afterEach(() => vi.restoreAllMocks())

it('signs in and opens the existing server workspace without creating another connection', async () => {
  const identity = vi.fn().mockResolvedValue({ authenticated: false })

  const login = vi.fn().mockResolvedValue({
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

it('signs into Project continuity while leaving the local connection alone', async () => {
  const save = vi.fn()
  const onConnect = vi.fn()
  const projects = vi.fn().mockResolvedValue([])
  const projectMaterials = vi.fn().mockResolvedValue({ notes: [], graphs: [] })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      workspace: {
        identity: vi.fn().mockResolvedValue({ authenticated: false }),
        login: vi.fn().mockResolvedValue({
          authenticated: true,
          user: { user_id: 'a' },
          workspace: { id: 'wa' },
          organization_id: 'org'
        }),
        projects,
        projectMaterials
      },
      connections: { save }
    }
  })
  render(<SemantierWorkspaceSettings connections={[]} onConnect={onConnect} projectsOnly />)
  fireEvent.click(await screen.findByRole('button', { name: 'Sign in to Workspace' }))
  expect(await screen.findByText('No admitted project context is available.')).toBeTruthy()
  expect(projects).toHaveBeenCalledWith('https://cloud.semantier.cn', {
    userId: 'a',
    workspaceId: 'wa',
    organizationId: 'org'
  })
  expect(save).not.toHaveBeenCalled()
  expect(onConnect).not.toHaveBeenCalled()
})

it('keeps a signed project selectable when its materials request times out', async () => {
  const projectRef = 'project:7b5a28f1740c41f3bf36d091f8f2880e'
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      workspace: {
        identity: vi.fn().mockResolvedValue({
          authenticated: true,
          user: { user_id: 'a' },
          workspace: { id: 'wa' },
          organization_id: 'org'
        }),
        projects: vi.fn().mockResolvedValue([{ project_ref: projectRef }]),
        projectMaterials: vi.fn().mockRejectedValue(new Error('TimeoutError'))
      }
    }
  })
  render(<SemantierWorkspaceSettings connections={[]} projectsOnly />)
  expect(await screen.findByRole('option', { name: projectRef })).toBeTruthy()
  expect((await screen.findByRole('alert')).textContent).toContain('Project materials unavailable')
})
