import { expect, it, vi } from 'vitest'

vi.mock('electron', () => ({ BrowserWindow: vi.fn(), ipcMain: { handle: vi.fn() }, session: { fromPartition: vi.fn() } }))

import { assertWorkspaceScope } from './semantier-workspace-ipc'

const scope = { userId: 'user-a', workspaceId: 'workspace-a', organizationId: 'org-a' }
const identity = { authenticated: true, user: { user_id: 'user-a', workspace_id: 'workspace-a' }, workspace: { id: 'workspace-a' }, organization_id: 'org-a' }

it('accepts only the current authenticated server scope', () => {
  expect(() => assertWorkspaceScope(identity, scope)).not.toThrow()
  for (const expected of [{ ...scope, userId: 'user-b' }, { ...scope, workspaceId: 'workspace-b' }, { ...scope, organizationId: 'org-b' }]) {
    expect(() => assertWorkspaceScope(identity, expected)).toThrow('changed')
  }
  expect(() => assertWorkspaceScope({ ...identity, authenticated: false }, scope)).toThrow('changed')
})
