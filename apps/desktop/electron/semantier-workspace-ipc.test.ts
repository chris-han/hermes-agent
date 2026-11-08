import { expect, it, vi } from 'vitest'

vi.mock('electron', () => ({ BrowserWindow: vi.fn(), ipcMain: { handle: vi.fn() }, session: { fromPartition: vi.fn() } }))

import { assertWorkspaceScope, registerSemantierWorkspaceIpc } from './semantier-workspace-ipc'
import { ipcMain, session } from 'electron'

const scope = { userId: 'user-a', workspaceId: 'workspace-a', organizationId: 'org-a' }
const identity = { authenticated: true, user: { user_id: 'user-a', workspace_id: 'workspace-a' }, workspace: { id: 'workspace-a' }, organization_id: 'org-a' }

it('accepts only the current authenticated server scope', () => {
  expect(() => assertWorkspaceScope(identity, scope)).not.toThrow()
  for (const expected of [{ ...scope, userId: 'user-b' }, { ...scope, workspaceId: 'workspace-b' }, { ...scope, organizationId: 'org-b' }]) {
    expect(() => assertWorkspaceScope(identity, expected)).toThrow('changed')
  }
  expect(() => assertWorkspaceScope({ ...identity, authenticated: false }, scope)).toThrow('changed')
})


it('signals account invalidation only after clearing the origin-owned cookie jar', async () => {
  const order: string[] = []
  const owned = { fetch: vi.fn(), clearStorageData: vi.fn(async () => { order.push('cookies') }),
    clearCache: vi.fn(async () => { order.push('cache') }) }
  vi.mocked(session.fromPartition).mockReturnValue(owned as never)
  vi.mocked(ipcMain.handle).mockClear()
  const changed = vi.fn(() => { order.push('changed') })
  registerSemantierWorkspaceIpc(changed)
  const handler = vi.mocked(ipcMain.handle).mock.calls.find(([name]) => name === 'hermes:workspace:logout')![1]
  await handler({} as never, 'https://cloud.example')
  expect(changed).toHaveBeenCalledWith('https://cloud.example', false)
  expect(order).toEqual(['cookies', 'cache', 'changed'])
})

it('forwards incremental reads through the same before-and-after account guard', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(Response.json(identity))
    .mockResolvedValueOnce(Response.json({ sessionId: 'original', messages: [], nextCursor: '218' }))
    .mockResolvedValueOnce(Response.json({ ...identity, user: { ...identity.user, user_id: 'user-b' } }))
  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler = vi.mocked(ipcMain.handle).mock.calls.find(([name]) => name === 'hermes:workspace:history')![1]
  await expect(handler({} as never, 'https://cloud.example', scope, 'original', '218')).rejects.toThrow('changed')
  expect(fetch.mock.calls[1][0]).toBe('https://cloud.example/api/history?sessionKey=original&afterId=218')
  expect(fetch).toHaveBeenCalledTimes(3)
})
