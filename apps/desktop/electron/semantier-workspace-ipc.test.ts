import { expect, it, vi } from 'vitest'

vi.mock('electron', () => ({ BrowserWindow: vi.fn(), ipcMain: { handle: vi.fn() }, session: { fromPartition: vi.fn() } }))

import { ipcMain, session } from 'electron'

import { assertWorkspaceScope, registerSemantierWorkspaceIpc } from './semantier-workspace-ipc'

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

it('checks the authenticated scope before canonical Chat creation', async () => {
  const fetch=vi.fn().mockResolvedValue(Response.json({...identity,user:{...identity.user,user_id:'user-b'}}))
  vi.mocked(session.fromPartition).mockReturnValue({fetch} as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler=vi.mocked(ipcMain.handle).mock.calls.find(([name])=>name==='hermes:workspace:ensure-winman-chat')![1]
  await expect(handler({} as never,'https://cloud.example',scope)).rejects.toThrow('changed')
  expect(fetch).toHaveBeenCalledTimes(1)
  expect(fetch.mock.calls[0][0]).toBe('https://cloud.example/auth/context')
})

it.each(['native-context','bind-native-context','send-native'])('rejects a changed account before %s can access its captured Native owner', async name => {
  const fetch=vi.fn().mockResolvedValue(Response.json({...identity,user:{...identity.user,user_id:'user-b'}}))
  vi.mocked(session.fromPartition).mockReturnValue({fetch} as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler=vi.mocked(ipcMain.handle).mock.calls.find(([channel])=>channel==='hermes:workspace:'+name)![1]
  await expect(handler({} as never,'https://cloud.example',scope,'canonical-a','main-a','request','binding')).rejects.toThrow('changed')
  expect(fetch).toHaveBeenCalledTimes(1)
})

it('creates only in the captured account and returns the real concurrent history snapshot', async () => {
  const fetcher=vi.fn(async (url:string,init?:RequestInit)=>{
    if(url.endsWith('/auth/context')) {return Response.json(identity)}

    if(url.endsWith('/api/sessions')&&init?.method==='POST') {return Response.json({ok:true,sessionKey:'session_desktop_create-001',entry:{key:'session_desktop_create-001'}})}

    return Response.json({sessionId:'session_desktop_create-001',messages:[{messageId:'91',role:'assistant',text:'Server-owned'}]})
  })

  vi.mocked(session.fromPartition).mockReturnValue({fetch:fetcher} as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler=vi.mocked(ipcMain.handle).mock.calls.find(([name])=>name==='hermes:workspace:create-session')![1]
  const result=await handler({} as never,'https://cloud.example',scope,'Main','create-001')
  expect(result).toMatchObject({sessionId:'session_desktop_create-001',messages:[{messageId:'91'}]})
  expect(fetcher.mock.calls.filter(([,init])=>init?.method==='POST')).toHaveLength(1)
  fetcher.mockClear()
  await expect(handler({} as never,'https://cloud.example',{...scope,userId:'foreign'},'Main','other')).rejects.toThrow('changed')
  expect(fetcher.mock.calls.some(([,init])=>init?.method==='POST')).toBe(false)
})
