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

it('checks the authenticated scope before canonical Chat creation', async () => {
  const fetch=vi.fn().mockResolvedValue(Response.json({...identity,user:{...identity.user,user_id:'user-b'}}))
  vi.mocked(session.fromPartition).mockReturnValue({fetch} as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler=vi.mocked(ipcMain.handle).mock.calls.find(([name])=>name==='hermes:workspace:ensure-winman-chat')![1]
  await expect(handler({sender:{id:99}} as never,'https://cloud.example',scope)).rejects.toThrow('changed')
  expect(fetch).toHaveBeenCalledTimes(1)
  expect(fetch.mock.calls[0][0]).toBe('https://cloud.example/auth/context')
})

it.each(['native-context','bind-native-context','send-native'])('rejects a changed account before %s can access its captured Native owner', async name => {
  const fetch=vi.fn().mockResolvedValue(Response.json({...identity,user:{...identity.user,user_id:'user-b'}}))
  vi.mocked(session.fromPartition).mockReturnValue({fetch} as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler=vi.mocked(ipcMain.handle).mock.calls.find(([channel])=>channel==='hermes:workspace:'+name)![1]
  await expect(handler({sender:{id:99}} as never,'https://cloud.example',scope,'canonical-a','main-a','request','binding')).rejects.toThrow('changed')
  expect(fetch).toHaveBeenCalledTimes(1)
})

it('creates only in the captured account and returns the real concurrent history snapshot', async () => {
  const fetcher=vi.fn(async (url:string,init?:RequestInit)=>{
    if(url.endsWith('/auth/context')) return Response.json(identity)
    if(url.endsWith('/api/sessions')&&init?.method==='POST') return Response.json({ok:true,sessionKey:'session_desktop_create-001',entry:{key:'session_desktop_create-001'}})
    return Response.json({sessionId:'session_desktop_create-001',messages:[{messageId:'91',role:'assistant',text:'Server-owned'}]})
  })
  vi.mocked(session.fromPartition).mockReturnValue({fetch:fetcher} as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler=vi.mocked(ipcMain.handle).mock.calls.find(([name])=>name==='hermes:workspace:create-session')![1]
  const result=await handler({sender:{id:99}} as never,'https://cloud.example',scope,'Main','create-001')
  expect(result).toMatchObject({sessionId:'session_desktop_create-001',messages:[{messageId:'91'}]})
  expect(fetcher.mock.calls.filter(([,init])=>init?.method==='POST')).toHaveLength(1)
  fetcher.mockClear()
  await expect(handler({} as never,'https://cloud.example',{...scope,userId:'foreign'},'Main','other')).rejects.toThrow('changed')
  expect(fetcher.mock.calls.some(([,init])=>init?.method==='POST')).toBe(false)
})


it('cancels only the caller-owned in-flight Main stream without retrying or submitting another turn', async () => {
  let streamSignal: AbortSignal | undefined
  let streamController: ReadableStreamDefaultController<Uint8Array> | undefined
  const fetch = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith('/auth/context')) return Response.json(identity)
    if (url.endsWith('/api/send-stream')) {
      streamSignal = init?.signal as AbortSignal
      return new Response(new ReadableStream<Uint8Array>({ start(controller) {
        streamController = controller
        streamSignal.addEventListener('abort', () => controller.error(new DOMException('Stopped', 'AbortError')), { once: true })
      } }), { headers: { 'Content-Type': 'text/event-stream' } })
    }
    throw new Error('Unexpected request: ' + url)
  })
  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler = (name: string) => vi.mocked(ipcMain.handle).mock.calls.find(([channel]) => channel === name)?.[1]
  const stop = handler('hermes:workspace:interrupt')
  expect(stop).toBeTypeOf('function')
  const event = { sender: { id: 99 } } as never
  const pending = handler('hermes:workspace:send')!(event, 'https://cloud.example', scope, 'original', 'hello', 'retained-request')
  const rejected = expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  await vi.waitFor(() => expect(streamController).toBeDefined())
  await expect(stop!(event, 'https://cloud.example', { ...scope, userId: 'other' }, 'original', 'retained-request')).rejects.toThrow('changed')
  expect(streamSignal?.aborted).toBe(false)
  await expect(stop!({ sender: { id: 100 } } as never, 'https://cloud.example', scope, 'original', 'retained-request')).resolves.toMatchObject({ cancel_requested: false })
  expect(streamSignal?.aborted).toBe(false)
  await expect(stop!(event, 'https://cloud.example', scope, 'original', 'older-request')).resolves.toMatchObject({ cancel_requested: false })
  expect(streamSignal?.aborted).toBe(false)
  await expect(stop!(event, 'https://cloud.example', scope, 'original', 'retained-request')).resolves.toMatchObject({ cancel_requested: true, request_id: 'retained-request' })
  await rejected
  expect(streamSignal?.aborted).toBe(true)
  await expect(stop!(event, 'https://cloud.example', scope, 'original', 'retained-request')).resolves.toMatchObject({ cancel_requested: false })
  expect(fetch.mock.calls.filter(([url]) => url.endsWith('/api/send-stream'))).toHaveLength(1)
})


it('withdraws a Main send stopped while its initial identity check is still waiting', async () => {
  let release: (() => void) | undefined
  let reads = 0
  const fetch = vi.fn(async (url: string) => {
    if (!url.endsWith('/auth/context')) throw new Error('Stopped request must not be sent')
    if (++reads === 1) await new Promise<void>(resolve => { release = resolve })
    return Response.json(identity)
  })
  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler = (name: string) => vi.mocked(ipcMain.handle).mock.calls.find(([channel]) => channel === name)![1]
  const event = { sender: { id: 101 } } as never
  const pending = handler('hermes:workspace:send')(event, 'https://cloud.example', scope, 'original', 'hello', 'withdrawn-request')
  const rejected = expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  await vi.waitFor(() => expect(release).toBeTypeOf('function'))
  await expect(handler('hermes:workspace:interrupt')(event, 'https://cloud.example', scope, 'original', 'withdrawn-request')).resolves.toMatchObject({ cancel_requested: true })
  release!()
  await rejected
  expect(fetch.mock.calls.every(([url]) => url.endsWith('/auth/context'))).toBe(true)
})


it('stops Native work by awaiting its original admission and cancelling that exact receipt once', async () => {
  let admit: ((value: Response) => void) | undefined
  let cancelled = false
  const admission = 'a'.repeat(32)
  const fetch = vi.fn(async (url: string) => {
    if (url.endsWith('/auth/context')) return Response.json(identity)
    if (url.endsWith('/api/winman/messages/canonical-a')) return new Promise<Response>(resolve => { admit = resolve })
    const receipt = { admission_id: admission, status: cancelled ? 'terminal' : 'queued', outcome: cancelled ? 'cancelled' : null }
    if (url.endsWith('/cancel')) { cancelled = true; return Response.json({ cancel_requested: true, receipt: { ...receipt, status:'terminal', outcome:'cancelled' } }) }
    if (url.includes('/api/winman/history/')) return Response.json({ session_id:'canonical-a', bot_id:'bot-a',native_profile_id:'profile-a',canonical_session:{id:'canonical-a'},messages:[],admission_receipt:receipt })
    throw new Error('Unexpected request '+url)
  })
  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler = (name: string) => vi.mocked(ipcMain.handle).mock.calls.find(([channel]) => channel === name)![1]
  const event = { sender: { id: 102 } } as never
  const pending = handler('hermes:workspace:send-native')(event,'https://cloud.example',scope,'canonical-a','hello','native-request','binding')
  await vi.waitFor(() => expect(admit).toBeTypeOf('function'))
  const stopped = handler('hermes:workspace:interrupt')(event,'https://cloud.example',scope,'canonical-a','native-request')
  expect(cancelled).toBe(false)
  admit!(Response.json({ session_id:'canonical-a',bot_id:'bot-a',receipt:{admission_id:admission,status:'queued',outcome:null} }))
  await expect(stopped).resolves.toMatchObject({cancel_requested:true,request_id:'native-request'})
  await expect(pending).resolves.toMatchObject({admission_receipt:{admission_id:admission,status:'terminal',outcome:'cancelled'}})
  expect(fetch.mock.calls.filter(([url])=>url.endsWith('/api/winman/messages/canonical-a'))).toHaveLength(1)
  expect(fetch.mock.calls.filter(([url])=>url.endsWith('/cancel'))).toHaveLength(1)
  await expect(handler('hermes:workspace:interrupt')(event,'https://cloud.example',scope,'canonical-a','native-request')).resolves.toMatchObject({cancel_requested:false})
})
