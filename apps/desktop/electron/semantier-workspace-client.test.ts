import { expect, it, vi } from 'vitest'

import { createSemantierWorkspaceClient, workspaceTranscript, workspaceDesktopRead, requireWorkspaceCompletion } from './semantier-workspace-client'

it('refuses insecure remote origins and URL credentials', () => {
  for (const origin of ['http://cloud.example', 'https://user:secret@cloud.example', 'https://cloud.example/other']) {
    expect(() => createSemantierWorkspaceClient(origin, fetch)).toThrow()
  }
})

it('does not adopt a foreign history placeholder as the requested session', async () => {
  const transport = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({ sessionId: 'new', messages: [] })))
  const client = createSemantierWorkspaceClient('https://cloud.example', transport)
  await expect(client.history('session_original')).rejects.toMatchObject({ statusCode: 404 })
})

it('does not retry an uncertain send and retains its request identity', async () => {
  const transport = vi.fn<typeof fetch>().mockRejectedValue(new TypeError('Connection lost'))
  const client = createSemantierWorkspaceClient('https://cloud.example', transport)
  await expect(client.send('session_original', 'Hello', 'request-001')).rejects.toThrow('Connection lost')
  expect(transport).toHaveBeenCalledTimes(1)
  const [url, options] = transport.mock.calls[0]
  expect(url).toBe('https://cloud.example/api/send-stream')
  expect(options).toMatchObject({ credentials: 'include', redirect: 'error', method: 'POST' })
  expect(JSON.parse(String(options?.body))).toMatchObject({ sessionKey: 'session_original', requestIdentity: 'request-001', idempotencyKey: 'request-001' })
})

it('marks revoked authentication for recovery without retry', async () => {
  const transport = vi.fn<typeof fetch>().mockResolvedValue(new Response('', { status: 403 }))
  const client = createSemantierWorkspaceClient('https://cloud.example', transport)
  await expect(client.identity()).rejects.toMatchObject({ statusCode: 403, needsWorkspaceLogin: true })
  expect(transport).toHaveBeenCalledTimes(1)
})

it('projects server conversation and message IDs without replacement or precision loss', async () => {
  const history = { sessionId: 'session_original', messages: [
    { messageId: '204', role: 'assistant', text: 'Reminder', timestamp: 1794102384000, metadata: { author_bot_id: 'bot_a' } },
    { messageId: '9007199254740993', role: 'user', text: 'Hello' }
  ] }
  const rows = workspaceTranscript(history)
  expect(rows[0]).toMatchObject({ messageId: '204', row_id: 204, content: 'Reminder', timestamp: 1794102384, display_metadata: { author_bot_id: 'bot_a' } })
  expect(rows[1].messageId).toBe('9007199254740993')
  expect(rows[1]).not.toHaveProperty('row_id')
  const transport = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify(history)))
  const client = createSemantierWorkspaceClient('https://cloud.example', transport)
  const page = await workspaceDesktopRead(client, '/api/sessions/session_original/messages?order=latest')
  expect(page).toMatchObject({ session_id: 'session_original', total: 2 })
  await expect(workspaceDesktopRead(client, '/api/config')).rejects.toMatchObject({ statusCode: 404 })
})

it('requires the actual completed SSE terminal and rejects failed or uncertain outcomes', async () => {
  await expect(requireWorkspaceCompletion(new Response('event: done\r\ndata: {"state":"complete"}\r\n\r\n'))).resolves.toBeUndefined()
  await expect(requireWorkspaceCompletion(new Response('event: done\ndata: {"state":"error"}\n\n'))).rejects.toThrow('failed')
  await expect(requireWorkspaceCompletion(new Response('event: delta\ndata: {"text":"unfinished"}\n\n'))).rejects.toThrow('uncertain')
})

it('projects only the server-selected Native Winman profile and canonical chat', async () => {
  const profile = { name: 'winman-server', path: 'profile-server', display_name: 'Custom Wing',
    canonical_session: { id: 'canonical-server', resolved_id: 'canonical-server' } }
  const transport = vi.fn<typeof fetch>().mockResolvedValue(Response.json({
    bot_id: 'bot-server', native_profile_id: 'profile-server', profile, bot_mode_protocol: true
  }))
  const client = createSemantierWorkspaceClient('https://cloud.example', transport)
  await expect(workspaceDesktopRead(client, '/api/profiles')).resolves.toMatchObject({
    profiles: [{ ...profile, runtime_bot_id: 'bot-server', native_profile_id: 'profile-server' }],
    bot_mode_protocol: true
  })
  expect(transport).toHaveBeenCalledWith('https://cloud.example/api/winman', expect.objectContaining({ method: 'GET' }))
  expect(transport).toHaveBeenCalledTimes(1)
})

it('keeps canonical Native history IDs and refuses another conversation', async () => {
  const snapshot = { session_id: 'canonical-a', canonical_session: { id: 'canonical-a' },
    bot_id: 'bot-a', native_profile_id: 'native-a', messages: [{ row_id: 44, role: 'assistant', content: 'Native reply', timestamp: 10 }] }
  const transport = vi.fn<typeof fetch>().mockImplementation(async () => Response.json(snapshot))
  const client = createSemantierWorkspaceClient('https://cloud.example', transport)
  await expect(client.nativeHistory('canonical-a')).resolves.toEqual(snapshot)
  await expect(client.nativeHistory('foreign')).rejects.toThrow('canonical')
  expect(transport.mock.calls[0][0]).toBe('https://cloud.example/api/winman/history/canonical-a')
})

it('uses server-owned keyset cursors without changing conversation identity', async () => {
  const transport = vi.fn<typeof fetch>().mockResolvedValue(Response.json({ sessionId: 'original', messages: [], nextCursor: '218', hasMore: false }))
  const client = createSemantierWorkspaceClient('https://cloud.example', transport)
  await expect(client.history('original', '218')).resolves.toMatchObject({ sessionId: 'original', nextCursor: '218', messages: [] })
  expect(transport.mock.calls[0][0]).toBe('https://cloud.example/api/history?sessionKey=original&afterId=218')
  await expect(client.history('original', '218&other=1')).rejects.toThrow('Invalid message cursor')
  expect(transport).toHaveBeenCalledTimes(1)
})

it('ensures only the server canonical Chat with an empty JSON request and no POST retry', async () => {
  const response={bot_id:'bot_a',native_profile_id:'/owned/winman-a',profile:{name:'winman-a',path:'/owned/winman-a',canonical_session:{id:'canonical-a'}}}
  const transport=vi.fn<typeof fetch>().mockResolvedValueOnce(Response.json(response)).mockRejectedValueOnce(new TypeError('lost response'))
  const client=createSemantierWorkspaceClient('https://cloud.example',transport)
  await expect(client.ensureWinmanChat()).resolves.toEqual(response)
  expect(transport.mock.calls[0]).toMatchObject(['https://cloud.example/api/winman',{method:'POST',body:'{}',credentials:'include',redirect:'error'}])
  await expect(client.ensureWinmanChat()).rejects.toThrow('lost response')
  expect(transport).toHaveBeenCalledTimes(2)
})

it('retains the server Main binding and Native request identity without retrying a lost send', async () => {
  const transport=vi.fn<typeof fetch>().mockResolvedValueOnce(Response.json({binding_id:'server-token',original_conversation:{session_key:'main-a'}}))
    .mockRejectedValueOnce(new TypeError('Connection lost'))
  const client=createSemantierWorkspaceClient('https://cloud.example',transport)
  await expect(client.bindNativeContext('canonical-a','main-a')).resolves.toMatchObject({binding_id:'server-token'})
  await expect(client.sendNative('canonical-a','Hello','same-request','server-token')).rejects.toThrow('Connection lost')
  expect(transport).toHaveBeenCalledTimes(2)
  expect(JSON.parse(String(transport.mock.calls[1][1]?.body))).toEqual({text:'Hello',request_id:'same-request',binding_id:'server-token'})
})

it('reconciles a completed Native admission by reads and refuses an unknown outcome', async () => {
  const { requireNativeCompletion }=await import('./semantier-workspace-client')
  const id='a'.repeat(32)
  const snapshot={session_id:'canonical-a',canonical_session:{id:'canonical-a'},bot_id:'bot-a',native_profile_id:'profile-a',messages:[],admission_receipt:{admission_id:id,status:'terminal' as const,outcome:'completed'}}
  const nativeHistory=vi.fn().mockResolvedValue(snapshot)
  await expect(requireNativeCompletion({nativeHistory},'canonical-a',id)).resolves.toEqual(snapshot)
  expect(nativeHistory).toHaveBeenCalledExactlyOnceWith('canonical-a',id)
  nativeHistory.mockResolvedValue({...snapshot,admission_receipt:{admission_id:id,status:'unknown',outcome:null}})
  await expect(requireNativeCompletion({nativeHistory},'canonical-a',id)).rejects.toThrow('unknown')
  expect(nativeHistory).toHaveBeenCalledTimes(2)
})
