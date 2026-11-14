import { expect, it, vi } from 'vitest'

import { createSemantierWorkspaceClient, requireWorkspaceCompletion, workspaceDesktopRead, workspaceTranscript } from './semantier-workspace-client'

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

it('projects real configured BYOK and managed rows without copying credentials or entitlements', async () => {
  const query = { discovery_contract: 'provider-query-v1', providers: [{slug:'openai',name:'OpenAI',authenticated:true,models:['returned-model'],discovery_state:'fresh',discovery_source_id:'opaque',discovery_fetched_at:'2026-11-08T00:00:00Z'}] }
  const transport = vi.fn<typeof fetch>().mockImplementation(async url => new Response(JSON.stringify(String(url).includes('available-models') ? query : String(url).includes('catalog') ? {object:'list',data:[{id:'managed-profile',capabilities:{},request_parameters:{}}]} : {activeProvider:'openai',activeModel:'returned-model'})))
  const client = createSemantierWorkspaceClient('https://cloud.example', transport)
  const result = await workspaceDesktopRead(client, '/api/model/options?query_only=1&refresh=1') as any
  expect(result.discovery_contract).toBe('provider-query-v1')
  expect(result.providers[0]).toMatchObject({slug:'openai',models:['returned-model']})
  expect(result.providers[0].eligible_models).toBeUndefined()
  expect(result.providers[1]).toMatchObject({slug:'semantier-cloud',models:['managed-profile'],eligible_models:['managed-profile']})
  expect(transport.mock.calls.map(([url])=>String(url))).toContain('https://cloud.example/api/hermes-proxy/api/available-models?provider=configured&refresh=true')
})

it('keeps provider identity and native credential pointers in an explicit Main save; never copies a key', async () => {
  const transport = vi.fn<typeof fetch>().mockImplementation(async (_url, init) => new Response(JSON.stringify(init?.method === 'PATCH' ? {ok:true} : {config:{providers:{openai:{base_url:'https://api.openai.com/v1',key_env:'OPENAI_API_KEY',api_key:'[configured]'}}}})))
  const client = createSemantierWorkspaceClient('https://cloud.example',transport)
  await expect(client.setMainModel({scope:'main',provider:'openai',model:'actual-model'},{userId:'a',workspaceId:'wa',organizationId:'org'})).resolves.toMatchObject({ok:true,provider:'openai',model:'actual-model'})
  const [,options] = transport.mock.calls[1]
  expect(options?.method).toBe('PATCH')
  expect(JSON.parse(String(options?.body))).toMatchObject({expected_scope:{user_id:'a',workspace_id:'wa',organization_id:'org'},config:{model:{provider:'openai',default:'actual-model',key_env:'OPENAI_API_KEY',api_key:''}}})
  await expect(client.setMainModel({scope:'main',provider:'openai',model:'actual-model',api_key:'forbidden'},{userId:'a',workspaceId:'wa',organizationId:'org'})).rejects.toThrow()
  expect(transport).toHaveBeenCalledTimes(2)
})

it('refuses a foreign profile before querying any inventory', async () => {
  const transport = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({bot_id:'bot',native_profile_id:'native',profile:{name:'winman',path:'native'}})))
  const client = createSemantierWorkspaceClient('https://cloud.example',transport)
  await expect(workspaceDesktopRead(client,'/api/model/options?profile=foreign','default')).rejects.toMatchObject({statusCode:404})
  expect(transport.mock.calls.every(([url])=>!String(url).includes('available-models'))).toBe(true)
})

it.each([403,503])('keeps managed refusal %s separate from BYOK discovery and grants', async status => {
 const transport=vi.fn<typeof fetch>().mockImplementation(async url=>String(url).includes('catalog')?new Response('{}',{status}):new Response(JSON.stringify(String(url).includes('available-models')?{discovery_contract:'provider-query-v1',providers:[{slug:'custom',models:['returned'],discovery_state:'fresh',discovery_source_id:'same-source',discovery_fetched_at:'2026-11-08T00:00:00Z'}]}:{activeProvider:'custom',activeModel:'returned'})))
 const result=await createSemantierWorkspaceClient('https://cloud.example',transport).modelOptions(true)
 expect(result.providers[0].models).toEqual(['returned'])
 expect(result.providers[1]).toMatchObject({slug:'semantier-cloud',authenticated:false,discovery_state:'error',models:[],eligible_models:[]})
})

it('creates Main through the existing Workspace owner with a retained request identity', async () => {
  const transport = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({ok:true,sessionKey:'session_desktop_create-001',entry:{key:'session_desktop_create-001'}})))
  const client = createSemantierWorkspaceClient('https://cloud.example', transport)
  await expect(client.createSession('New Main', 'create-001')).resolves.toMatchObject({sessionKey:'session_desktop_create-001'})
  expect(transport).toHaveBeenCalledTimes(1)
  expect(JSON.parse(String(transport.mock.calls[0][1]?.body))).toEqual({label:'New Main',requestIdentity:'create-001'})
})

it('never retries an uncertain Main creation', async () => {
  const transport = vi.fn<typeof fetch>().mockRejectedValue(new TypeError('Connection lost'))
  await expect(createSemantierWorkspaceClient('https://cloud.example',transport).createSession('', 'create-001')).rejects.toThrow()
  expect(transport).toHaveBeenCalledTimes(1)
})

it('cancels only a retained Native admission and validates the returned receipt', async () => {
  const fetch = vi.fn().mockResolvedValue(Response.json({ cancel_requested:true, receipt:{admission_id:'a'.repeat(32),status:'terminal',outcome:'cancelled'} }))
  const client = createSemantierWorkspaceClient('https://cloud.example',fetch)
  await expect(client.cancelNative('canonical-a','invalid')).rejects.toThrow('admission')
  expect(fetch).not.toHaveBeenCalled()
  await expect(client.cancelNative('canonical-a','a'.repeat(32))).resolves.toMatchObject({cancel_requested:true})
  expect(fetch).toHaveBeenCalledWith('https://cloud.example/api/winman/messages/canonical-a/cancel',expect.objectContaining({method:'POST',body:JSON.stringify({admission_id:'a'.repeat(32)})}))
  fetch.mockResolvedValue(Response.json({cancel_requested:true,receipt:{admission_id:'b'.repeat(32)}}))
  await expect(client.cancelNative('canonical-a','a'.repeat(32))).rejects.toThrow('receipt')
})
