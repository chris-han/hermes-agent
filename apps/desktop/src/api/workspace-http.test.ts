import { afterEach, expect, it, vi } from 'vitest'

import { forgetWorkspaceReadScope, hermesApi, HermesGateway, setApiRequestConnection } from './client'

const scope = { userId: 'a', workspaceId: 'wa', organizationId: 'org' }
const identity = { authenticated: true, user: { user_id: 'a', workspace_id: 'wa' }, workspace: { id: 'wa' }, organization_id: 'org' }

afterEach(() => vi.unstubAllGlobals())

it('opens HTTP with real identity checks and resumes exact server IDs without a WebSocket or model turn', async () => {
  const history = vi.fn().mockResolvedValue({ sessionId: 'original', messages: [{ messageId: '204', role: 'assistant', text: 'Reminder' }] })
  const send = vi.fn()
  const socket = vi.fn(() => { throw new Error('Cloud must not dial a socket') })
  vi.stubGlobal('WebSocket', socket)
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: { identity: vi.fn().mockResolvedValue(identity), history, send } } })
  const gateway = new HermesGateway()
  const ready = vi.fn()
  gateway.on('gateway.ready', ready)
  await gateway.connectWorkspace('https://cloud.example', scope)
  expect(gateway.connectionState).toBe('open')
  const result = await gateway.request('session.resume', { session_id: 'original' })
  expect(result).toMatchObject({ session_id: 'original', messages: [{ messageId: '204', row_id: 204 }] })
  expect(socket).not.toHaveBeenCalled()
  expect(ready).not.toHaveBeenCalled()
  expect(send).not.toHaveBeenCalled()
  await expect(gateway.request('session.rewind', { session_id: 'original' })).rejects.toMatchObject({ code: -32601 })
  gateway.close()
})

it('rejects a changed account before opening the HTTP connection', async () => {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: { identity: vi.fn().mockResolvedValue({ ...identity, organization_id: 'other' }) } } })
  const gateway = new HermesGateway()
  await expect(gateway.connectWorkspace('https://cloud.example', scope)).rejects.toThrow('changed')
  expect(gateway.connectionState).toBe('error')
  gateway.close()
})

it('pins REST reads to the identity that actually opened the Cloud connection', async () => {
  const api = vi.fn().mockResolvedValue({ sessions: [] })
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api, workspace: { identity: vi.fn().mockResolvedValue(identity) } } })
  const gateway = new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example', scope, 'cloud-a')
  setApiRequestConnection('cloud-a')
  await hermesApi({ path: '/api/sessions' })
  expect(api).toHaveBeenCalledWith(expect.objectContaining({ connectionId: 'cloud-a', workspaceScope: scope }))
  setApiRequestConnection(null)
  gateway.close()
})

it('uses server-owned Cloud admission for readiness and a real context read for ping', async () => {
  const read = vi.fn().mockResolvedValue({ ...identity, account_provisioning: { inference_ready: true } })
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: { identity: read } } })
  const gateway = new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example', scope)
  await expect(gateway.request('ping')).resolves.toEqual({ ok: true })
  await expect(gateway.request('setup.runtime_check')).resolves.toMatchObject({ ok: true })
  read.mockResolvedValue({ ...identity, account_provisioning: { inference_ready: false, reason: 'admission revoked' } })
  await expect(gateway.request('setup.runtime_check')).resolves.toEqual({ ok: false, error: 'admission revoked' })
  gateway.close()
})

it('reads the Native Winman roster through the explicit account-scoped HTTP owner', async () => {
  const roster = { profiles: [{ name: 'winman-a', runtime_bot_id: 'bot-a',
    canonical_session: { id: 'canonical-a' } }], bot_mode_protocol: true }

  const api = vi.fn().mockResolvedValue(roster)
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {
    api, workspace: { identity: vi.fn().mockResolvedValue(identity) }
  } })
  const gateway = new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example', scope, 'cloud-a')
  await expect(gateway.request('profiles.list', {})).resolves.toEqual(roster)
  expect(api).toHaveBeenCalledWith(expect.objectContaining({
    connectionId: 'cloud-a', workspaceScope: scope, profile: 'default', path: '/api/profiles'
  }))
  gateway.close()
})

it('resolves Bot Chat by its server title registry only for the selected Native profile', async () => {
  const api = vi.fn().mockResolvedValue({ profiles: [{ name: 'winman-a',
    canonical_session: { id: 'canonical-a', resolved_id: 'tip-a', title: 'Bot Chat', message_count: 7 } }] })

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {
    api, workspace: { identity: vi.fn().mockResolvedValue(identity) }
  } })
  const gateway = new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example', scope, 'cloud-a')
  await expect(gateway.request('session.list', { profile: 'winman-a', title: 'Bot Chat', include_hidden: true }))
    .resolves.toMatchObject({ sessions: [{ id: 'canonical-a', resolved_id: 'tip-a' }], scope: 'stored' })
  await expect(gateway.request('session.list', { profile: 'foreign', title: 'Bot Chat' })).rejects.toThrow('profile')
  await expect(gateway.request('session.list', { profile: 'winman-a', title: 'Other' })).rejects.toMatchObject({ code: -32601 })
  gateway.close()
})

it('resumes the explicit Native Cloud profile through its owner and never sends it to Main history', async () => {
  const snapshot = { session_id: 'canonical-a', messages: [{ row_id: 44, role: 'assistant', content: 'Native reply', timestamp: 10 }] }
  const nativeHistory = vi.fn().mockResolvedValue(snapshot)
  const history = vi.fn()
  const send = vi.fn()
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {
    workspace: { identity: vi.fn().mockResolvedValue(identity), nativeHistory, history, send }
  } })
  const gateway = new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example', scope, 'cloud-a', 'winman-a')
  await expect(gateway.request('session.resume', { session_id: 'canonical-a' })).resolves.toEqual(snapshot)
  expect(nativeHistory).toHaveBeenCalledWith('https://cloud.example', scope, 'canonical-a')
  expect(history).not.toHaveBeenCalled()
  await expect(gateway.request('prompt.submit', { session_id: 'canonical-a', text: 'Hello', submission_id: 'stable' })).rejects.toThrow('owner adapter')
  expect(send).not.toHaveBeenCalled()
  gateway.close()
})


it('rejects a late account response before it can enter renderer caches', async () => {
  let resolve!: (value: unknown) => void
  const api = vi.fn(() => new Promise(done => { resolve = done }))
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {
    api, workspace: { identity: vi.fn().mockResolvedValue(identity) }
  } })
  const gateway = new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example', scope, 'cloud-a')
  setApiRequestConnection('cloud-a')
  const pending = hermesApi({ path: '/api/sessions' })
  forgetWorkspaceReadScope('cloud-a')
  resolve({ sessions: [{ id: 'A-private' }] })
  await expect(pending).rejects.toThrow('changed during this read')
  setApiRequestConnection(null)
  gateway.close()
})

it('keeps concurrent reads valid when another profile opens for the same authenticated account', async () => {
  let resolve!: (value: unknown) => void
  const api = vi.fn(() => new Promise(done => { resolve = done }))
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {
    api, workspace: { identity: vi.fn().mockResolvedValue(identity) }
  } })
  const primary = new HermesGateway()
  const bot = new HermesGateway()
  await primary.connectWorkspace('https://cloud.example', scope, 'cloud-a')
  setApiRequestConnection('cloud-a')
  const pending = hermesApi({ path: '/api/sessions' })
  await bot.connectWorkspace('https://cloud.example', { ...scope }, 'cloud-a', 'winman-a')
  resolve({ sessions: [{ id: 'own-session' }] })
  await expect(pending).resolves.toEqual({ sessions: [{ id: 'own-session' }] })
  setApiRequestConnection(null)
  primary.close()
  bot.close()
})

it('recovers changed history on network restoration without sending a turn and suppresses duplicate refreshes', async () => {
  vi.useFakeTimers()
  const first = { sessionId: 'original', messages: [{ messageId: '204', role: 'assistant', text: 'Reminder' }] }
  const history = vi.fn().mockResolvedValue(first), send = vi.fn()
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: { identity: vi.fn().mockResolvedValue(identity), history, send } } })
  const gateway = new HermesGateway(), changed = vi.fn()
  gateway.on('session.replay_gap', changed)

  try {
    await gateway.connectWorkspace('https://cloud.example', scope)
    await gateway.request('session.resume', { session_id: 'original' })
    history.mockResolvedValue({ ...first, messages: [...first.messages, { messageId: '217', role: 'assistant', text: 'Missed update' }] })
    window.dispatchEvent(new Event('online'))
    await vi.advanceTimersByTimeAsync(1)
    expect(changed).toHaveBeenCalledTimes(1)
    expect(changed.mock.calls[0][0]).toMatchObject({ session_id: 'original', profile: 'default' })
    window.dispatchEvent(new Event('online'))
    await vi.advanceTimersByTimeAsync(1)
    expect(changed).toHaveBeenCalledTimes(1)
    expect(send).not.toHaveBeenCalled()
    gateway.close()
    const count = history.mock.calls.length
    window.dispatchEvent(new Event('online'))
    await vi.advanceTimersByTimeAsync(120_000)
    expect(history).toHaveBeenCalledTimes(count)
  } finally { gateway.close(); vi.useRealTimers() }
})

it('discards a pending recovery read after the Cloud owner closes', async () => {
  vi.useFakeTimers()
  let resolve!: (value: unknown) => void

  const history = vi.fn().mockResolvedValueOnce({ sessionId: 'original', messages: [] })
    .mockImplementationOnce(() => new Promise(done => { resolve = done }))

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: { identity: vi.fn().mockResolvedValue(identity), history } } })
  const gateway = new HermesGateway(), changed = vi.fn()
  gateway.on('session.replay_gap', changed)

  try {
    await gateway.connectWorkspace('https://cloud.example', scope)
    await gateway.request('session.resume', { session_id: 'original' })
    window.dispatchEvent(new Event('online'))
    await vi.advanceTimersByTimeAsync(1)
    expect(history).toHaveBeenCalledTimes(2)
    gateway.close()
    resolve({ sessionId: 'original', messages: [{ messageId: 'late' }] })
    await vi.advanceTimersByTimeAsync(120_000)
    expect(changed).not.toHaveBeenCalled()
    expect(history).toHaveBeenCalledTimes(2)
  } finally { gateway.close(); vi.useRealTimers() }
})

it('uses the server cursor for idle reads and periodically recovers an authoritative snapshot', async () => {
  vi.useFakeTimers()

  const history = vi.fn().mockResolvedValueOnce({ sessionId: 'original', nextCursor: '218', messages: [{ messageId: '218', role: 'assistant', text: 'last' }] })
    .mockResolvedValue({ sessionId: 'original', nextCursor: '218', messages: [] })

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: { identity: vi.fn().mockResolvedValue(identity), history } } })
  const gateway = new HermesGateway(), changed = vi.fn()
  gateway.on('session.replay_gap', changed)

  try {
    await gateway.connectWorkspace('https://cloud.example', scope)
    await gateway.request('session.resume', { session_id: 'original' })
    window.dispatchEvent(new Event('online'))
    await vi.advanceTimersByTimeAsync(1)
    expect(history).toHaveBeenLastCalledWith('https://cloud.example', scope, 'original', '218')
    expect(changed).not.toHaveBeenCalled()
    vi.setSystemTime(Date.now() + 61_000)
    window.dispatchEvent(new Event('online'))
    await vi.advanceTimersByTimeAsync(1)
    expect(history.mock.calls.at(-1)?.slice(3)).toEqual([])
  } finally { gateway.close(); vi.useRealTimers() }
})

it('creates only the authenticated Native Bot Chat through its owner adapter', async () => {
  const ensureWinmanChat = vi.fn().mockResolvedValue({ bot_id: 'bot_a', profile: { name: 'winman-a', canonical_session: { id: 'canonical-a' } } })
  const api = vi.fn().mockResolvedValue({ profiles: [{ name: 'winman-a' }] })
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api, workspace: { identity: vi.fn().mockResolvedValue(identity), ensureWinmanChat } } })
  const gateway = new HermesGateway()

  try {
    await gateway.connectWorkspace('https://cloud.example', scope, 'cloud-a', 'winman-a')
    await expect(gateway.request('session.create', { profile: 'foreign', title: 'Bot Chat' })).rejects.toThrow('profile')
    expect(ensureWinmanChat).not.toHaveBeenCalled()
    await expect(gateway.request('session.create', { profile: 'winman-a', title: 'Bot Chat', hidden: true, follow_profile_config: true })).resolves.toMatchObject({ stored_session_id: 'canonical-a', session_id: 'canonical-a' })
    expect(ensureWinmanChat).toHaveBeenCalledWith('https://cloud.example', scope)
    await expect(gateway.request('session.create', { title: 'Other' })).rejects.toThrow()
    expect(ensureWinmanChat).toHaveBeenCalledTimes(1)
  } finally { gateway.close() }
})

it('sends Native chat through its retained Main binding and never the Main send endpoint', async () => {
  const context={user_id:'a',workspace_id:'wa',organization_id:'org',native_profile_id:'profile-a',binding_id:'bound-a',original_conversation:{session_key:'main-a',session_id:'main-a'}}
  const snapshot={session_id:'canonical-a',canonical_session:{id:'canonical-a'},bot_id:'bot-a',native_profile_id:'profile-a',messages:[],admission_receipt:{admission_id:'a'.repeat(32),status:'terminal',outcome:'completed',ref:{session_id:'canonical-a',profile_id:'profile-a'}}}
  const nativeContext=vi.fn().mockResolvedValue(context)
  const bindNativeContext=vi.fn().mockResolvedValue({...context,binding_id:'bound-b',original_conversation:{session_key:'main-b',session_id:'main-b'}})
  const sendNative=vi.fn().mockResolvedValue(snapshot)
  const send=vi.fn()
  Object.defineProperty(window,'hermesDesktop',{configurable:true,value:{workspace:{identity:vi.fn().mockResolvedValue(identity),nativeContext,bindNativeContext,sendNative,nativeHistory:vi.fn().mockResolvedValue(snapshot),send}}})
  const gateway=new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example',scope,'cloud-a','winman-a')
  await gateway.request('session.resume',{session_id:'canonical-a'})
  await expect(gateway.request('prompt.submit',{session_id:'canonical-a',text:'Hello',submission_id:'same-request'})).resolves.toMatchObject({submission_id:'same-request',admission_id:'a'.repeat(32),status:'terminal',outcome:'completed'})
  expect(sendNative).toHaveBeenLastCalledWith('https://cloud.example',scope,'canonical-a','Hello','same-request','bound-a')
  nativeContext.mockResolvedValue({...context,binding_id:'unrelated-window'})
  await gateway.request('session.resume',{session_id:'canonical-a'})
  expect(nativeContext).toHaveBeenCalledTimes(1)
  await gateway.request('session.bind_original',{session_id:'canonical-a',original_session_key:'main-b'})
  await gateway.request('prompt.submit',{session_id:'canonical-a',text:'Next',submission_id:'next-request'})
  expect(sendNative).toHaveBeenLastCalledWith('https://cloud.example',scope,'canonical-a','Next','next-request','bound-b')
  expect(send).not.toHaveBeenCalled()
  gateway.close()
})

it('routes an owned Cloud model.options to scoped Workspace HTTP without a socket or provider credential', async () => {
 const api=vi.fn().mockResolvedValue({discovery_contract:'provider-query-v1',providers:[]})
 Object.defineProperty(window,'hermesDesktop',{configurable:true,value:{api,workspace:{identity:vi.fn().mockResolvedValue(identity)}}})
 const gateway=new HermesGateway();await gateway.connectWorkspace('https://cloud.example',scope,'cloud-discovery')
 await gateway.request('model.options',{query_only:true,query_scope:'cloud-source',refresh:true})
 expect(api).toHaveBeenCalledWith(expect.objectContaining({connectionId:'cloud-discovery',workspaceScope:scope,path:'/api/model/options?query_only=1&query_scope=cloud-source&refresh=1'}))
 await expect(gateway.request('model.options',{query_only:false})).rejects.toThrow(/query-only/)
 gateway.close();forgetWorkspaceReadScope('cloud-discovery')
})

it('coalesces identical Cloud authentication dials without weakening a different account fence', async () => {
 let release!: (value: unknown)=>void
 const pending=new Promise(resolve=>{release=resolve})
 const auth=vi.fn().mockReturnValueOnce(pending).mockResolvedValue(identity)
 Object.defineProperty(window,'hermesDesktop',{configurable:true,value:{workspace:{identity:auth}}})
 const gateway=new HermesGateway()
 const first=gateway.connectWorkspace('https://cloud.example',scope,'same-cloud')
 const second=gateway.connectWorkspace('https://cloud.example',{...scope},'same-cloud')
 release(identity)
 await expect(Promise.all([first,second])).resolves.toEqual([undefined,undefined])
 expect(auth).toHaveBeenCalledTimes(1)
 await expect(gateway.connectWorkspace('https://cloud.example',{...scope,userId:'foreign'},'same-cloud')).rejects.toThrow(/changed/)
 gateway.close();forgetWorkspaceReadScope('same-cloud')
})

it('retains Main creation identity across an uncertain response and retires it after success', async () => {
  const createSession = vi.fn().mockRejectedValueOnce(new Error('Outcome unknown')).mockImplementation(async (_origin,_scope,_title,id)=>({sessionId:'session_desktop_'+id,messages:[]}))
  Object.defineProperty(window,'hermesDesktop',{configurable:true,value:{workspace:{identity:vi.fn().mockResolvedValue(identity),createSession}}})
  const gateway=new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example',scope,'cloud-a')
  await expect(gateway.request('session.create',{title:'New Main',fast:false})).rejects.toThrow('Outcome unknown')
  const created=await gateway.request<{session_id:string}>('session.create',{title:'New Main',fast:false})
  expect(createSession.mock.calls[1][3]).toBe(createSession.mock.calls[0][3])
  expect(created.session_id).toBe('session_desktop_'+createSession.mock.calls[0][3])
  await gateway.request('session.create',{title:'New Main',fast:false})
  expect(createSession.mock.calls[2][3]).not.toBe(createSession.mock.calls[0][3])
  gateway.close()
})

it('does not silently discard explicit Main model or tier choices', async () => {
  const createSession=vi.fn()
  Object.defineProperty(window,'hermesDesktop',{configurable:true,value:{workspace:{identity:vi.fn().mockResolvedValue(identity),createSession}}})
  const gateway=new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example',scope,'cloud-a')
  await expect(gateway.request('session.create',{model:'manual-model'})).rejects.toThrow('explicit options')
  await expect(gateway.request('session.create',{fast:true})).rejects.toThrow('explicit options')
  expect(createSession).not.toHaveBeenCalled()
  gateway.close()
})

it('acknowledges Main completion with the retained request and actual server user row', async () => {
  const send=vi.fn().mockResolvedValue({sessionId:'main-original',nextCursor:'256',messages:[
    {messageId:'255',role:'user',text:'Acknowledge once',metadata:{durable_request_id:'request-main'}},
    {messageId:'256',role:'assistant',text:'Acknowledged'}
  ]})

  Object.defineProperty(window,'hermesDesktop',{configurable:true,value:{workspace:{identity:vi.fn().mockResolvedValue(identity),send}}})
  const gateway=new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example',scope,'cloud-a')
  const result=await gateway.request('prompt.submit',{session_id:'main-original',text:'Acknowledge once',submission_id:'request-main'})
  expect(result).toEqual({session_id:'main-original',submission_id:'request-main',status:'terminal',user_row_id:255})
  expect(send).toHaveBeenCalledTimes(1)
  gateway.close()
})

it('does not acknowledge a Main completion without the matching durable request row', async () => {
  const send=vi.fn().mockResolvedValue({sessionId:'main-original',messages:[{messageId:'255',role:'user',metadata:{durable_request_id:'other-request'}}]})
  Object.defineProperty(window,'hermesDesktop',{configurable:true,value:{workspace:{identity:vi.fn().mockResolvedValue(identity),send}}})
  const gateway=new HermesGateway()
  await gateway.connectWorkspace('https://cloud.example',scope,'cloud-a')
  await expect(gateway.request('prompt.submit',{session_id:'main-original',text:'Acknowledge once',submission_id:'request-main'})).rejects.toThrow('receipt')
  expect(send).toHaveBeenCalledTimes(1)
  gateway.close()
})
