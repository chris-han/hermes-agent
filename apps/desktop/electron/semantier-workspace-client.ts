/** Semantier's existing Workspace HTTP resources, using an owned Electron session. */
export interface WorkspaceIdentity {
  authenticated: boolean
  account_provisioning?: { inference_ready?: boolean; reason?: string | null }
  organization_id?: null | string
  user?: null | { user_id: string; workspace_id: string; name?: string }
  workspace?: null | { id: string; slug?: string }
}

export interface WorkspaceScope {
  userId: string
  workspaceId: string
  organizationId: string
}

export function assertWorkspaceScope(identity: WorkspaceIdentity, expected: WorkspaceScope): void {
  if (!identity.authenticated || !expected?.userId || !expected.workspaceId || !expected.organizationId ||
    identity.user?.user_id !== expected.userId || identity.workspace?.id !== expected.workspaceId ||
    identity.organization_id !== expected.organizationId) {
    throw new Error('The Semantier account or workspace changed. Reconnect before continuing.')
  }
}

export interface WorkspaceMessage {
  messageId: string
  role: string
  text?: string
  content?: unknown
  metadata?: Record<string, unknown>
}

export interface WorkspaceWinmanProfile {
  bot_id: string
  native_profile_id: string
  profile: { name: string; path: string; canonical_session?: { id: string; resolved_id?: string } | null; [key: string]: unknown }
  bot_mode_protocol?: boolean
}

export interface WorkspaceMainContext {
  user_id: string
  workspace_id: string
  organization_id: string
  native_profile_id: string
  binding_id: string
  original_conversation: { session_key: string; session_id: string }
}

export interface WorkspaceNativeReceipt {
  admission_id: string
  status: 'queued' | 'started' | 'unknown' | 'terminal'
  outcome: string | null
}

export interface WorkspaceNativeCancellation {
  cancel_requested: boolean
  receipt: WorkspaceNativeReceipt
}

export interface WorkspaceNativeSubmission {
  bot_id: string
  session_id: string
  receipt: WorkspaceNativeReceipt
}

export interface WorkspaceNativeSnapshot {
  admission_receipt?: WorkspaceNativeReceipt
  session_id: string
  canonical_session: { id: string; resolved_id?: string }
  bot_id: string
  native_profile_id: string
  messages: Array<Record<string, unknown>>
  [key: string]: unknown
}

export interface WorkspaceHistory {
  nextCursor?: string
  hasMore?: boolean
  sessionId: string
  messages: WorkspaceMessage[]
}

export function normalizeSemantierWorkspaceOrigin(origin: string): string {
  const base = new URL(origin)
  const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(base.hostname)

  if ((base.protocol !== 'https:' && !(base.protocol === 'http:' && loopback)) ||
    base.username || base.password || base.pathname !== '/' || base.search || base.hash) {
    throw new Error('Use a secure Semantier workspace origin.')
  }

  return base.origin
}

export function createSemantierWorkspaceClient(origin: string, fetchSession: (input: string, init?: RequestInit) => Promise<Response>) {
  const base = new URL(normalizeSemantierWorkspaceOrigin(origin))

  async function request(path: string, body?: Record<string, unknown>, method = 'POST', signal?: AbortSignal) {
    const response = await fetchSession(new URL(path, base.origin).href, {
      method: body ? method : 'GET',
      credentials: 'include',
      cache: 'no-store',
      redirect: 'error',
      headers: body ? { 'Content-Type': 'application/json', Origin:base.origin, 'X-Semantier-CSRF':'1' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(body ? 120_000 : 15_000)]) : AbortSignal.timeout(body ? 120_000 : 15_000)
    })

    if (!response.ok) {
      throw Object.assign(new Error(`Semantier workspace request failed (${response.status}).`), {
        statusCode: response.status,
        needsWorkspaceLogin: response.status === 401 || response.status === 403
      })
    }

    return response
  }

  return {
    async createSession(label: string, requestIdentity: string): Promise<{ sessionKey: string }> {
      if (!/^[A-Za-z0-9_-]{1,96}$/.test(requestIdentity)) { throw new Error('Invalid Main creation request identity.') }
      const result = await (await request('/api/sessions', { label, requestIdentity })).json()
      if (result.ok !== true || typeof result.sessionKey !== 'string' || !result.sessionKey ||
        result.sessionKey !== result.entry?.key) { throw new Error('Main creation did not return its server-owned conversation.') }
      return { sessionKey: result.sessionKey }
    },
    async auxiliaryModels() {
      const data = await (await request('/api/hermes-config')).json()
      const main = data.config?.model || {}
      return {main:{provider:main.provider || '',model:main.default || ''},
        tasks:Object.entries(data.config?.auxiliary || {}).map(([task,value])=>{
          const r = value as Record<string, unknown>
          return {task,provider:r.provider || '',model:r.model || r.default || '',base_url:r.base_url || ''}
        })}
    },
    async modelOptions(refresh: boolean) {
      const [inventory, config, catalog] = await Promise.all([
        request('/api/hermes-proxy/api/available-models?provider=configured&refresh=' + refresh).then(r => r.json()),
        request('/api/hermes-config').then(r => r.json()),
        request('/api/hermes-proxy/api/cloud-provider/catalog').then(r => r.json()).catch(error => {
          if (error.statusCode === 401) {throw error}
          return {object:'list',data:[],discovery_state:'error'}
        })
      ])
      if (inventory.discovery_contract !== 'provider-query-v1' || !Array.isArray(inventory.providers) ||
          catalog.object !== 'list' || !Array.isArray(catalog.data)) {
        throw new Error('The Workspace model discovery contract is unavailable.')
      }
      const managed = catalog.data
      return { discovery_contract: 'provider-query-v1',
        provider: config.activeProvider || '', model: config.activeModel || '',
        providers: [...inventory.providers, {slug:'semantier-cloud', name:'Semantier Model Provider',
          authenticated:catalog.discovery_state !== 'error', models:managed.map((r: any)=>r.id), eligible_models:managed.map((r: any)=>r.id),
          model_profiles:Object.fromEntries(managed.map((r: any)=>[r.id,r])),
          discovery_state:catalog.discovery_state || 'fresh', discovery_source_id:'managed:' + JSON.stringify(managed.map((r: any)=>[r.account_id,r.revision])),
          discovery_fetched_at:new Date().toISOString(), total_models:managed.length}]
      }
    },
    async setMainModel(body: Record<string, unknown>, scope: WorkspaceScope) {
      if (body.scope !== 'main' || Object.keys(body).some(k=>!['scope','provider','model'].includes(k)) ||
          typeof body.provider !== 'string' || typeof body.model !== 'string') {
        throw new Error('Only an explicit Main Model selection is supported here.')
      }
      const expected_scope = {user_id:scope.userId,workspace_id:scope.workspaceId,organization_id:scope.organizationId}
      const config = await (await request('/api/hermes-config')).json()
      if (body.provider === 'semantier-cloud') {
        const catalog = await (await request('/api/hermes-proxy/api/cloud-provider/catalog')).json()
        const profile = catalog.data?.find((p: any)=>p.id === body.model)
        if (!profile) {throw new Error('Refresh the authorized managed model catalog before selecting.')}
        await request('/api/hermes-proxy/api/model/set', {...body,account_id:profile.account_id,revision:profile.revision,expected_scope})
      } else {
        const route = config.config?.providers?.[body.provider]
        if (!route || !['alibaba','openai','custom','litellm'].includes(body.provider)) {throw new Error('Configure this provider in Workspace before selecting.')}
        await request('/api/hermes-config', {expected_scope,config:{provider:body.provider,model:{provider:body.provider,
          default:body.model,base_url:route.base_url || '',key_env:route.key_env || '',api_key:'',api_key_env:'',
          api_mode:route.api_mode || '',transport:route.transport || ''}}}, 'PATCH')
      }
      return {ok:true,provider:body.provider,model:body.model}
    },
    async identity(): Promise<WorkspaceIdentity> {
      return (await request('/auth/context')).json()
    },
    async winman(): Promise<WorkspaceWinmanProfile> {
      const result = await (await request('/api/winman')).json()
      if (!result?.bot_id || !result.native_profile_id || !result.profile?.name ||
        result.profile.path !== result.native_profile_id) {
        throw new Error('The server-owned Winman profile is unavailable.')
      }
      return result
    },
    async ensureWinmanChat(): Promise<WorkspaceWinmanProfile> {
      // No automatic retry. Native's durable title creation identity owns recovery.
      const result: WorkspaceWinmanProfile = await (await request('/api/winman', {})).json()
      if (!result?.bot_id || !result.native_profile_id || !result.profile?.name ||
          result.profile.path !== result.native_profile_id || !result.profile.canonical_session?.id) {
        throw new Error('The server-owned canonical Bot Chat is unavailable.')
      }
      return result
    },
    async sessions(): Promise<{ sessions: Array<{ key: string; title?: string }> }> {
      return (await request('/api/sessions')).json()
    },
    async nativeContext(sessionId: string): Promise<WorkspaceMainContext | null> {
      return (await request(`/api/winman/context/${encodeURIComponent(sessionId)}`)).json()
    },
    async bindNativeContext(sessionId: string, originalSessionKey: string): Promise<WorkspaceMainContext> {
      return (await request(`/api/winman/context/${encodeURIComponent(sessionId)}`, { original_session_key: originalSessionKey })).json()
    },
    async sendNative(sessionId: string, text: string, requestId: string, bindingId: string): Promise<WorkspaceNativeSubmission> {
      if (!/^[A-Za-z0-9_-]{1,128}$/.test(requestId) || !bindingId || !text.trim()) { throw new Error('A retained Native request identity and Main binding are required.') }
      const result: WorkspaceNativeSubmission = await (await request(`/api/winman/messages/${encodeURIComponent(sessionId)}`, {
        text, request_id: requestId, binding_id: bindingId
      })).json()
      if (!result.bot_id || result.session_id !== sessionId || !/^[a-f0-9]{32}$/.test(result.receipt?.admission_id)) {
        throw new Error('Native admission identity is unavailable. Recover history before sending again.')
      }
      return result
    },
    async cancelNative(sessionId: string, admissionId: string): Promise<WorkspaceNativeCancellation> {
      if (!/^[a-f0-9]{32}$/.test(admissionId)) throw new Error('Exact Native admission required.')
      const result: WorkspaceNativeCancellation = await (await request(`/api/winman/messages/${encodeURIComponent(sessionId)}/cancel`, { admission_id: admissionId })).json()
      if (typeof result.cancel_requested !== 'boolean' || result.receipt?.admission_id !== admissionId)
        throw new Error('Native cancellation receipt is uncertain; recover history before retrying.')
      return result
    },
    async nativeHistory(sessionId: string, admissionId?: string): Promise<WorkspaceNativeSnapshot> {
      if (admissionId !== undefined && !/^[a-f0-9]{32}$/.test(admissionId)) { throw new Error('Invalid Native admission identity.') }
      const query = admissionId === undefined ? '' : `?admission_id=${admissionId}`
      const result: WorkspaceNativeSnapshot = await (await request(`/api/winman/history/${encodeURIComponent(sessionId)}${query}`)).json()
      const ids = [result.canonical_session?.id, result.canonical_session?.resolved_id]
      if (!result.bot_id || !result.native_profile_id || !ids.includes(sessionId) || !ids.includes(result.session_id)) {
        throw new Error('This canonical Bot Chat is unavailable in the current workspace.')
      }
      return result
    },
    async history(sessionKey: string, afterId?: string): Promise<WorkspaceHistory> {
      if (afterId !== undefined && !/^\d+$/.test(afterId)) { throw new Error('Invalid message cursor.') }
      const query = new URLSearchParams({ sessionKey })
      if (afterId !== undefined) { query.set('afterId', afterId) }
      const result: WorkspaceHistory = await (await request(`/api/history?${query}`)).json()

      // A foreign/missing Workspace history may deliberately be an empty 200.
      // Never adopt its "new" placeholder as the requested durable session.
      if (result.sessionId !== sessionKey) {
        throw Object.assign(new Error('This conversation is unavailable in the current workspace.'), { statusCode: 404 })
      }

      return result
    },
    async send(sessionKey: string, message: string, requestIdentity: string, signal?: AbortSignal) {
      if (!requestIdentity.trim()) { throw new Error('A message request identity is required.') }

      // No automatic POST retry: a lost response is not permission to execute again.
      return request('/api/send-stream', {
        sessionKey, friendlyId: sessionKey, message, history: [],
        requestIdentity, idempotencyKey: requestIdentity, locale: 'en'
      }, 'POST', signal)
    }
  }
}

/** Present existing server IDs in Desktop's transcript shape; never mint replacements. */
export function workspaceTranscript(history: WorkspaceHistory) {
  return history.messages.map(message => {
    const numericId = /^\d+$/.test(message.messageId) ? Number(message.messageId) : NaN
    return {
      ...message,
      content: message.text ?? message.content,
      ...(Number.isSafeInteger(numericId) ? { id: numericId, row_id: numericId } : {}),
      display_metadata: message.metadata,
      // Workspace timestamps are milliseconds; Native transcript timestamps are seconds.
      timestamp: typeof (message as WorkspaceMessage & { timestamp?: number }).timestamp === 'number'
        ? (message as WorkspaceMessage & { timestamp: number }).timestamp / 1000
        : undefined
    }
  })
}

/** Read projection for the existing Desktop Sessions surface, not another store. */
export async function workspaceDesktopRead(
  client: ReturnType<typeof createSemantierWorkspaceClient>,
  path: string,
  routeProfile?: string
): Promise<unknown> {
  const url = new URL(path, 'https://desktop.invalid')
  const pathname = url.pathname
  const profile = routeProfile && routeProfile !== 'default' ? routeProfile : url.searchParams.get('profile')
  if (pathname === '/api/model/auxiliary' && (!profile || profile === 'default')) {
    return client.auxiliaryModels()
  }
  if (pathname === '/api/model/options' && (!profile || profile === 'default')) {
    return client.modelOptions(url.searchParams.get('refresh') === '1')
  }
  if (profile && profile !== 'default' && profile !== 'all') {
    const owner = await client.winman()
    if (owner.profile.name !== profile) {
      throw Object.assign(new Error('This Native profile is unavailable in the current workspace.'), { statusCode: 404 })
    }
    if (pathname === '/api/profiles') {
      return { profiles: [{ ...owner.profile, runtime_bot_id: owner.bot_id,
        native_profile_id: owner.native_profile_id }], bot_mode_protocol: owner.bot_mode_protocol }
    }
    const canonical = owner.profile.canonical_session as null | { id: string; resolved_id?: string; [key: string]: unknown }
    const match = /^\/api\/sessions\/([^/]+)(\/messages)?$/.exec(pathname)
    if (match && canonical && [canonical.id, canonical.resolved_id].includes(decodeURIComponent(match[1]))) {
      if (match[2]) {
        const snapshot = await client.nativeHistory(decodeURIComponent(match[1]))
        return { session_id: snapshot.session_id, messages: snapshot.messages, total: snapshot.messages.length }
      }
      return { ...canonical, id: decodeURIComponent(match[1]), profile }
    }
    if (pathname === '/api/sessions' || pathname === '/api/profiles/sessions') {
      return { sessions: canonical ? [{ ...canonical, profile }] : [], total: canonical ? 1 : 0, offset: 0 }
    }
    throw Object.assign(new Error('This Native Bot Chat operation is unavailable.'), { statusCode: 404 })
  }
  if (url.searchParams.get('archived') === 'only') {
    throw Object.assign(new Error('Archived-only enumeration is unavailable on this Workspace route.'), { statusCode: 501 })
  }
  const missing = () => Object.assign(new Error(`Semantier Workspace does not expose this Desktop operation: ${pathname}`), { statusCode: 404 })
  const list = async () => {
    const { sessions } = await client.sessions()
    return sessions.map(item => {
      const row = item as typeof item & { createdAt?: number; updatedAt?: number; status?: string; model?: string; platform?: string }
      return {
        ...row, id: row.key, profile: 'default',
        started_at: row.createdAt === undefined ? undefined : row.createdAt / 1000,
        last_active: row.updatedAt === undefined ? undefined : row.updatedAt / 1000,
        is_active: row.status === 'running', source: row.platform
      }
    })
  }

  if (pathname === '/api/profiles') {
    const result = await client.winman()
    return { profiles: [{ ...result.profile, runtime_bot_id: result.bot_id,
      native_profile_id: result.native_profile_id }], bot_mode_protocol: result.bot_mode_protocol }
  }

  if (pathname === '/api/sessions' || pathname === '/api/profiles/sessions') {
    const sessions = await list()
    // This is the actual returned page, not a fabricated global count.
    return { sessions, total: sessions.length, offset: 0 }
  }

  if (pathname === '/api/profiles/sessions/sidebar') {
    const sessions = await list()
    return {
      recents: { sessions },
      cron: { sessions: [], unavailable: true },
      messaging: { sessions: [], unavailable: true }
    }
  }

  const match = /^\/api\/sessions\/([^/]+)(\/messages)?$/.exec(pathname)
  if (match) {
    const sessionId = decodeURIComponent(match[1])
    if (match[2]) {
      const history = await client.history(sessionId)
      const messages = workspaceTranscript(history)
      return { session_id: history.sessionId, messages, total: messages.length }
    }
    const row = (await list()).find(item => item.id === sessionId)
    if (!row) { throw missing() }
    return row
  }

  throw missing()
}

/** A lost or error terminal event is not a successful send, and never authorizes retry. */
export async function requireWorkspaceCompletion(response: Response): Promise<void> {
  const stream = (await response.text()).replace(/\r\n/g, '\n')
  let completed = false
  for (const frame of stream.split('\n\n')) {
    const lines = frame.split('\n')
    if (!lines.some(line => line.trim() === 'event: done')) { continue }
    const data = lines.filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n')
    const terminal = JSON.parse(data)
    if (terminal.state === 'error') { throw new Error('The Cloud turn failed. Recover its history before trying again.') }
    if (terminal.state === 'complete') { completed = true }
  }
  if (!completed) { throw new Error('The Cloud send outcome is uncertain. Recover its history before trying again.') }
}

/** Finite receipt reads after one explicit submit; an uncertain outcome never resubmits. */
export async function requireNativeCompletion(
  client: Pick<ReturnType<typeof createSemantierWorkspaceClient>, 'nativeHistory'>,
  sessionId: string, admissionId: string, cancelRequested?: AbortSignal
): Promise<WorkspaceNativeSnapshot> {
  const deadline = Date.now() + 120_000
  let delay = 250
  while (Date.now() < deadline) {
    const snapshot = await client.nativeHistory(sessionId, admissionId)
    const receipt = snapshot.admission_receipt
    if (receipt?.admission_id !== admissionId) { throw new Error('Native receipt changed. Recover history before sending again.') }
    if (receipt.status === 'terminal') {
      if (cancelRequested?.aborted && ['cancelled', 'interrupted'].includes(receipt.outcome || '')) return snapshot
      if (receipt.outcome !== 'completed') { throw new Error('Winman did not complete this request. Recover its status before sending again.') }
      return snapshot
    }
    if (receipt.status === 'unknown') { throw new Error('Winman execution outcome is unknown. Recover its status before sending again.') }
    await new Promise(resolve => setTimeout(resolve, delay))
    delay = Math.min(delay * 2, 2_000)
  }
  throw new Error('Winman is still processing the retained request. Recover its status before sending again.')
}
