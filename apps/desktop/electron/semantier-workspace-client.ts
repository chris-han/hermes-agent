/** Semantier's existing Workspace HTTP resources, using an owned Electron session. */
export interface WorkspaceIdentity {
  authenticated: boolean
  account_provisioning?: { inference_ready?: boolean; reason?: string | null }
  organization_id?: null | string
  user?: null | { user_id: string; workspace_id: string }
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

export interface WorkspaceHistory {
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

  async function request(path: string, body?: Record<string, unknown>) {
    const response = await fetchSession(new URL(path, base.origin).href, {
      method: body ? 'POST' : 'GET',
      credentials: 'include',
      redirect: 'error',
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(body ? 120_000 : 15_000)
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
    async identity(): Promise<WorkspaceIdentity> {
      return (await request('/auth/context')).json()
    },
    async winman(): Promise<{ bot_id: string; native_profile_id: string;
      profile: { name: string; path: string; [key: string]: unknown }; bot_mode_protocol?: boolean }> {
      const result = await (await request('/api/winman')).json()
      if (!result?.bot_id || !result.native_profile_id || !result.profile?.name ||
        result.profile.path !== result.native_profile_id) {
        throw new Error('The server-owned Winman profile is unavailable.')
      }
      return result
    },
    async sessions(): Promise<{ sessions: Array<{ key: string; title?: string }> }> {
      return (await request('/api/sessions')).json()
    },
    async history(sessionKey: string): Promise<WorkspaceHistory> {
      const result: WorkspaceHistory = await (await request(`/api/history?sessionKey=${encodeURIComponent(sessionKey)}`)).json()

      // A foreign/missing Workspace history may deliberately be an empty 200.
      // Never adopt its "new" placeholder as the requested durable session.
      if (result.sessionId !== sessionKey) {
        throw Object.assign(new Error('This conversation is unavailable in the current workspace.'), { statusCode: 404 })
      }

      return result
    },
    async send(sessionKey: string, message: string, requestIdentity: string) {
      if (!requestIdentity.trim()) { throw new Error('A message request identity is required.') }

      // No automatic POST retry: a lost response is not permission to execute again.
      return request('/api/send-stream', {
        sessionKey, friendlyId: sessionKey, message, history: [],
        requestIdentity, idempotencyKey: requestIdentity, locale: 'en'
      })
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
  path: string
): Promise<unknown> {
  const url = new URL(path, 'https://desktop.invalid')
  const pathname = url.pathname
  const profile = url.searchParams.get('profile')
  if (profile && profile !== 'default' && profile !== 'all') {
    throw Object.assign(new Error('This profile requires its server-owned Cloud route.'), { statusCode: 404 })
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
