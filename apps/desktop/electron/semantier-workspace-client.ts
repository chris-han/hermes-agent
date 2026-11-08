/** Semantier's existing Workspace HTTP resources, using an owned Electron session. */
export interface WorkspaceIdentity {
  authenticated: boolean
  organization_id?: null | string
  user?: null | { user_id: string; workspace_id: string }
  workspace?: null | { id: string; slug?: string }
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

export function createSemantierWorkspaceClient(origin: string, fetchSession: typeof fetch) {
  const base = new URL(origin)
  const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(base.hostname)

  if ((base.protocol !== 'https:' && !(base.protocol === 'http:' && loopback)) ||
    base.username || base.password || base.pathname !== '/' || base.search || base.hash) {
    throw new Error('Use a secure Semantier workspace origin.')
  }

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
