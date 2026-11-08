import { createHash } from 'node:crypto'

import { BrowserWindow, ipcMain, session } from 'electron'

import { assertWorkspaceScope, createSemantierWorkspaceClient, requireNativeCompletion, requireWorkspaceCompletion, type WorkspaceIdentity, type WorkspaceScope } from './semantier-workspace-client'

export { assertWorkspaceScope } from './semantier-workspace-client'
export type { WorkspaceScope } from './semantier-workspace-client'

export interface DesktopWorkspaceApi {
  identity: (origin: string) => Promise<WorkspaceIdentity>
  login: (origin: string) => Promise<WorkspaceIdentity>
  logout: (origin: string) => Promise<void>
  ensureWinmanChat: (origin: string, scope: WorkspaceScope) => Promise<import('./semantier-workspace-client').WorkspaceWinmanProfile>
  sessions: (origin: string, scope: WorkspaceScope) => Promise<{ sessions: Array<{ key: string; title?: string }> }>
  history: (origin: string, scope: WorkspaceScope, sessionId: string, afterId?: string) => Promise<import('./semantier-workspace-client').WorkspaceHistory>
  nativeContext: (origin: string, scope: WorkspaceScope, sessionId: string) => Promise<import('./semantier-workspace-client').WorkspaceMainContext | null>
  bindNativeContext: (origin: string, scope: WorkspaceScope, sessionId: string, originalSessionKey: string) => Promise<import('./semantier-workspace-client').WorkspaceMainContext>
  sendNative: (origin: string, scope: WorkspaceScope, sessionId: string, message: string, requestId: string, bindingId: string) => Promise<import('./semantier-workspace-client').WorkspaceNativeSnapshot>
  nativeHistory: (origin: string, scope: WorkspaceScope, sessionId: string) => Promise<import('./semantier-workspace-client').WorkspaceNativeSnapshot>
  send: (origin: string, scope: WorkspaceScope, sessionId: string, message: string, requestIdentity: string) => Promise<import('./semantier-workspace-client').WorkspaceHistory>
}

/** Cookies stay in an origin-owned Chromium jar, never in renderer payloads. */
export function workspace(origin: string) {
  // Validate before creating a persisted jar; the HTTP client owns origin policy.
  createSemantierWorkspaceClient(origin, fetch)
  const canonicalOrigin = new URL(origin).origin
  const partition = 'persist:semantier-workspace-' + createHash('sha256').update(canonicalOrigin).digest('hex')
  const owned = session.fromPartition(partition)
  return { owned, origin: canonicalOrigin, client: createSemantierWorkspaceClient(canonicalOrigin, owned.fetch.bind(owned)) }
}

export function registerSemantierWorkspaceIpc(onAccountChanged?: (origin: string, authenticated: boolean) => void): void {
  ipcMain.handle('hermes:workspace:identity', (_event, origin: string) => workspace(origin).client.identity())
  ipcMain.handle('hermes:workspace:logout', async (_event, origin: string) => {
    const { owned } = workspace(origin)
    await owned.clearStorageData()
    await owned.clearCache()
    onAccountChanged?.(new URL(origin).origin, false)
  })
  ipcMain.handle('hermes:workspace:login', async (_event, origin: string) => {
    const target = workspace(origin)
    const win = new BrowserWindow({
      width: 520, height: 720, title: 'Semantier Cloud',
      webPreferences: { session: target.owned, sandbox: true, contextIsolation: true, nodeIntegration: false }
    })
    win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
    win.webContents.on('will-navigate', (event, url) => {
      if (new URL(url).origin !== target.origin) { event.preventDefault() }
    })
    try {
      await win.loadURL(target.origin)
      const deadline = Date.now() + 120_000
      while (!win.isDestroyed() && Date.now() < deadline) {
        try {
          const identity = await target.client.identity()
          if (identity.authenticated) {
            await target.owned.cookies.flushStore()
            onAccountChanged?.(target.origin, true)
            return identity
          }
        } catch { /* The hosted sign-in owns authentication; wait for its cookie. */ }
        await new Promise(resolve => setTimeout(resolve, 500))
      }
      throw new Error('Semantier Cloud sign-in was closed or timed out.')
    } finally {
      if (!win.isDestroyed()) { win.close() }
    }
  })
  ipcMain.handle('hermes:workspace:ensure-winman-chat', async (_event, origin: string, scope: WorkspaceScope) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const result = await client.ensureWinmanChat()
    assertWorkspaceScope(await client.identity(), scope)
    return result
  })
  ipcMain.handle('hermes:workspace:sessions', async (_event, origin: string, scope: WorkspaceScope) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const result = await client.sessions()
    assertWorkspaceScope(await client.identity(), scope)
    return result
  })
  ipcMain.handle('hermes:workspace:history', async (_event, origin: string, scope: WorkspaceScope, sessionId: string, afterId?: string) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const result = await client.history(sessionId, ...(afterId === undefined ? [] : [afterId]))
    assertWorkspaceScope(await client.identity(), scope)
    return result
  })
  ipcMain.handle('hermes:workspace:native-context', async (_event, origin: string, scope: WorkspaceScope, sessionId: string) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const result = await client.nativeContext(sessionId)
    assertWorkspaceScope(await client.identity(), scope)
    return result
  })
  ipcMain.handle('hermes:workspace:bind-native-context', async (_event, origin: string, scope: WorkspaceScope, sessionId: string, originalSessionKey: string) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const result = await client.bindNativeContext(sessionId, originalSessionKey)
    assertWorkspaceScope(await client.identity(), scope)
    return result
  })
  ipcMain.handle('hermes:workspace:send-native', async (_event, origin: string, scope: WorkspaceScope, sessionId: string, message: string, requestId: string, bindingId: string) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const admitted = await client.sendNative(sessionId, message, requestId, bindingId)
    const result = await requireNativeCompletion(client, sessionId, admitted.receipt.admission_id)
    assertWorkspaceScope(await client.identity(), scope)
    return result
  })
  ipcMain.handle('hermes:workspace:native-history', async (_event, origin: string, scope: WorkspaceScope, sessionId: string) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const result = await client.nativeHistory(sessionId)
    assertWorkspaceScope(await client.identity(), scope)
    return result
  })
  ipcMain.handle('hermes:workspace:send', async (_event, origin: string, scope: WorkspaceScope, sessionId: string, message: string, requestIdentity: string) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const response = await client.send(sessionId, message, requestIdentity)
    // Finite on-demand SSE. Synchronizing history itself never submits a turn.
    await requireWorkspaceCompletion(response)
    assertWorkspaceScope(await client.identity(), scope)
    const result = await client.history(sessionId)
    assertWorkspaceScope(await client.identity(), scope)
    return result
  })
}
