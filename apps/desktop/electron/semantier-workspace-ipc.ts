import { createHash } from 'node:crypto'

import { BrowserWindow, ipcMain, session } from 'electron'

import type {ManagedDeviceEnrollment} from './managed-device-enrollment'
import type {PersonalCapabilitySync,PersonalPin,PersonalSnapshot} from './personal-capability-sync'
import type { TutorProgress } from './semantier-workspace-client'
import { assertWorkspaceScope, createSemantierWorkspaceClient, type ManagedDeviceInventory, type ManagedDevicePermission, type PersistentFollowup, requireNativeCompletion, requireWorkspaceCompletion, type WinmanCapability, WORKSPACE_USER_STOP_ERROR, workspaceDesktopRead, type WorkspaceHistory, type WorkspaceIdentity, type WorkspaceMainContext, type WorkspaceNativeCancellation, type WorkspaceNativeSnapshot, type WorkspaceNativeSubmission, type WorkspaceScope, workspaceSidebarSessions, type WorkspaceWinmanProfile } from './semantier-workspace-client'
import type { WorkspaceProject, WorkspaceProjectCommit, WorkspaceProjectContinuation, WorkspaceProjectHead, WorkspaceProjectMaterials, WorkspaceProjectPin } from './semantier-workspace-client'

export { assertWorkspaceScope } from './semantier-workspace-client'
export type { WorkspaceScope } from './semantier-workspace-client'

export interface DesktopWorkspaceApi {
  projectMaterials: (origin: string, scope: WorkspaceScope) => Promise<WorkspaceProjectMaterials>
  projects: (origin: string, scope: WorkspaceScope) => Promise<WorkspaceProject[]>
  createProject: (origin: string, scope: WorkspaceScope, graph: WorkspaceProjectPin, consent: true) => Promise<WorkspaceProject>
  tutorProgress: (origin: string, scope: WorkspaceScope, projectRef: string, planRef: string) => Promise<TutorProgress>
  continueProject: (origin: string, scope: WorkspaceScope, projectRef: string) => Promise<WorkspaceProjectContinuation>
  commitProject: (origin: string, scope: WorkspaceScope, projectRef: string, decision: WorkspaceProjectCommit) => Promise<WorkspaceProjectHead>
  replayProject: (origin: string, scope: WorkspaceScope, projectRef: string, head: WorkspaceProjectPin) => Promise<{ head: WorkspaceProjectHead }>
  personalPreview:(origin:string,scope:WorkspaceScope,skillId:string)=>ReturnType<PersonalCapabilitySync['preview']>
  personalSources:(origin:string,scope:WorkspaceScope)=>ReturnType<PersonalCapabilitySync['sources']>
  personalShare:(origin:string,scope:WorkspaceScope,selected:PersonalSnapshot,context:PersonalSnapshot['contexts'][number])=>ReturnType<PersonalCapabilitySync['share']>
  personalPrepare:(origin:string,scope:WorkspaceScope,receipt:PersonalPin)=>ReturnType<PersonalCapabilitySync['prepare']>
  personalAdmit:(origin:string,scope:WorkspaceScope,receipt:PersonalPin,candidate:PersonalPin)=>ReturnType<PersonalCapabilitySync['admit']>
  personalInstall:(origin:string,scope:WorkspaceScope,receipt:PersonalPin,candidate:PersonalPin,replaceExisting?:boolean)=>ReturnType<PersonalCapabilitySync['install']>

  managedDeviceEnroll:(origin:string,scope:WorkspaceScope,replacesDeviceId?:string)=>ReturnType<ManagedDeviceEnrollment['enroll']>
  managedDevices: (origin: string, scope: WorkspaceScope) => Promise<ManagedDeviceInventory>
  managedDeviceConsent: (origin: string, scope: WorkspaceScope, deviceId: string, permissions: ManagedDevicePermission[]) => Promise<unknown>
  managedDeviceRevoke: (origin: string, scope: WorkspaceScope, deviceId: string) => Promise<{ device_id: string; status: 'revoked' }>
  identity: (origin: string) => Promise<WorkspaceIdentity>
  login: (origin: string) => Promise<WorkspaceIdentity>
  logout: (origin: string) => Promise<void>
  winmanCapability: (origin: string, scope: WorkspaceScope, enabled?: boolean) => Promise<WinmanCapability>
  persistentFollowups: (origin: string, scope: WorkspaceScope) => Promise<{ items: PersistentFollowup[] }>
  persistentReminder: (origin: string, scope: WorkspaceScope, workId: string) => Promise<string | null>
  ensureWinmanChat: (origin: string, scope: WorkspaceScope) => Promise<WorkspaceWinmanProfile>
  createSession: (origin: string, scope: WorkspaceScope, label: string, requestIdentity: string) => Promise<WorkspaceHistory>
  sessions: (origin: string, scope: WorkspaceScope) => Promise<{ sessions: Array<{ key: string; title?: string }> }>
  history: (origin: string, scope: WorkspaceScope, sessionId: string, afterId?: string) => Promise<WorkspaceHistory>
  nativeContext: (origin: string, scope: WorkspaceScope, sessionId: string) => Promise<WorkspaceMainContext | null>
  bindNativeContext: (origin: string, scope: WorkspaceScope, sessionId: string, originalSessionKey: string) => Promise<WorkspaceMainContext>
  sendNative: (origin: string, scope: WorkspaceScope, sessionId: string, message: string, requestId: string, bindingId: string) => Promise<WorkspaceNativeSnapshot>
  cancelNative: (origin: string, scope: WorkspaceScope, sessionId: string, admissionId: string) => Promise<WorkspaceNativeCancellation>
  nativeHistory: (origin: string, scope: WorkspaceScope, sessionId: string) => Promise<WorkspaceNativeSnapshot>
  interrupt: (origin: string, scope: WorkspaceScope, sessionId: string, requestId: string) => Promise<{ cancel_requested: boolean; request_id?: string }>
  send: (origin: string, scope: WorkspaceScope, sessionId: string, message: string, requestIdentity: string) => Promise<WorkspaceHistory>
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

/** These local Gateway surfaces have no Workspace HTTP adapter. */
export function isUnsupportedWorkspaceStartupRequest(method: string, pathname: string): boolean {
  return (method === 'POST' && pathname === '/api/sessions/owner-backfill') ||
    (method === 'GET' && (pathname === '/api/status' || pathname === '/api/cron/jobs' || pathname === '/api/profiles/active' ||
      /^\/api\/sessions\/[^/]+\/timeline$/.test(pathname)))
}

const startupReadPaths = new Set(['/api/profiles', '/api/sessions', '/api/profiles/sessions', '/api/profiles/sessions/sidebar'])
const pendingStartupReads = new WeakMap<object, Map<string, Promise<unknown>>>()

// The catalog, Main history and Bot binding compete for the same protected
// SessionDB reader. Start each signed operation when its partition is free.
const protectedReadTail = new WeakMap<object, Promise<void>>()

function scheduleProtectedRead<T>(partition: object, read: () => Promise<T>): Promise<T> {
  const prior = protectedReadTail.get(partition)
  const result = prior ? prior.then(read) : read()
  const tail = result.then(() => {}, () => {})
  protectedReadTail.set(partition, tail)
  void tail.then(() => {
    if (protectedReadTail.get(partition) === tail) {protectedReadTail.delete(partition)}
  })
  return result
}

const pendingReadContexts = new WeakMap<object, Promise<WorkspaceIdentity>>()

export function readPendingWorkspaceIdentity(target: ReturnType<typeof workspace>) {
  const { owned, client } = target
  let current = pendingReadContexts.get(owned)

  if (!current) {
    current = client.identity().finally(() => pendingReadContexts.delete(owned))
    pendingReadContexts.set(owned, current)
  }

  return current
}

export async function readScopedWorkspaceDesktop(
  origin: string, scope: WorkspaceScope, path: string, profile: string, resolvedScope?: WorkspaceScope
) {
  const url = new URL(path, 'https://desktop.invalid')
  const queryProfile = url.searchParams.get('profile')
  const sidebarProfile = url.searchParams.get('recents_profile')

  const mainSidebar = url.pathname === '/api/profiles/sessions/sidebar' && profile === 'default' &&
    (!queryProfile || queryProfile === 'default') && (!sidebarProfile || sidebarProfile === 'default')

  // Both entrypoints project the same server-owned Main inventory.
  if (mainSidebar) {
    const inventory = await readScopedWorkspaceDesktop(origin, scope, '/api/sessions', profile, resolvedScope) as { sessions: unknown[] }

    return workspaceSidebarSessions(inventory.sessions)
  }

  const target = workspace(origin)
  const { owned, client } = target

  if (resolvedScope) {
    // The registry dispatcher just obtained this scope from the same
    // origin-owned cookie jar. Only its internal call may supply it.
    if (scope?.userId !== resolvedScope.userId || scope?.workspaceId !== resolvedScope.workspaceId ||
        scope?.organizationId !== resolvedScope.organizationId) {
      throw new Error('The Semantier account or workspace changed. Reconnect before continuing.')
    }
  }

  const sessionDetail = /^\/api\/sessions\/[^/]+(?:\/messages)?$/.test(url.pathname)
  const protectedRead = startupReadPaths.has(url.pathname) || sessionDetail
  const checkBeforeRead = !resolvedScope || protectedRead

  const read = async (initialCheckFinished?: () => void) => {
    try {
      if (checkBeforeRead) {
        assertWorkspaceScope(await readPendingWorkspaceIdentity(target), scope)
      }
    } finally {
      initialCheckFinished?.()
    }

    const result = await workspaceDesktopRead(client, path, profile)
    assertWorkspaceScope(await client.identity(), scope)

    return result
  }

  if (!protectedRead) {return read()}
  if (sessionDetail) {return scheduleProtectedRead(owned, () => read())}

  let pending = pendingStartupReads.get(owned)

  if (!pending) {
    pending = new Map()
    pendingStartupReads.set(owned, pending)
  }

  const key = JSON.stringify([scope?.userId, scope?.organizationId, scope?.workspaceId, profile, path])
  const existing = pending.get(key)

  if (existing) {return existing}

  // Join only while the initial signed check is in flight. A later caller must
  // run its own check before starting a new server read.
  const operation = scheduleProtectedRead(owned, () => read(() => pending.delete(key)))
  pending.set(key, operation)

  return operation
}

export function registerSemantierWorkspaceIpc(onAccountChanged?: (origin: string, authenticated: boolean) => void, beforeAccountChange?: (origin:string)=>Promise<void>, enrollment?:ManagedDeviceEnrollment, personal?:PersonalCapabilitySync): void {
  const personalOwner=()=>{if(!personal){throw new Error('Personal sync is unavailable')}

return personal}

  ipcMain.handle('hermes:workspace:personal-preview',(_event,origin:string,scope:WorkspaceScope,skillId:string)=>personalOwner().preview(origin,scope,skillId))
  ipcMain.handle('hermes:workspace:personal-sources',(_event,origin:string,scope:WorkspaceScope)=>personalOwner().sources(origin,scope))
  ipcMain.handle('hermes:workspace:personal-share',(_event,origin:string,scope:WorkspaceScope,selected:PersonalSnapshot,context:PersonalSnapshot['contexts'][number])=>personalOwner().share(origin,scope,selected,context))
  ipcMain.handle('hermes:workspace:personal-prepare',(_event,origin:string,scope:WorkspaceScope,receipt:PersonalPin)=>personalOwner().prepare(origin,scope,receipt))
  ipcMain.handle('hermes:workspace:personal-admit',(_event,origin:string,scope:WorkspaceScope,receipt:PersonalPin,candidate:PersonalPin)=>personalOwner().admit(origin,scope,receipt,candidate))
  ipcMain.handle('hermes:workspace:personal-install',(_event,origin:string,scope:WorkspaceScope,receipt:PersonalPin,candidate:PersonalPin,replaceExisting?:boolean)=>personalOwner().install(origin,scope,receipt,candidate,replaceExisting))
  // Ephemeral ownership of this renderer's pending HTTP stream, never task state.
  const streams = new Map<string, { controller: AbortController; requestId: string }>()
  const nativeStreams = new Map<string, { controller: AbortController; requestId: string; admission?: Promise<WorkspaceNativeSubmission> }>()

  const streamKey = (senderId: number, origin: string, scope: WorkspaceScope, sessionId: string) =>
    JSON.stringify([senderId, new URL(origin).origin, scope.userId, scope.workspaceId, scope.organizationId, sessionId])

  ipcMain.handle('hermes:workspace:interrupt', async (event, origin: string, scope: WorkspaceScope, sessionId: string, requestId: string) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const key = streamKey(event.sender.id, origin, scope, sessionId)
    const native = nativeStreams.get(key)

    if (native && native.requestId === requestId) {
      native.controller.abort()

      if (!native.admission) {return { cancel_requested: true, request_id: native.requestId }}
      // Await the original admission, never abort/replay an uncertain admission POST.
      const admitted = await native.admission
      const result = await client.cancelNative(sessionId, admitted.receipt.admission_id)
      assertWorkspaceScope(await client.identity(), scope)

      return { ...result, request_id: native.requestId }
    }

    const stream = streams.get(key)

    if (!stream || stream.requestId !== requestId) {return { cancel_requested: false }}
    stream.controller.abort()

    return { cancel_requested: true, request_id: stream.requestId }
  })
  ipcMain.handle('hermes:workspace:managed-device-enroll',async (_event,origin:string,scope:WorkspaceScope,replacesDeviceId?:string)=>{
    const {client}=workspace(origin)
    assertWorkspaceScope(await client.identity(),scope)

    if(!enrollment){throw new Error('Native managed enrollment is unavailable')}

    try{return await enrollment.enroll(origin,scope,replacesDeviceId)}catch(error){
      const e=error as {managedDeviceCode?:string;nativeDeviceCode?:string}
      const code=e.managedDeviceCode||e.nativeDeviceCode

      if(code&&['DEVICE_SEAT_LIMIT','TECHNICAL_DEVICE_LIMIT','NATIVE_DEVICE_KEY_UNAVAILABLE','device_revoked'].includes(code)){throw new Error(code)}
      throw error
    }
  })

  async function withProjectScope<T>(
    origin: string, scope: WorkspaceScope,
    run: (client: ReturnType<typeof createSemantierWorkspaceClient>) => Promise<T>
  ): Promise<T> {
    const { client } = workspace(origin)

    assertWorkspaceScope(await client.identity(), scope)
    const result = await run(client)

    assertWorkspaceScope(await client.identity(), scope)

    return result
  }

  ipcMain.handle('hermes:workspace:project-materials', (_event, origin: string, scope: WorkspaceScope) =>
    withProjectScope(origin, scope, client => client.projectMaterials()))
  ipcMain.handle('hermes:workspace:projects', (_event, origin: string, scope: WorkspaceScope) =>
    withProjectScope(origin, scope, client => client.projects()))
  ipcMain.handle('hermes:workspace:create-project', (_event, origin: string, scope: WorkspaceScope, graph: WorkspaceProjectPin, consent: true) =>
    withProjectScope(origin, scope, client => client.createProject(graph, consent)))
  ipcMain.handle('hermes:workspace:tutor-progress', (_event, origin: string, scope: WorkspaceScope, ref: string, plan: string) =>
    withProjectScope(origin, scope, client => client.tutorProgress(ref, plan)))
  ipcMain.handle('hermes:workspace:continue-project', (_event, origin: string, scope: WorkspaceScope, ref: string) =>
    withProjectScope(origin, scope, client => client.continueProject(ref)))
  ipcMain.handle('hermes:workspace:commit-project', (_event, origin: string, scope: WorkspaceScope, ref: string, decision: WorkspaceProjectCommit) =>
    withProjectScope(origin, scope, client => client.commitProject(ref, decision)))
  ipcMain.handle('hermes:workspace:replay-project', (_event, origin: string, scope: WorkspaceScope, ref: string, head: WorkspaceProjectPin) =>
    withProjectScope(origin, scope, client => client.replayProject(ref, head)))

  ipcMain.handle('hermes:workspace:managed-devices', async (_event, origin: string, scope: WorkspaceScope) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const result = await client.managedDevices()
    assertWorkspaceScope(await client.identity(), scope)

    return result
  })
  ipcMain.handle('hermes:workspace:managed-device-consent', async (_event, origin: string, scope: WorkspaceScope, deviceId: string, permissions: ManagedDevicePermission[]) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const result = await client.managedDeviceConsent(deviceId, permissions)
    assertWorkspaceScope(await client.identity(), scope)

    return result
  })
  ipcMain.handle('hermes:workspace:managed-device-revoke', async (_event, origin: string, scope: WorkspaceScope, deviceId: string) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const result = await client.managedDeviceRevoke(deviceId)
    assertWorkspaceScope(await client.identity(), scope)

    return result
  })
  ipcMain.handle('hermes:workspace:identity', (_event, origin: string) => readPendingWorkspaceIdentity(workspace(origin)))
  ipcMain.handle('hermes:workspace:logout', async (_event, origin: string) => {
    await beforeAccountChange?.(new URL(origin).origin)
    const { owned } = workspace(origin)
    await owned.clearStorageData()
    await owned.clearCache()
    onAccountChanged?.(new URL(origin).origin, false)
  })
  ipcMain.handle('hermes:workspace:login', async (_event, origin: string) => {
    await beforeAccountChange?.(new URL(origin).origin)
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
  ipcMain.handle('hermes:workspace:winman-capability', async (_event, origin: string, scope: WorkspaceScope, enabled?: boolean) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const result = await client.winmanCapability(scope, enabled)
    assertWorkspaceScope(await client.identity(), scope)

    return result
  })
  ipcMain.handle('hermes:workspace:persistent-followups', async (_event, origin: string, scope: WorkspaceScope) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const result = await client.persistentFollowups(scope)
    assertWorkspaceScope(await client.identity(), scope)

    return result
  })
  ipcMain.handle('hermes:workspace:persistent-reminder', async (_event, origin: string, scope: WorkspaceScope, workId: string) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const { items } = await client.persistentFollowups(scope)
    const work = items.find(item => item.id === workId && item.receipt)

    if (!work) { return null }
    const content = await client.persistentReminder(work)
    assertWorkspaceScope(await client.identity(), scope)

    return content
  })
  ipcMain.handle('hermes:workspace:ensure-winman-chat', async (_event, origin: string, scope: WorkspaceScope) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const result = await client.ensureWinmanChat()
    assertWorkspaceScope(await client.identity(), scope)

    return result
  })
  ipcMain.handle('hermes:workspace:create-session', async (_event, origin: string, scope: WorkspaceScope, label: string, requestIdentity: string) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const created = await client.createSession(label, requestIdentity)
    const history = await client.history(created.sessionKey)
    assertWorkspaceScope(await client.identity(), scope)

    return history
  })
  ipcMain.handle('hermes:workspace:sessions', async (_event, origin: string, scope: WorkspaceScope) => {
    // Join the already owned, signed startup inventory path rather than making
    // a second independent context → sessions → context burst during Main open.
    return readScopedWorkspaceDesktop(origin, scope, '/api/sessions', 'default')
  })
  ipcMain.handle('hermes:workspace:history', async (_event, origin: string, scope: WorkspaceScope, sessionId: string, afterId?: string) => {
    const target = workspace(origin)
    const { client } = target
    return scheduleProtectedRead(target.owned, async () => {
      assertWorkspaceScope(await readPendingWorkspaceIdentity(target), scope)
      const result = await client.history(sessionId, ...(afterId === undefined ? [] : [afterId]))
      assertWorkspaceScope(await client.identity(), scope)

      return result
    })
  })
  ipcMain.handle('hermes:workspace:native-context', async (_event, origin: string, scope: WorkspaceScope, sessionId: string) => {
    const target = workspace(origin)
    const { client } = target
    return scheduleProtectedRead(target.owned, async () => {
      assertWorkspaceScope(await readPendingWorkspaceIdentity(target), scope)
      const result = await client.nativeContext(sessionId)
      assertWorkspaceScope(await client.identity(), scope)

      return result
    })
  })
  ipcMain.handle('hermes:workspace:bind-native-context', async (_event, origin: string, scope: WorkspaceScope, sessionId: string, originalSessionKey: string) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const result = await client.bindNativeContext(sessionId, originalSessionKey)
    assertWorkspaceScope(await client.identity(), scope)

    return result
  })
  ipcMain.handle('hermes:workspace:send-native', async (event, origin: string, scope: WorkspaceScope, sessionId: string, message: string, requestId: string, bindingId: string) => {
    const { client } = workspace(origin)
    const key = streamKey(event.sender.id, origin, scope, sessionId)

    if (nativeStreams.has(key)) {throw new Error('A Native admission is already pending in this renderer.')}
    const native: { controller: AbortController; requestId: string; admission?: Promise<WorkspaceNativeSubmission> } = { controller: new AbortController(), requestId }
    nativeStreams.set(key, native)

    try {
      assertWorkspaceScope(await client.identity(), scope)
      native.controller.signal.throwIfAborted()
      native.admission = client.sendNative(sessionId, message, requestId, bindingId)
      const admitted = await native.admission
      const result = await requireNativeCompletion(client, sessionId, admitted.receipt.admission_id, native.controller.signal)
      assertWorkspaceScope(await client.identity(), scope)

      return result
    } finally {
      if (nativeStreams.get(key) === native) {nativeStreams.delete(key)}
    }
  })
  ipcMain.handle('hermes:workspace:cancel-native', async (event, origin: string, scope: WorkspaceScope, sessionId: string, admissionId: string) => {
    const { client } = workspace(origin)
    assertWorkspaceScope(await client.identity(), scope)
    const pending = nativeStreams.get(streamKey(event.sender.id, origin, scope, sessionId))

    if (pending?.admission && (await pending.admission).receipt.admission_id === admissionId) {pending.controller.abort()}
    const result = await client.cancelNative(sessionId, admissionId)
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
  ipcMain.handle('hermes:workspace:send', async (event, origin: string, scope: WorkspaceScope, sessionId: string, message: string, requestIdentity: string) => {
    const { client } = workspace(origin)
    const key = streamKey(event.sender.id, origin, scope, sessionId)

    if (streams.has(key)) {throw new Error('A Main stream is already pending in this renderer.')}
    const stream = { controller: new AbortController(), requestId: requestIdentity }
    streams.set(key, stream)

    try {
      assertWorkspaceScope(await client.identity(), scope)
      stream.controller.signal.throwIfAborted()

      if (/^\/followup(?:\s|$)/.test(message)) {
        await client.explicitWork(sessionId, message, requestIdentity, scope, stream.controller.signal)
      } else {
        const response = await client.send(sessionId, message, requestIdentity, stream.controller.signal)
        // Closing this finite SSE does not authorize replay or cancel durable responsibilities.
        await requireWorkspaceCompletion(response)
      }

      assertWorkspaceScope(await client.identity(), scope)
      const result = await client.history(sessionId)
      assertWorkspaceScope(await client.identity(), scope)

      return result
    } catch (error) {
      if (stream.controller.signal.aborted) {throw new Error(`${WORKSPACE_USER_STOP_ERROR}:${requestIdentity}`)}
      throw error
    } finally {
      if (streams.get(key) === stream) {streams.delete(key)}
    }
  })
}
