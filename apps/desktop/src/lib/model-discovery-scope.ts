/** A non-secret, stable namespace for one desktop connection. */
export type DiscoveryConnection = {
  baseUrl: string
  connectionId?: string
  mode?: 'local' | 'remote'
  remoteIdentity?: string
  transport?: 'semantier-workspace-http'
  workspaceScope?: {userId:string; workspaceId:string; organizationId:string}
}

const remoteOwnerKeys = new WeakMap<DiscoveryConnection, string>()
let nextRemoteOwner = 0

/** Renderer-cache key: connection changes invalidate even same-named profiles. */
export function modelDiscoveryOwnerKey(
  ownerConnectionId: null | string | undefined,
  connection: DiscoveryConnection | null | undefined
): string | undefined {
  const id = ownerConnectionId?.trim() || connection?.connectionId?.trim()

  if (connection?.transport === 'semantier-workspace-http' &&
      (!ownerConnectionId || ownerConnectionId === connection.connectionId)) {
    const s = connection.workspaceScope
    let origin = 'unavailable'

    try {origin = new URL(connection.baseUrl).origin} catch { /* An incomplete connection has no reusable source. */ }

    return `${id || 'workspace'}:${JSON.stringify(s ? [origin,s.userId,s.workspaceId,s.organizationId] : [origin,'signed-out'])}`
  }

  if (id) {return id}

  if (connection?.mode === 'local') {return 'local'}

  if (connection?.mode !== 'remote') {return undefined}
  let key = remoteOwnerKeys.get(connection)

  if (!key) {
    key = `legacy-remote-owner:${++nextRemoteOwner}`
    remoteOwnerKeys.set(connection, key)
  }

  return key
}

export async function modelDiscoveryScope(
  ownerConnectionId: null | string | undefined,
  connection: DiscoveryConnection | null | undefined
): Promise<string> {
  const id = ownerConnectionId?.trim() || connection?.connectionId?.trim()

  if (connection?.transport === 'semantier-workspace-http' &&
      (!ownerConnectionId || ownerConnectionId === connection.connectionId)) {
    if (!connection.workspaceScope) {throw new Error('Sign in to the Workspace before model discovery.')}

    return modelDiscoveryOwnerKey(ownerConnectionId, connection)!
  }

  if (id) {return id}

  if (connection?.mode === 'local') {return 'local'}

  if (connection?.mode !== 'remote' || !connection.baseUrl || !globalThis.crypto?.subtle) {
    throw new Error('The model discovery connection is not ready.')
  }

  const endpoint = new URL(connection.baseUrl)
  // Query strings and URL userinfo can carry credentials; neither is sent as
  // the namespace. The authenticated server and provider principal remain
  // separate components of the cache key.
  const source = JSON.stringify([endpoint.origin, endpoint.pathname, connection.remoteIdentity ?? ''])
  const digest = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(source))

  return `legacy-remote:${Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')}`
}
