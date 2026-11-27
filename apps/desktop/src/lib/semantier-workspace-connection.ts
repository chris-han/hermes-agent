import type { DesktopRegistryConnection } from '@/global'
import { refreshConnectionsRegistry } from '@/store/connections'

export const DEFAULT_SEMANTIER_WORKSPACE_ORIGIN = 'https://cloud.semantier.cn'

export async function authenticateWorkspace(origin: string, isCurrent: () => boolean) {
  const api = window.hermesDesktop.workspace

  if (!api) {
    throw new Error('Semantier Cloud is unavailable.')
  }

  let identity = await api.identity(origin)

  if (!isCurrent()) {
    return null
  }

  if (!identity.authenticated) {
    identity = await api.login(origin)

    if (!isCurrent() || !identity.authenticated) {
      return null
    }
  }

  if (
    !identity.user?.user_id ||
    !identity.workspace?.id ||
    !identity.organization_id ||
    identity.user.workspace_id !== identity.workspace.id
  ) {
    throw new Error('Semantier Cloud did not return a verified account and workspace.')
  }

  return identity
}

export async function ensureWorkspaceConnection(
  origin: string,
  connections: readonly DesktopRegistryConnection[],
  label: string
) {
  const normalized = new URL(origin).origin

  const saved = connections.find(
    connection =>
      connection.transport === 'semantier-workspace-http' &&
      connection.url &&
      new URL(connection.url).origin === normalized
  )

  if (saved) {
    return saved
  }

  const { connection } = await window.hermesDesktop.connections.save({
    kind: 'cloud',
    label: connections.some(existing => existing.label === label) ? `${label} (${new URL(normalized).host})` : label,
    url: normalized,
    transport: 'semantier-workspace-http'
  })

  await refreshConnectionsRegistry()

  return connection
}
