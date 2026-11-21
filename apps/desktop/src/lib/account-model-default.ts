import type { DesktopConnectionsRegistry } from '@/global'

export type AccountModelDefault = { model: string; account: { origin: string; scope: { userId: string; workspaceId: string; organizationId: string } } }

/** Renderer check is a restriction on the main-process result, never a model grant. */
export function currentAccountModelDefault(candidate: AccountModelDefault | null | undefined, registry: DesktopConnectionsRegistry | null): string | null {
  const selected = registry?.accountContext
  if (!candidate || !selected || candidate.account.origin !== selected.origin ||
      candidate.account.scope.userId !== selected.scope.userId || candidate.account.scope.workspaceId !== selected.scope.workspaceId ||
      candidate.account.scope.organizationId !== selected.scope.organizationId || !candidate.model) {return null}
  return candidate.model
}
