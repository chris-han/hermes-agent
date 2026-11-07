import { atom } from 'nanostores'

/** In-flight startup work, scoped to the same connection/profile as catalog queries. */
export const $refreshingModelSources = atom<ReadonlySet<string>>(new Set())
export const modelDiscoveryRefreshKey = (owner: string, profile: string): string => JSON.stringify([owner, profile])

export function setModelDiscoveryRefreshing(owner: string, profile: string, refreshing: boolean): void {
  const next = new Set($refreshingModelSources.get())
  const key = modelDiscoveryRefreshKey(owner, profile)
  if (refreshing) {next.add(key)}
  else {next.delete(key)}
  $refreshingModelSources.set(next)
}
