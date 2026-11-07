import { type DiscoveryConnection, modelDiscoveryOwnerKey, modelDiscoveryScope } from '@/lib/model-discovery-scope'
import { requestModelOptions } from '@/lib/model-options'
import { queryClient } from '@/lib/query-client'
import { activeGatewayConnectionId, requestGatewayForAgent } from '@/store/gateway'
import { setModelDiscoveryRefreshing } from '@/store/model-discovery'
import { $connection } from '@/store/session'

// One bounded provider refresh per source/profile per renderer launch. Reconnects
// do not repeat it; the explicit refresh action remains available after failure.
const started = new Set<string>()

export async function refreshModelsOnConnectionReady(
  connectionId: null | string,
  profile: string,
  connection?: DiscoveryConnection | null
): Promise<void> {
  let source: string
  try {
    source = await modelDiscoveryScope(connectionId, connection)
  } catch {
    return
  }
  const profileKey = profile.trim() || 'default'
  const ownerKey = modelDiscoveryOwnerKey(connectionId, connection)
  const launchKey = JSON.stringify([source, profileKey])
  if (started.has(launchKey)) {return}
  started.add(launchKey)
  if (ownerKey) {setModelDiscoveryRefreshing(ownerKey, profileKey, true)}

  try {
    await requestModelOptions({
      profile: profileKey,
      queryOnly: true,
      queryScope: source,
      refresh: true,
      request: <T>(method: string, params?: Record<string, unknown>) =>
        requestGatewayForAgent<T>(connectionId, profileKey, method, params)
    })
    const activeSource = await modelDiscoveryScope(activeGatewayConnectionId(), $connection.get()).catch(() => null)
    await queryClient.invalidateQueries({
      predicate: query => {
        const key = query.queryKey
        if (key[0] !== 'model-options' || key[1] !== profileKey) {return false}
        if (key[3] === 'owner') {return key[4] === ownerKey}
        return activeSource === source
      }
    })
  } catch {
    // Startup must remain usable offline; a manual refresh can retry later.
  } finally {
    if (ownerKey) {setModelDiscoveryRefreshing(ownerKey, profileKey, false)}
  }
}
