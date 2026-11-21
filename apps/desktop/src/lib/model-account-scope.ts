import type { ModelOptionsResult } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { useEffect } from 'react'

import type { DiscoveryConnection } from '@/lib/model-discovery-scope'
import { $connectionsRegistry } from '@/store/connection-registry-state'

type ScopeState = { ready: boolean; scope: string; epoch: number }
const accountScope = atom<ScopeState>({ ready: false, scope: 'unresolved', epoch: 0 })
let users = 0
let revision = 0
let unsubscribe: (() => void) | undefined

function refreshAccountScope() {
  const current = ++revision
  const previous = accountScope.get()
  accountScope.set({ ...previous, ready: false, epoch: previous.epoch + 1 })
  const bridge = window.hermesDesktop
  const list = bridge?.connections?.list

  if (list) {
    void list().then(registry => {
      if (current !== revision) {return}
      $connectionsRegistry.set(registry)
      accountScope.set({ ready: true, scope: JSON.stringify(registry.accountContext ?? null), epoch: accountScope.get().epoch })
    }).catch(() => {})
  } else if (!bridge?.workspace?.onAccountChanged) {
    accountScope.set({ ready: true, scope: 'legacy-standalone', epoch: accountScope.get().epoch })
  }
}

/** Renderer cache partition only; authorization remains at the credential owner. */
export function useModelAccountScope(connection: DiscoveryConnection | null | undefined, ownerConnectionId?: null | string) {
  const local = connection?.mode === 'local' && (!ownerConnectionId || !connection.connectionId || ownerConnectionId === connection.connectionId)
  const state = useStore(accountScope)
  useEffect(() => {
    if (!local) {return}
    users++

    if (users === 1) {
      unsubscribe = window.hermesDesktop?.workspace?.onAccountChanged?.(refreshAccountScope)
      refreshAccountScope()
    }

    return () => {
      users--

      if (!users) {
        revision++
        unsubscribe?.()
        unsubscribe = undefined
        const previous = accountScope.get()
        accountScope.set({ ...previous, ready: false })
      }
    }
  }, [local])

  return { key: local ? ['account', state.scope, state.epoch] : [], allowCloud: !local || state.ready }
}

/** Do not show a previously observed cloud catalog while the account is unknown. */
export function visibleAccountModelOptions(data: ModelOptionsResult | undefined, allowCloud: boolean): ModelOptionsResult | undefined {
  if (!data || allowCloud) {return data}

  return { ...data, ...(data.provider === 'semantier-cloud' ? { provider: '', model: '' } : {}),
    providers: (data.providers ?? []).filter(provider => provider.slug !== 'semantier-cloud') }
}
