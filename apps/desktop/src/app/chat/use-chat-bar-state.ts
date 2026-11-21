import type { ModelOptionsResult } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import type * as React from 'react'
import { useEffect, useMemo, useState } from 'react'

import type { HermesGateway } from '@/hermes'
import { type AccountModelDefault, currentAccountModelDefault } from '@/lib/account-model-default'
import { quickModelOptions } from '@/lib/chat-runtime'
import { useModelAccountScope, visibleAccountModelOptions } from '@/lib/model-account-scope'
import { modelDiscoveryOwnerKey } from '@/lib/model-discovery-scope'
import { currentModelCapabilities, modelOptionsQueryKey, requestDiscoveredModelOptions } from '@/lib/model-options'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { requestGatewayForAgent } from '@/store/gateway'
import { $connection, $currentModelSource } from '@/store/session'

import type { ChatBarState } from './composer/types'

interface ChatBarStateOptions {
  activeGatewayProfile: string
  activeSessionId: null | string
  contextSuggestions: ChatBarState['tools']['suggestions']
  currentModel: string
  currentProvider: string
  gateway: HermesGateway | null
  gatewayOpen: boolean
  modelMenuContent?: React.ReactNode
  modelOptionsOwnerConnectionId?: string
  modelOptionsProfile?: string
  reasoningMenuContent?: React.ReactNode
  requestModelOptionsForOwner?: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}

// The composer's model / tools / voice state, backed by the model-options query
// for the active session's profile.
export function useChatBarState({
  activeGatewayProfile,
  activeSessionId,
  contextSuggestions,
  currentModel,
  currentProvider,
  gateway,
  gatewayOpen,
  modelMenuContent,
  modelOptionsOwnerConnectionId,
  modelOptionsProfile,
  reasoningMenuContent,
  requestModelOptionsForOwner
}: ChatBarStateOptions): ChatBarState {
  const connection = useStore($connection)
  const registry = useStore($connectionsRegistry)
  const modelSource = useStore($currentModelSource)
  const account = useModelAccountScope(connection, modelOptionsOwnerConnectionId)
  const ownerKey = modelDiscoveryOwnerKey(modelOptionsOwnerConnectionId, connection)
  const profile = modelOptionsProfile || activeGatewayProfile
  const [candidate, setCandidate] = useState<AccountModelDefault | null>(null)
  useEffect(() => {
    if (connection?.mode !== 'local' || profile !== 'default' || activeSessionId ||
        (modelOptionsOwnerConnectionId && modelOptionsOwnerConnectionId !== connection.connectionId)) {setCandidate(null); return}
    const api = window.hermesDesktop?.workspace
    let live = true
    let revision = 0
    const refresh = () => {
      const current = ++revision
      void api?.modelDefault?.().then(value => {if (live && current === revision) {setCandidate(value)}})
        .catch(() => {if (live && current === revision) {setCandidate(null)}})
    }
    const unsubStatus = api?.onModelStatus?.(status => {
      if (status.state === 'ready') {refresh()} else {revision++; setCandidate(null)}
    })
    const unsubAccount = api?.onAccountChanged?.(() => {revision++; setCandidate(null)})
    refresh()
    return () => {live = false; revision++; unsubStatus?.(); unsubAccount?.()}
  }, [connection, profile, activeSessionId, modelOptionsOwnerConnectionId])
  const accountDefault = account.allowCloud && !activeSessionId && modelSource !== 'manual'
    ? currentAccountModelDefault(candidate, registry) : null
  const displayModel = accountDefault || currentModel
  const displayProvider = accountDefault ? 'semantier-cloud' : currentProvider

  const modelOptionsQuery = useQuery<ModelOptionsResult>({
    queryKey: [...modelOptionsQueryKey(profile, activeSessionId, ownerKey), ...account.key],
    queryFn: () =>
      requestDiscoveredModelOptions({
        connection,
        gateway: gateway || undefined,
        ownerConnectionId: modelOptionsOwnerConnectionId,
        profile,
        request: requestModelOptionsForOwner ?? (<T,>(method: string, params?: Record<string, unknown>) =>
          requestGatewayForAgent<T>(modelOptionsOwnerConnectionId ?? connection?.connectionId ?? null, profile, method, params)),
        sessionId: activeSessionId
      }),
    enabled: gatewayOpen && !!ownerKey
  })

  const options = useMemo(() => visibleAccountModelOptions(modelOptionsQuery.data, account.allowCloud), [modelOptionsQuery.data, account.allowCloud])
  const quickModels = useMemo(() => quickModelOptions(options, displayProvider, displayModel), [displayModel, displayProvider, options])

  const supportsReasoning = currentModelCapabilities(options, displayProvider, displayModel)?.reasoning

  return useMemo<ChatBarState>(
    () => ({
      model: {
        model: displayModel,
        provider: displayProvider,
        canSwitch: gatewayOpen,
        loading: !gatewayOpen || (!displayModel && !displayProvider),
        modelMenuContent,
        quickModels,
        reasoningMenuContent,
        supportsReasoning
      },
      tools: {
        enabled: true,
        label: 'Add context',
        suggestions: contextSuggestions
      },
      voice: {
        enabled: true,
        active: false
      }
    }),
    [
      contextSuggestions,
      displayModel,
      displayProvider,
      gatewayOpen,
      modelMenuContent,
      quickModels,
      reasoningMenuContent,
      supportsReasoning
    ]
  )
}
