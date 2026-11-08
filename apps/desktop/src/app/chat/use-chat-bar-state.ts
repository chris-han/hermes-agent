import type { ModelOptionsResult } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import type * as React from 'react'
import { useMemo } from 'react'

import type { HermesGateway } from '@/hermes'
import { quickModelOptions } from '@/lib/chat-runtime'
import { modelDiscoveryOwnerKey } from '@/lib/model-discovery-scope'
import { currentModelCapabilities, modelOptionsQueryKey, requestDiscoveredModelOptions } from '@/lib/model-options'
import { requestGatewayForAgent } from '@/store/gateway'
import { $connection } from '@/store/session'

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
  const ownerKey = modelDiscoveryOwnerKey(modelOptionsOwnerConnectionId, connection)
  const profile = modelOptionsProfile || activeGatewayProfile
  const modelOptionsQuery = useQuery<ModelOptionsResult>({
    queryKey: modelOptionsQueryKey(profile, activeSessionId, ownerKey),
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

  const quickModels = useMemo(
    () => quickModelOptions(modelOptionsQuery.data, currentProvider, currentModel),
    [currentModel, currentProvider, modelOptionsQuery.data]
  )

  const supportsReasoning = currentModelCapabilities(modelOptionsQuery.data, currentProvider, currentModel)?.reasoning

  return useMemo<ChatBarState>(
    () => ({
      model: {
        model: currentModel,
        provider: currentProvider,
        canSwitch: gatewayOpen,
        loading: !gatewayOpen || (!currentModel && !currentProvider),
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
      currentModel,
      currentProvider,
      gatewayOpen,
      modelMenuContent,
      quickModels,
      reasoningMenuContent,
      supportsReasoning
    ]
  )
}
