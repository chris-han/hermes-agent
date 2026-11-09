import type { ModelOptionsResult } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

import { testProviderModel } from '@/api/models'
import { useSessionView } from '@/app/chat/session-view'
import { Codicon } from '@/components/ui/codicon'
import { DropdownMenuItem, dropdownMenuRow } from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { providerJourneyCopy, providerTestStatus } from '@/i18n/model-service'
import { modelDiscoveryOwnerKey } from '@/lib/model-discovery-scope'
import { modelOptionsQueryKey, requestDiscoveredModelOptions } from '@/lib/model-options'
import { cn } from '@/lib/utils'
import { confirm } from '@/store/confirm'
import { notify, notifyError } from '@/store/notifications'
import { $connection, $currentModelSource } from '@/store/session'

import { ModelCatalogMenu } from './model-catalog-menu'
import { type ModelMenuHostProps, useModelMenuController } from './use-model-menu-controller'

export { ModelMenuCloseContext } from './model-catalog-menu'
export type { ModelSelection } from './use-model-menu-controller'

interface ModelMenuPanelProps extends ModelMenuHostProps {
  /** Drop the sticky composer pick so new chats follow Settings → Model. */
  onFollowDefaultModel?: () => void
}

/**
 * The composer's model menu: `ModelCatalogMenu` (the shared renderer) plus the
 * controller that gives a selection its meaning HERE (`useModelMenuController`).
 */
export function ModelMenuPanel({ onFollowDefaultModel, ...props }: ModelMenuPanelProps) {
  const { gateway, ownerConnectionId, profile = 'default', requestGateway } = props
  const { t, locale } = useI18n()
  const copy = t.shell.modelMenu
  const [refreshing, setRefreshing] = useState(false)
  const queryClient = useQueryClient()
  const view = useSessionView()
  const modelSource = useStore($currentModelSource)
  const connection = useStore($connection)
  const ownerKey = modelDiscoveryOwnerKey(ownerConnectionId, connection)
  const { activeSessionId, controller } = useModelMenuController(props)
  const [testingOwner, setTestingOwner] = useState<string | null>(null)
  const testOwner = `${ownerKey}:${profile}:${controller.current.provider}:${controller.current.model}`
  const ownerRevision = useRef({ owner: testOwner, revision: 0 })

  if (ownerRevision.current.owner !== testOwner) {
    ownerRevision.current = { owner: testOwner, revision: ownerRevision.current.revision + 1 }
  }

  const testToken = JSON.stringify([testOwner, ownerRevision.current.revision])
  const testing = testingOwner === testToken
  const testOwnerRef = useRef(testToken)
  testOwnerRef.current = testToken
  useEffect(() => () => { testOwnerRef.current = '' }, [])
  const journeyCopy = providerJourneyCopy(locale)
  const testLabel = journeyCopy.test

  const testModel = async () => {
    const captured = testToken
    const { provider, model } = controller.current

    const description = journeyCopy.cost

    if (!(await confirm({ title: `${testLabel}: ${provider} / ${model}`, description, confirmLabel: testLabel }))) { return }

    if (testOwnerRef.current !== captured) { return }
    setTestingOwner(captured)

    try {
      const result = await testProviderModel(provider, model, { connectionId: ownerConnectionId, profile })

      if (testOwnerRef.current !== captured) { return }
      notify({ kind: result.ok ? 'success' : 'error', title: testLabel,
        message: `${provider} / ${model}: ${providerTestStatus(locale, result.status)}` })
    } catch (error) {
      if (testOwnerRef.current === captured) { notifyError(error, testLabel) }
    } finally {
      if (testOwnerRef.current === captured) { setTestingOwner(null) }
    }
  }

  // Same condition as the pill's pin dot: a draft whose next session.create
  // ships the manual pick instead of the Settings default (#107410).
  const pinnedDraft = view.kind === 'primary' && !activeSessionId && modelSource === 'manual'

  // Explicit "Refresh Models": re-fetch the catalog with refresh:true so the
  // backend busts its 1h provider-model disk cache and re-pulls each provider's
  // live list. Fixes live-only models (e.g. OpenCode Zen free tier) vanishing
  // when the cache expires and falls back to the curated static list.
  const refreshModels = async () => {
    if (refreshing) {
      return
    }

    setRefreshing(true)

    try {
      const queryKey = modelOptionsQueryKey(profile, activeSessionId, ownerKey)

      const next = await requestDiscoveredModelOptions({
        connection,
        gateway,
        ownerConnectionId,
        profile,
        refresh: true,
        request: requestGateway,
        sessionId: activeSessionId
      })

      // The refreshed catalog is a hint list, never a reason to move the pick:
      // a custom slug the row lacks is still what the user selected.
      queryClient.setQueryData<ModelOptionsResult>(queryKey, next)
      void queryClient.invalidateQueries({
        predicate: query => query.queryKey[0] === 'model-options' && query.queryKey[1] === profile &&
          query.queryKey[2] !== (activeSessionId || 'global') &&
          query.queryKey[3] === 'owner' && query.queryKey[4] === ownerKey
      })
    } catch {
      // Reload this source's snapshot so a failed refresh can surface its stale marker.
      void queryClient.invalidateQueries({
        predicate: query => query.queryKey[0] === 'model-options' && query.queryKey[1] === profile &&
          query.queryKey[3] === 'owner' && query.queryKey[4] === ownerKey
      })
    } finally {
      setRefreshing(false)
    }
  }

  return (
    <ModelCatalogMenu
      controller={controller}
      footer={
        <>
          <DropdownMenuItem
            disabled={testing || !controller.current.provider || !controller.current.model || connection?.transport === 'semantier-workspace-http'}
            onSelect={event => { event.preventDefault(); void testModel() }}
          >
            <Codicon name="beaker" size="0.75rem" />
            {testLabel}
          </DropdownMenuItem>
          {pinnedDraft && onFollowDefaultModel && (
            <DropdownMenuItem
              className={cn(dropdownMenuRow, 'text-(--ui-text-tertiary)')}
              onSelect={onFollowDefaultModel}
            >
              <Codicon name="discard" size="0.75rem" />
              {copy.followDefault}
            </DropdownMenuItem>
          )}
          <DropdownMenuItem
            className={cn(dropdownMenuRow, 'text-(--ui-text-tertiary)')}
            disabled={refreshing}
            onSelect={event => {
              event.preventDefault()
              void refreshModels()
            }}
          >
            <Codicon name="sync" size="0.75rem" spinning={refreshing} />
            {copy.refreshModels}
          </DropdownMenuItem>
        </>
      }
      gateway={gateway}
      includeMoa
      ownerConnectionId={ownerConnectionId}
      profile={profile}
      request={requestGateway}
      sessionId={activeSessionId}
    />
  )
}
