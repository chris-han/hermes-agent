import type * as React from 'react'
import { useCallback, useMemo, useRef, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { useInventoryCopy } from '@/i18n/capability-inventory'
import { queryClient } from '@/lib/query-client'
import { invalidateSlashCompletions } from '@/lib/slash-completion-cache'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $gateway } from '@/store/gateway'
import { OFFICIAL_SKILLS_KEY } from '@/store/hub-actions'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { PanelEmpty } from '../overlays/panel'
import { PageSearchShell } from '../page-search-shell'
import type { SetStatusbarItemGroup } from '../shell/statusbar-controls'

import { ConnectorsTab } from './connectors/connectors-tab'
import { CapabilityInventory } from './inventory'
import { PluginsTab } from './plugins/plugins-tab'
import { CapabilityScopeSelector, useCapabilityScope } from './scope-selector'
import { type CapabilitySearchHit, CapabilitySearchResults } from './search-results'
import { MarketplaceTab } from './skills/marketplace-tab'
import { SKILLS_QUERY_KEY, skillSearchTerms, useSkillsQuery } from './skills/skills-data'
import { SkillsTab } from './skills/skills-tab'
import { refreshToolCalls } from './toolsets/tool-calls'
import { TOOLSETS_QUERY_KEY, toolsetSearchTerms, useToolsetsQuery, visibleToolsetCount } from './toolsets/toolsets-data'
import { ToolsetsTab } from './toolsets/toolsets-tab'

// Legacy hub links land on the native Marketplace destination.
const CAPABILITY_MODES = ['skills', 'toolsets', 'plugins', 'connectors', 'marketplace', 'hub'] as const

type CapabilityMode = (typeof CAPABILITY_MODES)[number]

interface CapabilitiesViewProps extends React.ComponentProps<'section'> {
  setStatusbarItemGroup?: SetStatusbarItemGroup
  /** Embedded mode (plugin dialogs — e.g. Bot Mode's Advanced section): tab
   *  state lives in local React state instead of the route's `?tab=` param,
   *  so an embedding dialog never fights the page router. */
  embedded?: boolean
  /** Pin the WHOLE view to one profile: the scope selector is hidden and
   *  every tab reads/writes THAT profile. This is the plugin door — Bot Mode
   *  renders the real Capabilities surface pinned to a bot. */
  fixedProfile?: string
  /** Pin the view to a REGISTERED gateway connection alongside `fixedProfile`:
   *  every read/write routes to that machine's backend instead of the active
   *  gateway. `''`/`'local'` mean the local pool. This is Bot Mode's
   *  remote-target door — a bot living on another registered gateway gets the
   *  live surface pointed at ITS backend. Ignored without `fixedProfile`. */
  fixedConnection?: string
}

/** The Capabilities page SHELL: tab selection, the search header, the profile /
 *  connection scope, the refresh hotkey, and the dispatch to one tab component
 *  per tab. Each tab owns its list, detail pane and writes; the two installed
 *  lists are fetched here because the tab pills count them for the tab the user
 *  is NOT on. */
export function CapabilitiesView({
  embedded = false,
  fixedConnection,
  fixedProfile,
  setStatusbarItemGroup: _setStatusbarItemGroup,
  ...props
}: CapabilitiesViewProps) {
  const { t } = useI18n()
  const copy = useInventoryCopy()
  // Both hooks run unconditionally (rules of hooks); embedded picks the local
  // one so tab clicks inside a dialog don't rewrite the page URL.
  const routeTab = useRouteEnumParam('tab', CAPABILITY_MODES, 'skills')
  const localTab = useState<CapabilityMode>('skills')
  const [mode, setMode] = embedded ? localTab : routeTab
  const gateway = useStoreSelector($gateway, g => (mode === 'connectors' ? g : null))

  const [query, setQuery] = useState('')
  const [searchTarget, setSearchTarget] = useState<CapabilitySearchHit | null>(null)
  const searchRoot = useRef<HTMLElement>(null)
  const tabQuery = searchTarget?.name ?? ''
  const searching = Boolean(query.trim()) && !searchTarget

  const scope = useCapabilityScope({ fixedConnection, fixedProfile })

  // The two installed lists the tab pills count. They are fetched here, as a
  // pair, because the counts stay live for the tab the user is NOT on.
  const { data: skills, isError: skillsFailed, error: skillsError } = useSkillsQuery(scope.profile)
  const { data: toolsets, isError: toolsetsFailed } = useToolsetsQuery(scope.profile)

  const refreshCapabilities = useCallback(async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: SKILLS_QUERY_KEY }),
      queryClient.invalidateQueries({ queryKey: TOOLSETS_QUERY_KEY }),
      queryClient.invalidateQueries({ queryKey: OFFICIAL_SKILLS_KEY })
    ])

    invalidateSlashCompletions()

    void refreshToolCalls(scope.profile)
  }, [scope.profile])

  useRefreshHotkey(refreshCapabilities)

  // Rotating placeholder nudges from the user's own data — teach that search
  // understands categories and tool names, not just titles.
  const searchHints = useMemo(() => {
    if (mode === 'skills' && skills?.length) {
      return skillSearchTerms(skills).map(term => t.common.tryHint(term))
    }

    if (mode === 'toolsets' && toolsets?.length) {
      return toolsetSearchTerms(toolsets).map(term => t.common.tryHint(term))
    }

    return undefined
  }, [mode, skills, t, toolsets])

  // MCP and Plugins load independently of the installed Skills/Tools lists.
  const gated = mode === 'toolsets' || mode === 'skills'
  const pending = gated && !(skills && toolsets)

  const loadGate = !pending ? null : skillsFailed || toolsetsFailed ? (
    <PanelEmpty
      action={
        <Button onClick={() => void refreshCapabilities()} size="sm">
          {t.skills.refresh}
        </Button>
      }
      description={skillsError instanceof Error ? skillsError.message : undefined}
      icon="error"
      title={t.skills.skillsLoadFailed}
    />
  ) : (
    <PageLoader label={t.skills.loading} />
  )

  const tabContent = {
    marketplace: () => (
      <MarketplaceTab initialSelected={searchTarget?.id} key={scope.key} profile={scope.profile} query={tabQuery} />
    ),
    hub: () => (
      <MarketplaceTab initialSelected={searchTarget?.id} key={scope.key} profile={scope.profile} query={tabQuery} />
    ),
    // The gateway instance backs ONLY the live `reload.mcp` RPC, and it is the
    // ACTIVE gateway's socket — for a scope pinned to a different backend that
    // (config edits still apply on that backend's next session).
    connectors: () => (
      <ConnectorsTab
        gateway={scope.crossBackend ? null : gateway}
        initialSelected={searchTarget?.id}
        key={`connectors-${scope.key}`}
        profile={scope.profile}
        query={tabQuery}
      />
    ),
    // Agent plugins for the scoped profile (selector in the section header),
    // app-level desktop plugins, and the docs catalog picker underneath.
    plugins: () => (
      <PluginsTab
        initialQuery={tabQuery}
        key={`plugins-${scope.key}`}
        profile={scope.profile}
        scopeLabel={scope.label}
        scopeSelector={scope.options.length > 1 ? <CapabilityScopeSelector compact scope={scope} /> : undefined}
      />
    ),
    skills: () => (
      <SkillsTab
        initialSelected={searchTarget?.id}
        key={`skills-${scope.key}`}
        onRefresh={() => void refreshCapabilities()}
        profile={scope.profile}
        query={tabQuery}
        skills={skills ?? []}
      />
    ),
    toolsets: () => (
      <ToolsetsTab
        initialSelected={searchTarget?.id}
        key={`toolsets-${scope.key}`}
        profile={scope.profile}
        query={tabQuery}
        toolsets={toolsets ?? []}
      />
    )
  } satisfies Record<CapabilityMode, () => React.ReactNode>

  return (
    <PageSearchShell
      {...props}
      activeTab={mode === 'hub' ? 'marketplace' : mode}
      onSearchChange={value => {
        setQuery(value)
        setSearchTarget(null)
      }}
      onTabChange={id => {
        setMode(id as CapabilityMode)
        setQuery('')
        setSearchTarget(null)
      }}
      ref={searchRoot}
      searchHints={query ? undefined : searchHints}
      searchPlaceholder={copy.searchAll}
      searchValue={query}
      tabs={[
        { id: 'skills', label: t.skills.tabSkills, meta: skills?.length ?? null },
        { id: 'toolsets', label: t.skills.tabToolsets, meta: toolsets ? visibleToolsetCount(toolsets) : null },
        { id: 'plugins', label: t.skills.tabPlugins },
        { id: 'connectors', label: t.connectorsPage.title },
        { id: 'marketplace', label: copy.marketplace }
      ]}
    >
      <div className="flex h-full flex-col">
        {(searching || mode !== 'plugins') && <CapabilityScopeSelector scope={scope} />}
        {searchTarget && (
          <Button
            className="self-start"
            onClick={() => {
              setSearchTarget(null)
              requestAnimationFrame(() => searchRoot.current?.querySelector<HTMLInputElement>('input')?.focus())
            }}
            size="xs"
            variant="text"
          >
            {copy.backToSearch}
          </Button>
        )}
        <div className="flex min-h-0 flex-1 flex-col">
          {searching ? (
            <CapabilitySearchResults
              key={scope.key}
              onOpen={hit => {
                setSearchTarget(hit)
                setMode(hit.tab)
              }}
              onRetry={() => void refreshCapabilities()}
              profile={scope.profile}
              query={query}
              skills={skills}
              skillsFailed={skillsFailed}
              toolsets={toolsets}
              toolsetsFailed={toolsetsFailed}
            />
          ) : (
            <CapabilityInventory key={`${mode}:${scope.key}`} tab={mode === 'hub' ? 'marketplace' : mode}>
              {loadGate ?? tabContent[mode]()}
            </CapabilityInventory>
          )}
        </div>
      </div>
    </PageSearchShell>
  )
}

// Feature-detection flag for plugins (Bot Mode): TRUE means this build's
// CapabilitiesView routes `fixedConnection` to the pinned connection's backend.
// Older builds export CapabilitiesView WITHOUT the prop — passing it there would
// silently read/write the ACTIVE gateway under the remote bot's profile name,
// which is exactly the wrong-machine bug the prop exists to prevent. A static
// property is probe-able without rendering.
CapabilitiesView.supportsFixedConnection = true as const
