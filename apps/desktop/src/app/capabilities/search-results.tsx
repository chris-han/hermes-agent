import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { $pluginRecords } from '@/contrib/plugins-store'
import { type ProfileScope, profileScopeKey, searchSkillsHub } from '@/hermes'
import { useI18n } from '@/i18n'
import { useInventoryCopy } from '@/i18n/capability-inventory'
import { isDesktopToolsetVisible } from '@/lib/desktop-toolsets'
import type { PluginCatalogEntry } from '@/lib/plugin-catalog'
import { type AgentPluginRow, isDesktopRelevantPlugin } from '@/store/agent-plugins'
import { activeGatewayConnectionId, requestGatewayForAgent } from '@/store/gateway'
import { normalizeProfileKey } from '@/store/profile'
import type { SkillInfo, ToolsetInfo } from '@/types/hermes'

import { toolNames, toolsetDisplayLabel } from '../settings/helpers'

import { joinBundledEntries, joinLocalServers } from './connectors/data/join'
import { useHostedConnectors, usePluginServers } from './connectors/data/queries'
import { cardKey, deriveCards } from './connectors/derive'
import { useMcpServers } from './mcp/use-mcp-servers'
import { mergePluginPackages } from './plugins/plugin-packages'

export type SearchTab = 'skills' | 'toolsets' | 'plugins' | 'connectors' | 'marketplace'
export interface CapabilitySearchHit {
  id: string
  name: string
  description: string
  tab: SearchTab
}

/** Read-only cross-tab search; mutation and inspection remain in each destination. */
export function CapabilitySearchResults({
  query,
  profile,
  skills,
  toolsets,
  skillsFailed,
  toolsetsFailed,
  onOpen,
  onRetry
}: {
  query: string
  profile: ProfileScope
  skills: SkillInfo[] | undefined
  toolsets: ToolsetInfo[] | undefined
  skillsFailed: boolean
  toolsetsFailed: boolean
  onOpen: (hit: CapabilitySearchHit) => void
  onRetry?: () => void
}) {
  const { t } = useI18n()
  const c = useInventoryCopy()
  const [search, setSearch] = useState(query)
  useEffect(() => {
    const timer = setTimeout(() => setSearch(query), 250)

    return () => clearTimeout(timer)
  }, [query])
  const scopeKey = profileScopeKey(profile)

  const connectionId =
    typeof profile === 'object' && profile?.connectionId
      ? profile.connectionId
      : (activeGatewayConnectionId() ?? 'local')

  const profileName = normalizeProfileKey(profile && typeof profile === 'object' ? profile.profile : profile)
  const desktop = useStore($pluginRecords)

  const plugins = useQuery({
    queryKey: ['capability-search-plugins', scopeKey],
    queryFn: () =>
      requestGatewayForAgent<{ plugins?: AgentPluginRow[] }>(connectionId, profileName, 'plugins.manage', {
        action: 'list',
        profile: profileName
      }),
    retry: false
  })

  const catalog = useQuery({
    queryKey: ['native-plugin-catalog', connectionId, profileName],
    queryFn: () =>
      requestGatewayForAgent<{ available: boolean; entries: (PluginCatalogEntry & { description: string })[] }>(
        connectionId,
        profileName,
        'plugins.catalog.list',
        { profile: profileName },
        20_000
      ),
    retry: false
  })

  const marketplace = useQuery({
    queryKey: ['skill-hub-search', scopeKey, search, 'all'],
    queryFn: () => searchSkillsHub(search, 'all', 100, profile),
    retry: false
  })

  const hosted = useHostedConnectors(profile)
  const mcp = useMcpServers({ gateway: null, profile })
  const pluginServers = usePluginServers(profile)

  const connectors = deriveCards({
    bundled: joinBundledEntries(mcp.availableCatalog),
    hosted: hosted.rows,
    local: [
      ...joinLocalServers({
        catalog: mcp.catalog,
        servers: mcp.servers,
        status: mcp.statuses,
        toolCounts: mcp.toolCounts,
        usage: mcp.usageByServer
      }),
      ...pluginServers
    ],
    titles: hosted.titles
  })

  const needle = query.trim().toLocaleLowerCase()

  const matches = (...values: (string | undefined)[]) =>
    values.some(value => value?.toLocaleLowerCase().includes(needle))

  const groups = [
    {
      tab: 'skills',
      label: t.skills.tabSkills,
      loading: !skills && !skillsFailed,
      failed: skillsFailed,
      retry: onRetry,
      rows: (skills ?? [])
        .filter(row => matches(row.name, row.description, row.category))
        .map(row => ({ id: row.name, name: row.name, description: row.description ?? '' }))
    },
    {
      tab: 'toolsets',
      label: t.skills.tabToolsets,
      loading: !toolsets && !toolsetsFailed,
      failed: toolsetsFailed,
      retry: onRetry,
      rows: (toolsets ?? [])
        .filter(
          row =>
            isDesktopToolsetVisible(row.name) &&
            matches(row.name, toolsetDisplayLabel(row), row.description, ...toolNames(row))
        )
        .map(row => ({ id: row.name, name: toolsetDisplayLabel(row), description: row.description ?? '' }))
    },
    {
      tab: 'plugins',
      label: t.skills.tabPlugins,
      loading: plugins.isPending || catalog.isPending,
      failed: plugins.isError || catalog.isError || catalog.data?.available === false,
      retry: () => {
        void plugins.refetch()
        void catalog.refetch()
      },
      rows: [
        ...mergePluginPackages(
          Object.values(desktop),
          (plugins.data?.plugins ?? []).filter(isDesktopRelevantPlugin)
        ).map(row => ({ id: row.key, name: row.name, description: row.description })),
        ...(catalog.data?.entries ?? [])
          .filter(
            row =>
              !mergePluginPackages(Object.values(desktop), plugins.data?.plugins ?? []).some(
                pkg => pkg.name === row.name
              )
          )
          .map(row => ({ id: row.name, name: row.name, description: row.description }))
      ].filter(row => matches(row.name, row.description))
    },
    {
      tab: 'connectors',
      label: t.connectorsPage.title,
      loading: mcp.configLoading || mcp.catalogLoading,
      failed: mcp.configFailed || hosted.phase === 'unavailable',
      retry: () => {
        mcp.refetchConfig()
        void hosted.refetch()
      },
      rows: connectors
        .filter(row => matches(row.name, row.slug, row.description))
        .map(row => ({ id: cardKey(row), name: row.name, description: row.description ?? '' }))
    },
    {
      tab: 'marketplace',
      label: c.marketplace,
      loading: marketplace.isPending || search !== query,
      failed: marketplace.isError,
      retry: () => {
        void marketplace.refetch()
      },
      rows:
        search === query
          ? (marketplace.data?.results ?? []).map(row => ({
              id: row.identifier,
              name: row.name,
              description: row.description ?? ''
            }))
          : []
    }
  ] satisfies {
    tab: SearchTab
    label: string
    loading: boolean
    failed: boolean
    retry?: () => void
    rows: { id: string; name: string; description: string }[]
  }[]

  return (
    <div aria-label={c.searchAll} className="h-full overflow-y-auto p-3" role="region">
      {groups.map(group => (
        <section aria-label={group.label} className="mb-5" key={group.tab}>
          <h2 className="mb-2 text-sm font-medium">
            {group.label} <span className="text-muted-foreground">{group.rows.length}</span>
          </h2>
          {group.loading && (
            <p className="text-xs text-muted-foreground" role="status">
              {t.skills.loading}
            </p>
          )}
          {group.failed && (
            <div className="text-xs" role="alert">
              {c.partial}{' '}
              {group.retry && (
                <Button onClick={group.retry} size="xs">
                  {c.retry}
                </Button>
              )}
            </div>
          )}
          {!group.loading && !group.failed && group.rows.length === 0 && (
            <p className="text-xs text-muted-foreground">{c.noMatches}</p>
          )}
          <div className="grid gap-1 sm:grid-cols-2 xl:grid-cols-3">
            {group.rows.map(row => (
              <Button
                aria-label={row.name}
                className="h-auto min-w-0 items-start justify-start whitespace-normal text-left"
                key={row.id}
                onClick={() => onOpen({ ...row, tab: group.tab })}
                variant="ghost"
              >
                <span className="min-w-0">
                  <span className="block break-words">{row.name}</span>
                  <span className="line-clamp-2 block text-xs font-normal text-muted-foreground">
                    {row.description}
                  </span>
                </span>
              </Button>
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}
