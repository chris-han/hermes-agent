import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { SearchField } from '@/components/ui/search-field'
import { useI18n } from '@/i18n'
import { useInventoryCopy } from '@/i18n/capability-inventory'
import type { PluginCatalogEntry } from '@/lib/plugin-catalog'
import { queryClient } from '@/lib/query-client'
import type { GatewayRequest } from '@/store/agent-plugins'
import { openCatalogPluginInstall } from '@/store/plugin-catalog-install'

export function NativePluginCatalog({
  request,
  profile,
  connectionId
}: {
  request: GatewayRequest
  profile: string | null
  connectionId?: string
}) {
  const { t } = useI18n()
  const c = useInventoryCopy()
  const [query, setQuery] = useState('')

  const catalog = useQuery(
    {
      queryKey: ['native-plugin-catalog', connectionId, profile],
      queryFn: () =>
        request<{ available: boolean; entries: (PluginCatalogEntry & { description: string })[] }>(
          'plugins.catalog.list',
          profile ? { profile } : {},
          20_000
        ),
      retry: false
    },
    queryClient
  )

  return (
    <section className="grid gap-2 border-t border-(--ui-stroke-secondary) p-3">
      <h2 className="text-sm font-medium">{t.skills.plugins.catalogTitle}</h2>
      <SearchField onChange={setQuery} placeholder={t.skills.plugins.catalogTitle} value={query} />
      {catalog.isLoading ? (
        <p role="status">{t.skills.loading}</p>
      ) : catalog.isError || !catalog.data?.available ? (
        <div role="alert">
          {c.unavailable}{' '}
          <Button onClick={() => void catalog.refetch()} size="xs">
            {c.retry}
          </Button>
        </div>
      ) : (
        <div className="grid gap-2">
          {catalog.data.entries
            .filter(row => `${row.name} ${row.description}`.toLowerCase().includes(query.toLowerCase()))
            .map(row => (
              <div
                className="flex min-w-0 items-start gap-2 border-b border-(--ui-stroke-tertiary) py-2"
                key={row.name}
              >
                <div className="min-w-0 flex-1">
                  <h3 className="break-words text-sm">{row.name}</h3>
                  <p className="break-words text-xs text-(--ui-text-tertiary)">{row.description}</p>
                  <p className="break-all text-xs text-(--ui-text-tertiary)">{row.repo}</p>
                </div>
                <Button onClick={() => openCatalogPluginInstall(row, profile, { connectionId })} size="xs">
                  {t.skills.hub.install}
                </Button>
              </div>
            ))}
        </div>
      )}
    </section>
  )
}
