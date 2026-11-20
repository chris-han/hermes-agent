import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { getSkillHubSources, previewSkillHub, type ProfileScope, profileScopeKey, searchSkillsHub } from '@/hermes'
import { useI18n } from '@/i18n'
import { useInventoryCopy } from '@/i18n/capability-inventory'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $hubActions, hubActionKey, installHubSkill, updateHubSkills } from '@/store/hub-actions'
import { notifyError } from '@/store/notifications'

import { DetailColumn, ListColumn, MasterDetail } from '../../master-detail'
import { InventoryItems, InventoryRow, InventorySelect } from '../inventory'

import { marketplaceSources, readMarketplaceSource, saveMarketplaceSource } from './marketplace-source'

export function MarketplaceTab({
  profile,
  query,
  initialSelected
}: {
  profile: ProfileScope
  query: string
  initialSelected?: string
}) {
  return (
    <ScopedMarketplace
      initialSelected={initialSelected}
      key={profileScopeKey(profile)}
      profile={profile}
      query={query}
    />
  )
}

function ScopedMarketplace({
  profile,
  query,
  initialSelected
}: {
  profile: ProfileScope
  query: string
  initialSelected?: string
}) {
  const { t } = useI18n()
  const c = useInventoryCopy()
  const [search, setSearch] = useState(query)
  const [source, setSource] = useState(() => readMarketplaceSource(profileScopeKey(profile)))
  const [url, setUrl] = useState('')
  const [urlIdentifier, setUrlIdentifier] = useState('')
  const [selected, setSelected] = useState<string | null>(initialSelected ?? null)
  const [busy, setBusy] = useState<string | null>(null)
  const [completed, setCompleted] = useState<Set<string>>(new Set())
  const lock = useRef(false)
  const [logId, setLogId] = useState<string | null>(null)
  const action = useStoreSelector($hubActions, actions => (logId ? actions[hubActionKey(logId, profile)] : undefined))
  useEffect(() => {
    const timer = setTimeout(() => setSearch(query), 250)

    return () => clearTimeout(timer)
  }, [query])

  const { sources, results, visibleRows, active, preview } = useMarketplaceData(
    profile,
    search,
    source,
    selected,
    urlIdentifier
  )

  const data = results

  const installed = (id: string) =>
    completed.has(id) || Boolean((results.data?.installed ?? sources.data?.installed ?? {})[id])

  async function install(id: string) {
    if (lock.current || installed(id)) {
      return
    }

    lock.current = true
    setBusy(id)
    setLogId(id)

    try {
      await installHubSkill(id, profile)
      const confirmed = await getSkillHubSources(profile)

      if (!confirmed.installed[id]) {
        throw new Error(c.installUnconfirmed)
      }

      setCompleted(prev => new Set([...prev, id]))
      void data.refetch()
    } catch (error) {
      notifyError(error, t.skills.hub.actionFailed)
    } finally {
      lock.current = false
      setBusy(null)
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex flex-wrap items-center gap-2 px-3 py-2">
        <InventorySelect
          label={c.source}
          onChange={value => {
            setSource(value)
            saveMarketplaceSource(profileScopeKey(profile), value)
            setSelected(null)
            setUrlIdentifier('')
          }}
          options={sourceOptions(sources.data?.sources ?? [], source, c.all, c.unavailableSource)}
          value={source}
        />
        <Button
          disabled={busy !== null}
          onClick={() => {
            if (lock.current) {
              return
            }

            lock.current = true
            setBusy('__update__')
            setLogId('__update_all__')
            void updateHubSkills(profile)
              .catch(error => notifyError(error, t.skills.hub.actionFailed))
              .finally(() => {
                lock.current = false
                setBusy(null)
              })
          }}
          size="xs"
        >
          {t.skills.hub.updateAll}
        </Button>
      </div>
      {source === 'url' && (
        <form
          className="flex items-center gap-2 px-3 py-2"
          onSubmit={event => {
            event.preventDefault()
            setUrlIdentifier(url.trim())
          }}
        >
          <Input
            aria-label="SKILL.md URL"
            onChange={event => setUrl(event.target.value)}
            placeholder="https://…/SKILL.md"
            required
            type="url"
            value={url}
          />
          <Button size="xs" type="submit">
            {c.preview}
          </Button>
        </form>
      )}
      {(data.isError || results.data?.errors?.length) && (
        <div className="px-3 py-2 text-sm" role="alert">
          {c.unavailable}{' '}
          {results.data?.errors?.map(error => (
            <p key={error.source}>
              {error.source}: {error.message}
            </p>
          ))}
          <Button onClick={() => void data.refetch()} size="xs">
            {c.retry}
          </Button>
        </div>
      )}
      {results.data?.timed_out.length ? (
        <p className="px-3 text-xs" role="status">
          {c.partial}
        </p>
      ) : null}
      {source === 'url' && active ? (
        <div className="overflow-auto p-3">
          <p>{c.community}</p>
          {preview.isError ? (
            <p role="alert">{c.unavailable}</p>
          ) : (
            <pre className="whitespace-pre-wrap break-words text-xs">{preview.data?.skill_md}</pre>
          )}
          <Button
            disabled={!preview.data || preview.isError || busy !== null || installed(active.identifier)}
            onClick={() => void install(active.identifier)}
          >
            {t.skills.hub.install}
          </Button>
        </div>
      ) : data.isLoading && source !== 'url' ? (
        <p className="px-3" role="status">
          {t.skills.loading}
        </p>
      ) : visibleRows.length === 0 ? (
        <p className="px-3">{t.skills.hub.noResults}</p>
      ) : (
        <MasterDetail resizeId="capabilities-split" split="wide">
          <ListColumn>
            <InventoryItems>
              {visibleRows.map(row => (
                <InventoryRow
                  action={
                    <Button
                      disabled={busy !== null || installed(row.identifier)}
                      onClick={() => void install(row.identifier)}
                      size="xs"
                    >
                      {installed(row.identifier)
                        ? t.skills.hub.installed
                        : busy === row.identifier
                          ? t.skills.hub.installing
                          : t.skills.hub.install}
                    </Button>
                  }
                  active={active?.identifier === row.identifier}
                  enabled={installed(row.identifier)}
                  key={row.identifier}
                  onSelect={() => setSelected(row.identifier)}
                  subtitle={
                    <>
                      <span>{row.description}</span>
                      <span>
                        {sourceLabel(row.source)} · {row.version}
                      </span>
                    </>
                  }
                  title={row.name}
                />
              ))}
            </InventoryItems>
          </ListColumn>
          <DetailColumn>
            {active && (
              <>
                <h2 className="break-words text-lg font-semibold">{active.name}</h2>
                <p>{active.description}</p>
                <p className="text-xs">
                  {sourceLabel(active.source)} · {active.repo} · {active.version}
                </p>
                <MarketplaceVerification
                  integrityVerified={preview.data?.integrity_verified}
                  publisherVerified={active.publisher_verified}
                  source={active.source}
                />
                {preview.isError ? (
                  <Button onClick={() => void preview.refetch()}>{c.retry}</Button>
                ) : (
                  <pre className="whitespace-pre-wrap break-words text-xs">{preview.data?.skill_md}</pre>
                )}
              </>
            )}
          </DetailColumn>
        </MasterDetail>
      )}
      {action && (
        <details className="max-h-40 shrink-0 overflow-auto border-t border-(--ui-stroke-secondary) px-3 text-xs" open>
          <summary>{t.skills.hub.actionLog}</summary>
          <pre className="whitespace-pre-wrap break-words">{action.lines.join('\n')}</pre>
        </details>
      )}
    </div>
  )
}

function useMarketplaceData(
  profile: ProfileScope,
  search: string,
  source: string,
  selected: string | null,
  urlIdentifier: string
) {
  const sources = useQuery({
    queryKey: ['skill-hub-sources', profileScopeKey(profile)],
    queryFn: () => getSkillHubSources(profile),
    retry: false
  })

  const results = useQuery({
    queryKey: ['skill-hub-search', profileScopeKey(profile), search, source],
    queryFn: () => searchSkillsHub(search, source, 100, profile),
    enabled: source !== 'url',
    retry: false
  })

  const rows = results.data?.results ?? []

  const visibleRows = rows
    .filter(row => source === 'all' || Boolean(search.trim()) || row.source === source)
    .sort((a, b) => a.name.localeCompare(b.name))

  const listed = visibleRows.find(row => row.identifier === selected) ?? visibleRows[0]

  const active =
    source === 'url'
      ? urlIdentifier
        ? {
            identifier: urlIdentifier,
            name: urlIdentifier,
            description: '',
            source: 'url',
            repo: '',
            version: undefined,
            publisher_verified: false
          }
        : undefined
      : listed

  const preview = useQuery({
    queryKey: ['skill-hub-preview', profileScopeKey(profile), active?.identifier],
    queryFn: () => previewSkillHub(active!.identifier, profile),
    enabled: Boolean(active),
    retry: false
  })

  return { sources, results, visibleRows, active, preview }
}

function sourceLabel(source: string) {
  return marketplaceSources.find(item => item.value === source)?.label ?? source
}

function sourceOptions(sources: { id: string; label: string }[], source: string, all: string, unavailable: string) {
  const options = [
    ...marketplaceSources,
    { value: 'all', label: all },
    ...sources
      .filter(row => !marketplaceSources.some(fixed => fixed.value === row.id))
      .map(row => ({ value: row.id, label: row.label }))
  ]

  if (!options.some(row => row.value === source)) {
    options.push({ value: source, label: `${source} · ${unavailable}` })
  }

  return options
}

function MarketplaceVerification({
  source,
  publisherVerified,
  integrityVerified
}: {
  source: string
  publisherVerified?: boolean
  integrityVerified?: boolean
}) {
  const c = useInventoryCopy()

  return (
    <>
      <p className="text-xs">{source === 'semantier' && publisherVerified ? c.publisherVerified : c.community}</p>
      {integrityVerified && <p className="text-xs">{c.integrityVerified}</p>}
      <p className="text-xs">{c.scannerSeparate}</p>
    </>
  )
}
