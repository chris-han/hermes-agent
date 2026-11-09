import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { modelProviderName } from '@/i18n/model-service'
import { Plus, X } from '@/lib/icons'
import { modelDiscoveryOwnerKey } from '@/lib/model-discovery-scope'
import { findCatalogProvider, modelOptionsQueryKey, providerManualModels, providerSelectableModels, requestDiscoveredModelOptions } from '@/lib/model-options'
import { $customModels } from '@/store/custom-models'
import { requestGatewayForAgent } from '@/store/gateway'
import { $connection } from '@/store/session'
import { $settingsScopeProfile } from '@/store/settings-scope'

import { ModelSelect } from './model-select'
import { SearchableSelect } from './searchable-select'

// An entry is `{provider, model}` plus whatever routing the user hand-wrote
// (`base_url`, `api_key`, `key_env`, `api_mode`, ...). The editor only edits
// the two selects; every other key rides along untouched, or an autosave
// would rewrite a local-gateway chain as bare provider/model pairs and route
// fallbacks to the public provider (#89184).
interface FallbackEntry extends Record<string, unknown> {
  provider: string
  model: string
}

// Normalize the raw config value (`fallback_providers`: a list of
// `{provider, model, ...}` dicts) into editor rows. Defensive against legacy
// string entries ("provider/model") so the editor never crashes on odd data.
function normalizeEntries(value: unknown): FallbackEntry[] {
  if (!Array.isArray(value)) {
    return []
  }

  return value.map(item => {
    if (item && typeof item === 'object') {
      const record = item as Record<string, unknown>

      return { ...record, provider: String(record.provider ?? ''), model: String(record.model ?? '') }
    }

    if (typeof item === 'string') {
      const slash = item.indexOf('/')

      return slash > 0
        ? { provider: item.slice(0, slash), model: item.slice(slash + 1) }
        : { provider: '', model: item }
    }

    return { provider: '', model: '' }
  })
}

function completeEntries(rows: FallbackEntry[]): FallbackEntry[] {
  return rows.filter(entry => entry.provider && entry.model)
}

function entriesEqual(a: FallbackEntry[], b: FallbackEntry[]): boolean {
  return a.length === b.length && a.every((entry, index) => JSON.stringify(entry) === JSON.stringify(b[index]))
}

/** Profile/source changes discard drafts and outstanding picker state. */
export function FallbackModelsField(props: { value: unknown; onChange: (next: FallbackEntry[]) => void }) {
  const connection = useStore($connection)
  const profile = useStore($settingsScopeProfile)
  const ownerKey = modelDiscoveryOwnerKey(null, connection)

  return <ScopedFallbackModelsField {...props} key={`${ownerKey}:${profile}`} />
}

function ScopedFallbackModelsField({ value, onChange }: { value: unknown; onChange: (next: FallbackEntry[]) => void }) {
  const { t, locale } = useI18n()
  const m = t.settings.model
  const connection = useStore($connection)
  const profile = useStore($settingsScopeProfile)
  const ownerKey = modelDiscoveryOwnerKey(null, connection)
  const manualScope = `${ownerKey}:${profile}`

  const modelOptions = useQuery({
    queryKey: modelOptionsQueryKey(profile, null, ownerKey),
    enabled: !!ownerKey,
    queryFn: () => requestDiscoveredModelOptions({
      connection,
      ownerConnectionId: connection?.connectionId,
      profile,
      request: <T,>(method: string, params?: Record<string, unknown>) =>
        requestGatewayForAgent<T>(connection?.connectionId ?? null, profile, method, params)
    })
  })

  const customModels = useStore($customModels)

  const providers = (modelOptions.data?.providers ?? []).filter(provider => provider.slug)

  const [rows, setRows] = useState<FallbackEntry[]>(() => normalizeEntries(value))
  // Last complete chain we emitted (or seeded). Autosave echoes the same
  // filtered list back through `value`; ignore that echo so draft rows stay.
  const lastEmittedRef = useRef(normalizeEntries(value))
  // One saved slot per draft row. Incomplete replacements retain this route.
  const savedRows = useRef<(FallbackEntry | undefined)[]>(normalizeEntries(value))

  // Resync on real external changes (profile switch / config reload). Skip
  // when `value` is just our own commit echoing through the parent.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    const persisted = normalizeEntries(value)

    if (entriesEqual(persisted, lastEmittedRef.current)) {
      return
    }

    lastEmittedRef.current = persisted
    savedRows.current = persisted
    setRows(persisted)
  }, [value])

  const commit = (next: FallbackEntry[], saved = savedRows.current) => {
    const slots = next.map((entry, index) => entry.provider && entry.model ? entry : saved[index])
    const complete = completeEntries(slots.filter((entry): entry is FallbackEntry => !!entry))

    savedRows.current = slots
    setRows(next)
    lastEmittedRef.current = complete
    onChange(complete)
  }

  const updateRow = (index: number, patch: Partial<FallbackEntry>) =>
    commit(rows.map((entry, i) => (i === index ? { ...entry, ...patch } : entry)))

  const switchProvider = (index: number, provider: string) => {
    if ((findCatalogProvider(providers, rows[index].provider)?.slug ?? rows[index].provider) === provider) {return}
    // The previous route remains saved, but none of its metadata enters the new draft.
    commit(rows.map((entry, i) => i === index ? { provider, model: '' } : entry))
  }

  return (
    <div className="grid w-full gap-1.5">
      {rows.length === 0 && <p className="text-xs text-muted-foreground">{m.fallbackEmpty}</p>}
      {rows.map((entry, index) => {
        const providerRow = findCatalogProvider(providers, entry.provider)

        return (
          <div className="flex flex-wrap items-center gap-2" key={index}>
            <span className="w-4 shrink-0 text-center font-mono text-[0.7rem] text-muted-foreground">{index + 1}</span>
            <SearchableSelect
              aria-label={m.provider}
              onChange={provider => switchProvider(index, provider)}
              options={providers.map(provider => ({ value: provider.slug, label: modelProviderName(provider, locale), disabled: provider.authenticated === false }))}
              placeholder={m.provider}
              value={providerRow?.slug ?? entry.provider}
            />
            <ModelSelect
              allowedModels={providerRow?.eligible_models}
              className="min-w-52 flex-1"
              disabled={!providerRow || providerRow.authenticated === false}
              excludeUnavailable
              manualScope={manualScope}
              models={[...providerSelectableModels(providerRow), ...providerManualModels(providerRow), ...customModels.filter(custom => custom.scope === manualScope && custom.provider === (providerRow?.slug ?? entry.provider)).map(custom => custom.model)]}
              onValueChange={model => updateRow(index, { model })}
              provider={providerRow}
              providerSlug={entry.provider}
              searchable
              value={entry.model}
            />
            {!entry.model && savedRows.current[index] && (
              <Button onClick={() => updateRow(index, savedRows.current[index]!)} size="sm" variant="ghost">
                {t.common.cancel}
              </Button>
            )}
            <Button
              aria-label={t.common.remove}
              onClick={() => commit(rows.filter((_, i) => i !== index), savedRows.current.filter((_, i) => i !== index))}
              size="icon-xs"
              variant="ghost"
            >
              <X className="size-3.5" />
            </Button>
          </div>
        )
      })}
      <div>
        <Button onClick={() => commit([...rows, { provider: '', model: '' }])} size="sm" variant="textStrong">
          <Plus className="size-3.5" />
          {m.fallbackAdd}
        </Button>
      </div>
    </div>
  )
}
