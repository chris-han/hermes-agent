import { REASONING_EFFORT_VALUES } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { modelProviderName } from '@/i18n/model-service'
import { Plus } from '@/lib/icons'
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
// provider/model/reasoning; every other key rides along untouched, or a save
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

function entriesEqual(a: FallbackEntry[], b: FallbackEntry[]): boolean {
  return a.length === b.length && a.every((entry, index) => JSON.stringify(entry) === JSON.stringify(b[index]))
}

/** Profile/source changes discard drafts and outstanding picker state. */
export function FallbackModelsField(props: { value: unknown; onChange: (next: FallbackEntry[]) => void | Promise<void> }) {
  const connection = useStore($connection)
  const profile = useStore($settingsScopeProfile)
  const ownerKey = modelDiscoveryOwnerKey(null, connection)

  return <ScopedFallbackModelsField {...props} key={`${ownerKey}:${profile}`} />
}

function ScopedFallbackModelsField({ value, onChange }: { value: unknown; onChange: (next: FallbackEntry[]) => void | Promise<void> }) {
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
  const [savedRows, setSavedRows] = useState(() => normalizeEntries(value))
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState(false)
  const lastReceived = useRef(normalizeEntries(value))

  // A genuine reload replaces the saved snapshot; ordinary parent renders do not discard drafts.
  // eslint-disable-next-line no-restricted-syntax -- local persisted editor snapshot, not an atom mirror
  useEffect(() => {
    const persisted = normalizeEntries(value)

    if (!entriesEqual(persisted, lastReceived.current)) {
      lastReceived.current = persisted
      setSavedRows(persisted)
      setRows(persisted)
      setSaveError(false)
    }
  }, [value])

  const dirty = !entriesEqual(rows, savedRows)
  const complete = rows.every(entry => entry.provider && entry.model.trim())

  const commit = (next: FallbackEntry[]) => {
    setRows(next)
    setSaveError(false)
  }

  const apply = async () => {
    if (!dirty || !complete || saving) {return}
    setSaving(true)
    setSaveError(false)

    try {
      await onChange(rows)
      setSavedRows(rows)
    } catch {
      setSaveError(true)
    } finally {
      setSaving(false)
    }
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
      {rows.length > 0 && <ol className="grid gap-4">
        {rows.map((entry, index) => {
          const providerRow = findCatalogProvider(providers, entry.provider)

          return (
            <li className="grid gap-2" key={index}>
              <div className="flex items-center gap-2">
                <span className="w-4 shrink-0 text-center font-mono text-[0.7rem] text-muted-foreground">{index + 1}</span>
                <Button
                  aria-label={t.common.remove}
                  disabled={saving}
                  onClick={() => commit(rows.filter((_, i) => i !== index))}
                  size="sm"
                  variant="text"
                >
                  {t.common.remove}
                </Button>
              </div>
              <div className="grid gap-2 sm:grid-cols-3">
                <label className="grid min-w-0 gap-1">
                  <span className="text-xs text-muted-foreground">{m.provider}</span>
                  <SearchableSelect
                    aria-label={m.provider}
                    className="w-full min-w-0"
                    disabled={saving}
                    onChange={provider => switchProvider(index, provider)}
                    options={providers.map(provider => ({ value: provider.slug, label: modelProviderName(provider, locale), disabled: provider.authenticated === false }))}
                    placeholder={m.provider}
                    value={providerRow?.slug ?? entry.provider}
                  />
                </label>
                <label className="grid min-w-0 gap-1">
                  <span className="text-xs text-muted-foreground">{m.model}</span>
                  <ModelSelect
                    allowedModels={providerRow?.eligible_models}
                    aria-label={m.model}
                    className="w-full min-w-0"
                    disabled={saving || !providerRow || providerRow.authenticated === false}
                    excludeUnavailable
                    manualScope={manualScope}
                    models={[...providerSelectableModels(providerRow), ...providerManualModels(providerRow), ...customModels.filter(custom => custom.scope === manualScope && custom.provider === (providerRow?.slug ?? entry.provider)).map(custom => custom.model)]}
                    onValueChange={model => updateRow(index, { model })}
                    provider={providerRow}
                    providerSlug={entry.provider}
                    searchable
                    value={entry.model}
                  />
                </label>
                <label className="grid min-w-0 gap-1">
                  <span className="text-xs text-muted-foreground">{m.reasoning}</span>
                  <Select
                    disabled={saving || providerRow?.capabilities?.[entry.model]?.reasoning === false}
                    onValueChange={effort => {
                      const { reasoning_effort: _previous, ...route } = entry
                      commit(rows.map((row, i) => i === index
                        ? effort === '__inherit__' ? route as FallbackEntry : { ...entry, reasoning_effort: effort }
                        : row))
                    }}
                    value={entry.reasoning_effort === false ? 'none' : String(entry.reasoning_effort ?? '__inherit__')}
                  >
                    <SelectTrigger aria-label={`${m.reasoning} ${index + 1}`} className="w-full"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="__inherit__">{m.fallbackInheritEffort}</SelectItem>
                      {REASONING_EFFORT_VALUES.map(effort => (
                        <SelectItem key={effort} value={effort}>{effort === 'none' ? m.reasoningOff : t.shell.modelOptions[effort]}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </label>
              </div>
            </li>
          )
        })}
      </ol>}
      <div>
        <Button disabled={saving} onClick={() => commit([...rows, { provider: '', model: '' }])} size="sm" variant="textStrong">
          <Plus className="size-3.5" />
          {m.fallbackAdd}
        </Button>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button disabled={!dirty || !complete || saving} onClick={() => void apply()} size="sm">
          {saving ? t.common.saving : t.common.apply}
        </Button>
        <Button disabled={!dirty || saving} onClick={() => { setRows(savedRows); setSaveError(false) }} size="sm" variant="ghost">
          {t.common.cancel}
        </Button>
      </div>
      {saveError && <p className="text-xs text-destructive" role="alert">{t.settings.config.autosaveFailed}</p>}
    </div>
  )
}
