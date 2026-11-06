import type { ModelOptionProvider } from '@hermes/shared'

import type { IconComponent } from '@/lib/icons'
import { normalize } from '@/lib/text'
import type { ConfigFieldSchema, EnvVarInfo, HermesConfigRecord } from '@/types/hermes'

import { FIELD_LABELS, SECTIONS } from './constants'
import { credentialRowLabel } from './credential-key-ui'
import { fieldCopyForSchemaKey } from './field-copy'
import { prettyName, sectionFieldEntries, voiceFieldVisible } from './helpers'
import { settingsSubpageForTarget } from './subpages'
import type { DesktopConfigSection, SettingsView } from './types'

export type CredentialSettingsView = 'settings' | 'tools'

export interface SettingsSearchTarget {
  subpage?: string
  field?: string
  key?: string
  keysView?: CredentialSettingsView
  providerView?: 'accounts' | 'custom-endpoints' | 'keys'
  setting?: string
  modelProvider?: string
  modelId?: string
  modelProfile?: string
  modelConnection?: string
  view: SettingsView
}

export interface SettingsSearchEntry {
  context: string
  description?: string
  icon: IconComponent
  id: string
  keywords: string[]
  label: string
  target: SettingsSearchTarget
}

interface ConfigSearchCopy {
  fieldDescriptions: Record<string, string>
  fieldLabels: Record<string, string>
  sections: Record<string, string>
}

interface CredentialSearchCopy {
  settings: string
  tools: string
}

function offeredEligibleModels(provider: ModelOptionProvider): string[] {
  if (provider.authenticated === false || provider.eligible_models == null) {
    return []
  }

  const unavailable = new Set((provider.unavailable_models ?? []).map(normalize))
  const listed = new Set(provider.models ?? [])

  return provider.eligible_models.filter(model => listed.has(model) && !unavailable.has(normalize(model)))
}

/** Validate a navigation intent against the currently loaded, identically scoped inventory. */
export function resolveEligibleModelTarget(
  params: URLSearchParams,
  providers: readonly ModelOptionProvider[],
  scope: { profile: string; connection: string }
): { provider: string; model?: string } | null {
  const providerId = params.get('model_provider')
  if (!providerId || !scope.profile || !scope.connection || params.get('model_profile') !== scope.profile ||
      params.get('model_connection') !== scope.connection) {
    return null
  }

  const provider = providers.find(row => row.slug === providerId)
  const eligible = provider ? offeredEligibleModels(provider) : []
  const model = params.get('model')
  if (!eligible.length || (params.has('model') && (!model || !eligible.includes(model)))) {
    return null
  }

  return model ? { provider: providerId, model } : { provider: providerId }
}

/** Index only producer-established model sets, never mixed picker hints. */
export function buildEligibleModelSearchEntries(
  providers: readonly ModelOptionProvider[],
  scope: { profile: string; connection: string; context: string; icon: IconComponent }
): SettingsSearchEntry[] {
  const seen = new Set<string>()
  const entries: SettingsSearchEntry[] = []

  for (const provider of providers) {
    if (!provider.slug || provider.authenticated === false || provider.eligible_models == null) {
      continue
    }

    const models = offeredEligibleModels(provider)
    const target = { view: 'config:model' as const, subpage: 'main', modelProvider: provider.slug,
      modelProfile: scope.profile, modelConnection: scope.connection }
    const providerKey = JSON.stringify([provider.slug])

    if (models.length && !seen.has(providerKey)) {
      seen.add(providerKey)
      entries.push({ context: scope.context, icon: scope.icon, id: `model-provider:${providerKey}`,
        keywords: [provider.slug], label: provider.name, target })
    }

    for (const model of models) {
      const key = JSON.stringify([provider.slug, model])
      if (seen.has(key)) {
        continue
      }
      seen.add(key)
      entries.push({ context: provider.name, icon: scope.icon, id: `model:${key}`,
        keywords: [provider.slug, provider.name, model], label: model, target: { ...target, modelId: model } })
    }
  }

  return entries
}

export function credentialSettingsView(info: EnvVarInfo): CredentialSettingsView | null {
  if (info.channel_managed) {
    return null
  }

  if (info.category === 'tool') {
    return 'tools'
  }

  if (info.category === 'setting' || info.category === 'messaging') {
    return 'settings'
  }

  return null
}

function configFieldLabel(key: string, copy: ConfigSearchCopy): string {
  return (
    fieldCopyForSchemaKey(copy.fieldLabels, key) ??
    fieldCopyForSchemaKey(FIELD_LABELS, key) ??
    prettyName(key.split('.').pop() ?? key)
  )
}

function configFieldDescription(key: string, field: ConfigFieldSchema, copy: ConfigSearchCopy): string {
  return fieldCopyForSchemaKey(copy.fieldDescriptions, key) ?? field.description ?? ''
}

export function buildConfigSearchEntries(
  schema: Record<string, ConfigFieldSchema> | null | undefined,
  config: HermesConfigRecord | null | undefined,
  copy: ConfigSearchCopy,
  sections: DesktopConfigSection[] = SECTIONS
): SettingsSearchEntry[] {
  if (!schema || !config) {
    return []
  }

  const sectionFields = sectionFieldEntries(schema, config)

  return sections.flatMap(section => {
    const context = copy.sections[section.id] ?? section.label
    const fields = sectionFields.get(section.id) ?? []
    const visibleFields = section.id === 'voice' ? fields.filter(([key]) => voiceFieldVisible(key, config)) : fields

    return visibleFields.map(([key, field]) => ({
      context,
      description: configFieldDescription(key, field, copy),
      icon: section.icon,
      id: `config-field:${key}`,
      keywords: ['settings', section.id, section.label, key],
      label: configFieldLabel(key, copy),
      target: {
        field: key,
        view: `config:${section.id}` as SettingsView
      }
    }))
  })
}

export function buildCredentialSearchEntries(
  vars: Record<string, EnvVarInfo> | null | undefined,
  copy: CredentialSearchCopy,
  icons: Record<CredentialSettingsView, IconComponent>
): SettingsSearchEntry[] {
  if (!vars) {
    return []
  }

  return Object.entries(vars)
    .sort(([a], [b]) => a.localeCompare(b))
    .flatMap(([key, info]) => {
      const view = credentialSettingsView(info)

      if (!view) {
        return []
      }

      return [
        {
          context: view === 'tools' ? copy.tools : copy.settings,
          description: info.description || undefined,
          icon: icons[view],
          id: `credential:${key}`,
          keywords: [key, info.url ?? '', ...(Array.isArray(info.tools) ? info.tools : [])],
          label: credentialRowLabel(key, info),
          target: {
            key,
            keysView: view,
            view: 'keys' as const
          }
        }
      ]
    })
}

function searchScore(entry: SettingsSearchEntry, query: string): number {
  const needle = normalize(query)

  if (!needle) {
    return 0
  }

  const label = normalize(entry.label)
  const context = normalize(entry.context)
  const haystack = normalize([entry.label, entry.context, entry.description ?? '', ...entry.keywords].join(' '))
  const terms = needle.split(/\s+/).filter(Boolean)

  if (!terms.every(term => haystack.includes(term))) {
    return 0
  }

  if (label === needle) {
    return 100
  }

  if (label.startsWith(needle)) {
    return 90
  }

  if (label.includes(needle)) {
    return 80
  }

  if (context.includes(needle)) {
    return 70
  }

  if (terms.every(term => label.includes(term) || context.includes(term))) {
    return 60
  }

  return 50
}

export function filterSettingsSearchEntries(entries: SettingsSearchEntry[], query: string): SettingsSearchEntry[] {
  return entries
    .map((entry, index) => ({ entry, index, score: searchScore(entry, query) }))
    .filter(result => result.score > 0)
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map(result => result.entry)
}

/** Serialize a search target to the Settings route's query string. */
export function settingsSearchTargetQuery(target: SettingsSearchTarget): string {
  const params = new URLSearchParams()
  params.set('tab', target.view)
  const subpage = target.subpage ?? settingsSubpageForTarget(target.view, target.field, target.setting)

  if (subpage) {
    params.set('page', subpage)
  }

  if (target.providerView) {
    params.set('pview', target.providerView)
  }

  if (target.keysView) {
    params.set('kview', target.keysView)
  }

  if (target.field) {
    params.set('field', target.field)
  }

  if (target.setting) {
    params.set('setting', target.setting)
  }

  if (target.key) {
    params.set('key', target.key)
  }

  if (target.modelProvider) {
    params.set('model_provider', target.modelProvider)
  }

  if (target.modelId) {
    params.set('model', target.modelId)
  }

  if (target.modelProfile) {
    params.set('model_profile', target.modelProfile)
  }

  if (target.modelConnection) {
    params.set('model_connection', target.modelConnection)
  }

  return params.toString()
}
