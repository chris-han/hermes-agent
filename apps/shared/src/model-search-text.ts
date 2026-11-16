/**
 * Extra tokens used only for model-picker search ranking.
 *
 * Wire IDs stay unchanged — some providers report short or brand-less ids
 * (Kimi Coding's flagship is literally `k3`) that users still search for by
 * the familiar `kimi-…` naming of sibling models.
 *
 * Keep in sync with the Python twin, hermes_cli/model_search.py.
 */
const MODEL_SEARCH_ALIASES: Record<string, readonly string[]> = {
  k3: ['kimi-k3', 'kimi'],
  // OpenCode Zen serves the "Ox Alpha" stealth model under an opaque
  // preview slug; let users find it by its public codename.
  'x-preview-f-free': ['ox-alpha', 'ox']
}

/** Haystack for fuzzy/substring model search; never changes the wire id. */
export function modelSearchText(model: string): string {
  const id = model.trim()

  if (!id) {
    return model
  }

  const aliases = MODEL_SEARCH_ALIASES[id.toLowerCase()]

  if (!aliases?.length) {
    return id
  }

  return `${id} ${aliases.join(' ')}`
}

/** Safe catalog identity only. This projection never changes routing or grants. */
export interface CatalogModelMetadata {
  label?: string
  provider_id?: string
  provider_display?: string
  provider_aliases?: string[]
  source_model_id?: string
  modality?: string
  context_window?: number | null
  max_output_tokens?: number | null
  max_input_tokens?: number | null
  capabilities?: Record<string, boolean | null>
  revision?: string
  deployment_revision?: string
  sources?: Record<string, string>
  protocol?: string
  unknown_reason?: string | null
}

interface CatalogSearchProvider {
  slug: string
  name: string
  aliases?: string[] | null
  model_profiles?: Record<string, CatalogModelMetadata> | null
}

export function catalogModelMetadata(provider: CatalogSearchProvider | undefined, model: string): CatalogModelMetadata | undefined {
  return provider?.model_profiles?.[model]
}

/** Required-field issues affect only this Cloud model. */
export function catalogModelIssue(provider: CatalogSearchProvider | undefined, model: string): string | undefined {
  if (provider?.slug !== 'semantier-cloud') { return undefined }
  const metadata = catalogModelMetadata(provider, model)

  const missing = (['context_window', 'max_output_tokens'] as const).filter(key =>
    !Number.isSafeInteger(metadata?.[key]) || (metadata?.[key] ?? 0) <= 0)

  if (metadata?.protocol && !['chat', 'chat_completions', 'openai_chat_completions'].includes(metadata.protocol)) { return 'protocol' }

  if (missing.length) { return missing.join(', ') }

  if (metadata!.max_output_tokens! > metadata!.context_window!) { return 'context_window / max_output_tokens' }

  return undefined
}

/** The same field union is used by Settings and every Composer catalog. */
export function catalogModelSearchText(provider: CatalogSearchProvider, model: string): string {
  const metadata = catalogModelMetadata(provider, model)

  return [modelSearchText(model), provider.slug, provider.name, ...(provider.aliases ?? []),
    metadata?.label, metadata?.provider_id, metadata?.provider_display,
    ...(metadata?.provider_aliases ?? []), metadata?.source_model_id].filter(value => typeof value === 'string').join(' ')
}

/** Terms may span provider and model fields; a provider filter is applied first. */
export function catalogSearchMatches(text: string, query: string): boolean {
  const haystack = text.toLowerCase()

  return query.trim().toLowerCase().split(/\s+/).every(term => haystack.includes(term))
}
