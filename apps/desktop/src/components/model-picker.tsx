import type { ModelOptionProvider, ModelPricing } from '@hermes/shared'
import { fuzzyRank, modelSearchText } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { type ReactElement, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { catalogProviderMatches, modelOptionsQueryKey, requestModelOptions } from '@/lib/model-options'
import { currentPickerSelection } from '@/lib/model-status-label'
import { accountResetMs, formatReset, modelResetMs } from '@/lib/provider-limit'
import { foldIncludes, normalize } from '@/lib/text'
import { cn } from '@/lib/utils'
import { $customModels, addCustomModel, customModelCandidate, withCustomModels } from '@/store/custom-models'

import type { HermesGateway } from '../hermes'
import { startManualOnboarding } from '../store/onboarding'

import { InlineNotice } from './notifications'
import { ProviderStatusChip } from './provider-status-chip'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from './ui/command'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from './ui/dialog'
import { HighlightMatches } from './ui/highlight-matches'
import { Skeleton } from './ui/skeleton'

interface ModelPickerDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  gw?: HermesGateway
  sessionId?: string | null
  currentModel: string
  currentProvider: string
  onSelect: (selection: { provider: string; model: string }) => void
  ownerConnectionId?: null | string
  profile?: string
  /** Desktop route profile for provider setup; `profile` may be the backend-side target. */
  setupProfile?: string
  request?: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  /**
   * Optional class for DialogContent. Use it to lift the picker onto a higher
   * rung of the overlay ladder when it opens over another fixed overlay (the
   * desktop onboarding overlay, say) — on the default modal rung it renders
   * underneath and blocks pointer events.
   */
  contentClassName?: string
}

export function ModelPickerDialog({
  open,
  onOpenChange,
  gw,
  sessionId,
  currentModel,
  currentProvider,
  onSelect,
  ownerConnectionId,
  profile = 'default',
  request,
  setupProfile,
  contentClassName
}: ModelPickerDialogProps): ReactElement {
  const { t } = useI18n()
  const copy = t.modelPicker
  // Own the search term so we can filter manually. cmdk's built-in
  // shouldFilter reorders items by its fuzzy-match score (≈alphabetical with
  // an empty query), which destroys the backend's curated order. We disable
  // it: an empty query shows the curated list verbatim (like the `hermes
  // model` CLI picker) and a query ranks with the shared fuzzyRank.
  const [search, setSearch] = useState<string>('')
  // "Add custom model…" flips the search into slug entry: the typed id is
  // offered per provider even while it fuzzy-matches catalog rows.
  const [slugEntry, setSlugEntry] = useState(false)
  const searchRef = useRef<HTMLInputElement>(null)

  const modelOptions = useQuery({
    queryKey: modelOptionsQueryKey(profile, sessionId, ownerConnectionId),
    queryFn: () => requestModelOptions({ gateway: gw, profile, request, sessionId }),
    enabled: open
  })

  const customModels = useStore($customModels)
  const providers = withCustomModels(modelOptions.data?.providers ?? [], customModels)

  const { model: optionsModel, provider: optionsProvider } = currentPickerSelection(
    { model: currentModel, provider: currentProvider },
    modelOptions.data
  )

  const loading = modelOptions.isPending && !modelOptions.data

  const error = modelOptions.error
    ? modelOptions.error instanceof Error
      ? modelOptions.error.message
      : String(modelOptions.error)
    : null

  const selectModel = (provider: ModelOptionProvider, model: string) => {
    onSelect({ provider: provider.slug, model })
    onOpenChange(false)
  }

  const selectCustomModel = (provider: ModelOptionProvider, model: string) => {
    addCustomModel(provider.slug, model, provider)
    selectModel(provider, model)
  }

  // Open the full onboarding provider selector to add/switch a provider.
  // Reuses the entire onboarding flow (OAuth rows, API-key form, device-code,
  // model-confirm) instead of duplicating provider UI here. Closes the picker
  // so the onboarding overlay isn't rendered underneath it.
  const addProvider = () => {
    const ownerProfile = setupProfile ?? profile

    startManualOnboarding(
      undefined,
      ownerConnectionId !== undefined ? { connectionId: ownerConnectionId, profile: ownerProfile } : ownerProfile
    )
    onOpenChange(false)
  }

  const enterSlug = () => {
    setSlugEntry(true)
    searchRef.current?.focus()
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent
        bodyClassName="gap-0 overflow-hidden p-0"
        className={cn('max-h-[85vh] max-w-2xl', contentClassName)}
      >
        <DialogHeader className="border-b border-border px-4 py-3">
          <DialogTitle>{copy.title}</DialogTitle>
          <DialogDescription className="font-mono text-xs leading-relaxed">
            {copy.current} {optionsModel || currentModel || copy.unknown}
            {optionsProvider || currentProvider ? ` · ${optionsProvider || currentProvider}` : ''}
          </DialogDescription>
        </DialogHeader>

        <Command className="rounded-none bg-card" shouldFilter={false}>
          <CommandInput
            autoFocus
            onValueChange={setSearch}
            placeholder={slugEntry ? copy.customModelPlaceholder : copy.search}
            ref={searchRef}
            value={search}
          />
          <CommandList className="max-h-96">
            {!loading && !error && <CommandEmpty>{copy.noModels}</CommandEmpty>}
            <ModelResults
              currentModel={optionsModel || currentModel}
              currentProvider={optionsProvider || currentProvider}
              error={error}
              loading={loading}
              offerCustom={slugEntry}
              onSelectCustomModel={selectCustomModel}
              onSelectModel={selectModel}
              providers={providers}
              search={search}
            />
          </CommandList>
        </Command>

        <DialogFooter className="flex-row items-center justify-end gap-2 bg-card p-3">
          <Button className="mr-auto" onClick={enterSlug} variant="ghost">
            {copy.addCustomModelAction}
          </Button>
          <Button onClick={addProvider} variant="ghost">
            {copy.addProvider}
          </Button>
          <Button onClick={() => onOpenChange(false)} variant="outline">
            {t.common.cancel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function ModelResults({
  loading,
  error,
  providers,
  currentModel,
  currentProvider,
  onSelectModel,
  onSelectCustomModel,
  offerCustom,
  search
}: {
  loading: boolean
  error: string | null
  providers: readonly ModelOptionProvider[]
  currentModel: string
  currentProvider: string
  onSelectModel: (provider: ModelOptionProvider, model: string) => void
  onSelectCustomModel: (provider: ModelOptionProvider, model: string) => void
  /** Offer the typed id as a custom model even while catalog rows match. */
  offerCustom: boolean
  search: string
}): ReactElement {
  const { t } = useI18n()
  const copy = t.modelPicker

  if (loading) {
    return <LoadingResults />
  }

  if (error) {
    return (
      <div className="px-3 py-3">
        <InlineNotice kind="error" title={copy.loadFailed}>
          {error}
        </InlineNotice>
      </div>
    )
  }

  if (providers.length === 0) {
    return <div className="px-4 py-6 text-sm text-muted-foreground">{copy.noAuthenticatedProviders}</div>
  }

  const q = normalize(search)

  // Model rows rank with the same fuzzyRank + modelSearchText the web and TUI
  // pickers use, so one query orders identically on every surface. A query
  // that names the provider itself keeps its whole curated list, in order.
  const rankModels = (provider: ModelOptionProvider, models: readonly string[]) => {
    if (!q || foldIncludes(provider.name, q) || foldIncludes(provider.slug, q)) {
      return [...models]
    }

    return fuzzyRank(models, q, modelSearchText).map(r => r.item)
  }

  const configured = providers.filter(p => (p.models ?? []).length > 0)

  const groups = configured.map(provider => ({
    provider,
    // Empty query: the backend's curated order, verbatim.
    models: rankModels(provider, provider.models ?? [])
  }))

  const hasMatches = groups.some(g => g.models.length > 0)

  // A typed id nothing lists: one row per configured provider, current
  // provider first, so the slug is one Enter away and remembered afterwards.
  // While the query still matches catalog rows the section stays out of the
  // way unless the user asked for it via "Add custom model…".
  const customSlug = offerCustom || !hasMatches ? customModelCandidate(search, configured) : null

  const customProviders = customSlug
    ? [...configured].sort(
        (a, b) =>
          Number(catalogProviderMatches(b, currentProvider)) - Number(catalogProviderMatches(a, currentProvider))
      )
    : []

  return (
    <>
      {groups.map(({ provider, models }) => {
        if (models.length === 0) {
          return null
        }

        const unavailable = new Set(provider.unavailable_models ?? [])
        const accountLimited = accountResetMs(provider) !== null

        return (
          <CommandGroup heading={<ProviderHeading provider={provider} />} key={provider.slug}>
            {provider.warning && (
              <div className="px-2 pb-2">
                <InlineNotice className="px-2.5 py-1.5 text-xs" kind="warning">
                  {provider.warning}
                </InlineNotice>
              </div>
            )}
            {models.map(model => {
              const isCurrent = model === currentModel && catalogProviderMatches(provider, currentProvider)
              const price = provider.pricing?.[model]
              const locked = unavailable.has(model)
              const resetMs = modelResetMs(provider, model)
              const resetLabel = resetMs === null ? null : formatReset(resetMs)
              const dimmed = !locked && (accountLimited || resetLabel !== null)

              return (
                <CommandItem
                  className={cn(
                    'flex items-center gap-2 pl-6 font-mono',
                    isCurrent &&
                      'bg-primary text-primary-foreground data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground',
                    locked && 'cursor-not-allowed opacity-45'
                  )}
                  disabled={locked}
                  key={`${provider.slug}:${model}`}
                  onSelect={() => {
                    if (!locked) {
                      onSelectModel(provider, model)
                    }
                  }}
                  value={`${provider.slug}:${model}`}
                >
                  <span className={cn('min-w-0 flex-1 truncate', dimmed && !isCurrent && 'text-muted-foreground')}>
                    <HighlightMatches foldSeparators query={search} text={model} />
                  </span>

                  {locked && (
                    <span className="shrink-0 text-[0.62rem] uppercase tracking-wide opacity-80">{copy.pro}</span>
                  )}
                  {resetLabel && (
                    <Badge
                      className="font-sans tabular-nums"
                      size="xs"
                      title={t.shell.modelMenu.modelLimitedTip(resetLabel)}
                      variant="warn"
                    >
                      {t.shell.modelMenu.modelResets(resetLabel)}
                    </Badge>
                  )}
                  <ModelPrice isCurrent={isCurrent} price={price} />
                </CommandItem>
              )
            })}

            {unavailable.size > 0 && (
              <div className="px-6 pb-2 pt-1 text-[0.62rem] leading-relaxed text-muted-foreground">
                {copy.proNeedsSubscription}
              </div>
            )}
          </CommandGroup>
        )
      })}

      {customSlug && customProviders.length > 0 && (
        <CommandGroup heading={copy.customModel} key="custom-model">
          {customProviders.map(provider => (
            <CommandItem
              className="flex items-center gap-2 pl-6 font-mono"
              key={`custom:${provider.slug}`}
              onSelect={() => onSelectCustomModel(provider, customSlug)}
              value={`custom:${provider.slug}:${customSlug}`}
            >
              <span className="min-w-0 flex-1 truncate">{customSlug}</span>
              <span className="shrink-0 text-[0.66rem] text-muted-foreground">{provider.name}</span>
            </CommandItem>
          ))}
        </CommandGroup>
      )}
    </>
  )
}

// Compact In/Out $/Mtok price tag, mirroring the CLI picker's price columns.
// Renders nothing when pricing is unavailable for the model.
function ModelPrice({ price, isCurrent }: { price?: ModelPricing; isCurrent: boolean }) {
  const { t } = useI18n()
  const copy = t.modelPicker

  if (!price || (!price.input && !price.output)) {
    return null
  }

  if (price.free) {
    return (
      <span className="shrink-0 inline-flex items-center gap-1.5">
        {typeof price.discount_percent === 'number' ? (
          <span
            className={cn(
              'rounded-sm px-1 py-0.5 text-[0.62rem] font-semibold',
              isCurrent ? 'bg-primary-foreground/20' : 'bg-amber-500/15 text-amber-700 dark:text-amber-400'
            )}
          >
            -{price.discount_percent}%
          </span>
        ) : null}
        <span
          className={cn(
            'shrink-0 rounded-sm px-1 py-0.5 text-[0.62rem] font-semibold uppercase tracking-wide',
            isCurrent ? 'bg-primary-foreground/20' : 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400'
          )}
        >
          {copy.free}
        </span>
      </span>
    )
  }

  const onSale = typeof price.discount_percent === 'number' && Boolean(price.was_input || price.was_output)

  return (
    <span
      className={cn(
        'shrink-0 inline-flex items-center gap-1.5 text-[0.66rem] tabular-nums',
        isCurrent ? 'text-primary-foreground/80' : 'text-muted-foreground'
      )}
      title={copy.priceTitle}
    >
      {onSale ? (
        <span
          className={cn(
            'rounded-sm px-1 py-0.5 text-[0.62rem] font-semibold',
            isCurrent ? 'bg-primary-foreground/20' : 'bg-amber-500/15 text-amber-700 dark:text-amber-400'
          )}
        >
          -{price.discount_percent}%
        </span>
      ) : null}
      <span>
        {price.input || '?'} / {price.output || '?'}
      </span>
      {onSale ? (
        <span
          className={cn(
            'line-through decoration-from-font opacity-70',
            isCurrent ? 'text-primary-foreground/60' : 'text-muted-foreground/80'
          )}
        >
          {copy.wasPrice} {price.was_input || '?'} / {price.was_output || '?'}
        </span>
      ) : null}
    </span>
  )
}

function LoadingResults() {
  return (
    <CommandGroup heading={<Skeleton className="h-3 w-32" />}>
      {Array.from({ length: 4 }, (_, rowIndex) => (
        <div className="rounded-sm py-1.5 pl-6 pr-2" key={rowIndex}>
          <Skeleton className={cn('h-5', rowIndex % 3 === 0 ? 'w-3/5' : rowIndex % 3 === 1 ? 'w-4/5' : 'w-1/2')} />
        </div>
      ))}
    </CommandGroup>
  )
}

function ProviderHeading({ provider }: { provider: ModelOptionProvider }) {
  const { t } = useI18n()
  const copy = t.modelPicker

  // Two different facts wear the same badge: `free_tier` is a signed-in Nous
  // account on the free plan; `free_tier_row` is the no-account route's own
  // row. Either way the user is on free inference, so say so. Never match the
  // route by name — the label is copy.
  const tierBadge =
    provider.free_tier === true || provider.free_tier_row === true ? (
      <span className="rounded-sm bg-emerald-500/15 px-1 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-emerald-600 dark:text-emerald-400">
        {copy.freeTier}
      </span>
    ) : provider.free_tier === false ? (
      <span className="rounded-sm bg-primary/15 px-1 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-primary">
        {copy.pro}
      </span>
    ) : null

  return (
    <span className="flex min-w-0 items-center gap-2">
      <span className="truncate">{provider.name}</span>
      <span className="font-mono text-xs font-normal normal-case tracking-normal text-muted-foreground">
        {provider.slug} · {provider.total_models ?? provider.models?.length ?? 0}
      </span>
      {tierBadge}
      <ProviderStatusChip provider={provider} />
    </span>
  )
}
