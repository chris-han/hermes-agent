import { catalogModelIssue, catalogModelMetadata, catalogModelSearchText, catalogSearchMatches, type ModelOptionProvider } from '@hermes/shared'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectSeparator, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { Plus, X } from '@/lib/icons'
import { isSubmitEnter } from '@/lib/ime'
import { normalize } from '@/lib/text'
import { cn } from '@/lib/utils'
import { addCustomModel, customModelSlug } from '@/store/custom-models'

import { CONTROL_TEXT } from './constants'
import { SearchableSelect } from './searchable-select'

// Radix <Select> renders a blank trigger when `value` matches no <SelectItem>.
// A custom model (e.g. one added via config that isn't in the provider's
// curated list) would vanish — surface the active value so it stays selectable.
export const withActive = (models: readonly string[], active: string): readonly string[] =>
  active && !models.includes(active) ? [active, ...models] : models

// Radix <Select> items cannot be typed into; an action row swaps the control
// for a text field. Model ids live under their own prefix so no slug — this
// feature admits any whitespace-free string — can alias the action's value.
const CUSTOM_ITEM = 'custom'
const MODEL_PREFIX = 'model:'

export const toItemValue = (model: string): string => (model ? MODEL_PREFIX + model : '')

/** The model id behind a Radix item value; `null` for the custom action row. */
export const fromItemValue = (item: string): string | null =>
  item.startsWith(MODEL_PREFIX) ? item.slice(MODEL_PREFIX.length) : null

interface ModelSelectProps {
  manualScope?: string
  'aria-label'?: string
  className?: string
  disabled?: boolean
  /** The provider's catalog row when known: the typed id is remembered under
   *  it and made visible in the pickers. Absent (provider not in the catalog)
   *  the id is still applied, just not remembered. */
  provider?: ModelOptionProvider
  /** Provider slug the id belongs to. Empty while none is picked. */
  providerSlug: string
  models: readonly string[]
  excludeUnavailable?: boolean
  allowedModels?: readonly string[] | null
  searchable?: boolean
  onValueChange: (model: string) => void
  value: string
}

/**
 * A model <Select> with a "Custom model…" row that turns into an Input, so a
 * slug the provider's catalog lacks can be typed in place. Every keystroke
 * reaches `onValueChange` (the parent's Apply/autosave reads live state); the
 * id is remembered as a custom model when the field commits (Enter/blur).
 */
export function ModelSelect({
  'aria-label': ariaLabel,
  manualScope,
  allowedModels,
  className,
  disabled = false,
  excludeUnavailable = false,
  models,
  onValueChange,
  provider,
  providerSlug,
  searchable = false,
  value
}: ModelSelectProps) {
  const { t, locale } = useI18n()
  const m = t.settings.model
  const [typing, setTyping] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const cloud = providerSlug === 'semantier-cloud'
  const eligible = cloud ? (allowedModels ?? provider?.eligible_models ?? []) : allowedModels

  const isUnavailable = (model: string) =>
    (eligible != null && !eligible.includes(model)) ||
    (excludeUnavailable && (provider?.unavailable_models ?? []).some(item => normalize(item) === normalize(model)))

  useEffect(() => {
    if (typing) {
      inputRef.current?.focus()
      inputRef.current?.select()
    }
  }, [typing])

  // A provider change swaps the catalog; drop back to the list for it.
  useEffect(() => {
    setTyping(false)
  }, [providerSlug])

  const remember = () => {
    const slug = customModelSlug(value)

    if (!slug || !providerSlug || isUnavailable(slug)) {
      return
    }

    if (slug !== value) {
      onValueChange(slug)
    }

    if (!models.includes(slug)) {
      addCustomModel(providerSlug, slug, provider, manualScope)
    }
  }

  if (typing) {
    return (
      <Input
        aria-label={ariaLabel}
        className={cn(CONTROL_TEXT, 'font-mono')}
        containerClassName={className}
        disabled={disabled}
        onBlur={remember}
        onChange={event => onValueChange(event.target.value)}
        onKeyDown={event => {
          if (isSubmitEnter(event)) {
            remember()
            setTyping(false)
          } else if (event.key === 'Escape') {
            setTyping(false)
          }
        }}
        placeholder={m.customModelPlaceholder}
        ref={inputRef}
        suffix={
          <Button
            aria-label={m.chooseFromList}
            className="pointer-events-auto -my-1.5 -mr-1.5 text-muted-foreground"
            onClick={() => setTyping(false)}
            size="icon-xs"
            type="button"
            variant="ghost"
          >
            <X />
          </Button>
        }
        value={value}
      />
    )
  }

  if (searchable || cloud) {
    return (
      <SearchableSelect
        aria-label={ariaLabel ?? m.model}
        className={className}
        disabled={disabled}
        filter={(item, query, keywords) => catalogSearchMatches([item, ...(keywords ?? [])].join(' '), query) ? 1 : 0}
        onChange={next => {
          const model = fromItemValue(next)

          if (model === null) {
            setTyping(true)
          } else {
            onValueChange(model)
          }
        }}
        options={[
          ...withActive(models.filter(model => !isUnavailable(model)), value).map(model => ({
            value: toItemValue(model), label: cloud ? `${catalogModelMetadata(provider, model)?.label ?? model}${catalogModelMetadata(provider, model)?.provider_display ? ` · ${catalogModelMetadata(provider, model)?.provider_display}` : ''}${catalogModelMetadata(provider, model)?.capabilities?.stream === true ? ` · ${locale === 'zh' ? '支持流式' : 'Streaming'}` : ''}${catalogModelIssue(provider, model) ? ` · ${catalogModelIssue(provider, model)} · ${locale === 'zh' ? '请刷新模型或检查供给配置' : 'Refresh models or check offer configuration'}` : ''}` : model,
            keywords: provider ? [catalogModelSearchText(provider, model)] : [model], disabled: isUnavailable(model) || !!(cloud && catalogModelIssue(provider, model))
          })),
          ...(eligible == null ? [{ value: CUSTOM_ITEM, label: m.customModel }] : [])
        ]}
        placeholder={cloud ? t.shell.modelMenu.search : m.model}
        value={toItemValue(value)}
      />
    )
  }

  return (
    <Select
      disabled={disabled}
      onValueChange={next => {
        const model = fromItemValue(next)

        if (model === null) {
          setTyping(true)
        } else {
          onValueChange(model)
        }
      }}
      value={toItemValue(value)}
    >
      <SelectTrigger aria-label={ariaLabel} className={cn(className, CONTROL_TEXT)}>
        <SelectValue placeholder={m.model} />
      </SelectTrigger>
      <SelectContent>
        {withActive(models.filter(model => !isUnavailable(model)), value).map(model => (
          <SelectItem disabled={isUnavailable(model)} key={model} value={toItemValue(model)}>
            {model}
          </SelectItem>
        ))}
        <SelectSeparator />
        <SelectItem className="text-muted-foreground" value={CUSTOM_ITEM}>
          <span className="inline-flex items-center gap-1.5">
            <Plus className="size-3" />
            {m.customModel}
          </span>
        </SelectItem>
      </SelectContent>
    </Select>
  )
}
