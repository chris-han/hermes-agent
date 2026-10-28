import { useEffect, useRef, useState } from 'react'

import { OrganizationBrand } from '@/components/organization-brand'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { Monitor, Search } from '@/lib/icons'

import { type DesktopBranding, desktopBrandingName } from '../../../electron/desktop-branding-types'
import {
  type BrandShortcutResult,
  type ShortcutDestination,
  validBrandShortcutName
} from '../../../electron/windows-brand-shortcuts-types'

import { SectionHeading } from './primitives'

interface BrandingShortcutsProps {
  draft: DesktopBranding
  dirty: boolean
  disabled: boolean
  onApply: () => Promise<DesktopBranding | null>
  onBusy: (busy: boolean) => void
}

export function BrandingShortcuts({ draft, dirty, disabled, onApply, onBusy }: BrandingShortcutsProps) {
  const { t, locale } = useI18n()
  const c = t.branding
  const bridge = window.hermesDesktop?.branding?.shortcuts
  const editedName = useRef(false)
  const [name, setName] = useState(() => desktopBrandingName(draft, locale))
  const [selected, setSelected] = useState<ShortcutDestination[]>(['desktop'])
  const [running, setRunning] = useState(false)
  const [results, setResults] = useState<BrandShortcutResult[]>([])
  const [resultName, setResultName] = useState('')
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    if (!editedName.current) {
      setName(desktopBrandingName(draft, locale))
    }
  }, [draft, locale])

  if (!bridge?.available) {
    return null
  }

  const failed = results.filter(row => row.status === 'failed').map(row => row.destination)

  const label = (destination: ShortcutDestination) =>
    destination === 'desktop' ? c.shortcutDesktop : c.shortcutStartMenu

  const create = async (destinations: ShortcutDestination[]) => {
    setRunning(true)
    onBusy(true)
    setError(null)
    const chosenName = name

    try {
      const branding = dirty ? await onApply() : draft

      if (!branding) {
        setError(c.saveFailed)

        return
      }

      const next = await bridge.create({ name: chosenName, destinations, branding })

      setResultName(chosenName)
      setResults(previous => [...previous.filter(row => !destinations.includes(row.destination)), ...next])
    } catch {
      setError(c.shortcutError)
    } finally {
      setRunning(false)

      onBusy(false)
    }
  }

  return (
    <section aria-label={c.shortcutTitle} className="mt-8 border-t border-(--ui-stroke-secondary) pt-6">
      <SectionHeading icon={Monitor} title={c.shortcutTitle} />
      <p className="mb-4 text-sm text-muted-foreground">{c.shortcutScope}</p>
      <div className="grid min-w-0 gap-5 lg:grid-cols-2">
        <div className="grid min-w-0 gap-4">
          <label className="grid gap-2 text-sm" htmlFor="branding-shortcut-name">
            {c.shortcutName}
            <Input
              aria-invalid={!validBrandShortcutName(name)}
              disabled={disabled || running}
              id="branding-shortcut-name"
              maxLength={120}
              onChange={event => {
                editedName.current = true
                setName(event.target.value)
                setResults([])
                setError(null)
              }}
              value={name}
            />
          </label>
          {!validBrandShortcutName(name) && (
            <p className="text-sm text-destructive" role="alert">
              {c.shortcutInvalidName}
            </p>
          )}
          <fieldset className="grid gap-2" disabled={disabled || running}>
            <legend className="mb-2 text-sm">{c.shortcutDestinations}</legend>
            {(['desktop', 'startMenu'] as const).map(destination => {
              const Icon = destination === 'desktop' ? Monitor : Search

              return (
                <label className="flex cursor-pointer items-center gap-2 text-sm" key={destination}>
                  <input
                    checked={selected.includes(destination)}
                    onChange={event =>
                      setSelected(previous =>
                        event.target.checked
                          ? [...previous, destination]
                          : previous.filter(item => item !== destination)
                      )
                    }
                    type="checkbox"
                  />
                  <Icon className="size-4" />
                  {label(destination)}
                </label>
              )
            })}
          </fieldset>
        </div>
        <div aria-label={c.shortcutPreview} className="min-w-0">
          <OrganizationBrand compact draft={{ ...draft, displayName: name, displayNameZh: name }} showDefault />
          <p className="mt-2 text-xs text-muted-foreground">{c.shortcutSnapshot}</p>
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <Button
          disabled={disabled || running || !selected.length || !validBrandShortcutName(name)}
          onClick={() => void create(selected)}
          size="sm"
        >
          {running
            ? c.shortcutCreating
            : dirty
              ? c.shortcutApplyCreate
              : results.some(row => row.status === 'created' || row.status === 'updated')
                ? c.shortcutUpdate
                : c.shortcutCreate}
        </Button>
        {failed.length > 0 && (
          <Button disabled={disabled || running} onClick={() => void create(failed)} size="sm" variant="secondary">
            {c.shortcutRetry}
          </Button>
        )}
      </div>
      <div aria-busy={running} aria-live="polite" className="mt-3 grid gap-2 text-sm">
        {error && (
          <p className="text-destructive" role="alert">
            {error}
          </p>
        )}
        {results.map(row => (
          <div className="flex flex-wrap items-center gap-2" key={row.destination}>
            <span>
              {label(row.destination)} ·{' '}
              {row.status === 'created'
                ? c.shortcutCreated
                : row.status === 'updated'
                  ? c.shortcutUpdated
                  : row.status === 'conflict'
                    ? c.shortcutConflict
                    : c.shortcutFailed}
            </span>
            {(row.status === 'created' || row.status === 'updated') && (
              <Button
                disabled={running}
                onClick={() =>
                  void bridge.reveal({ name: resultName, destination: row.destination }).catch(() => {
                    setError(c.shortcutRevealFailed)
                  })
                }
                size="sm"
                variant="ghost"
              >
                {c.shortcutReveal}
              </Button>
            )}
          </div>
        ))}
      </div>
      <p className="mt-3 text-xs text-muted-foreground">{c.shortcutPin}</p>
    </section>
  )
}
