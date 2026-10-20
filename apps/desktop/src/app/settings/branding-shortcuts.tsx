import { useEffect, useRef, useState } from 'react'

import { OrganizationBrand } from '@/components/organization-brand'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { Monitor, Search } from '@/lib/icons'

import { desktopBrandingName, type DesktopBranding } from '../../../electron/desktop-branding-types'
import {
  validBrandShortcutName,
  type BrandShortcutResult,
  type ShortcutDestination
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
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])
  useEffect(() => {
    if (!editedName.current) {
      setName(desktopBrandingName(draft, locale))
    }
  }, [draft.displayName, draft.displayNameZh, locale])

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
        if (alive.current) {
          setError(c.saveFailed)
        }
        return
      }
      const next = await bridge.create({ name: chosenName, destinations, branding })
      if (alive.current) {
        setResultName(chosenName)
        setResults(previous => [...previous.filter(row => !destinations.includes(row.destination)), ...next])
      }
    } catch {
      if (alive.current) {
        setError(c.shortcutError)
      }
    } finally {
      if (alive.current) {
        setRunning(false)
      }
      onBusy(false)
    }
  }

  return (
    <section className="mt-8 border-t border-(--ui-stroke-secondary) pt-6" aria-label={c.shortcutTitle}>
      <SectionHeading icon={Monitor} title={c.shortcutTitle} />
      <p className="mb-4 text-sm text-muted-foreground">{c.shortcutScope}</p>
      <div className="grid min-w-0 gap-5 lg:grid-cols-2">
        <div className="grid min-w-0 gap-4">
          <label className="grid gap-2 text-sm" htmlFor="branding-shortcut-name">
            {c.shortcutName}
            <Input
              id="branding-shortcut-name"
              maxLength={120}
              value={name}
              disabled={disabled || running}
              aria-invalid={!validBrandShortcutName(name)}
              onChange={event => {
                editedName.current = true
                setName(event.target.value)
                setResults([])
                setError(null)
              }}
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
                <label key={destination} className="flex cursor-pointer items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={selected.includes(destination)}
                    onChange={event =>
                      setSelected(previous =>
                        event.target.checked
                          ? [...previous, destination]
                          : previous.filter(item => item !== destination)
                      )
                    }
                  />
                  <Icon className="size-4" />
                  {label(destination)}
                </label>
              )
            })}
          </fieldset>
        </div>
        <div className="min-w-0" aria-label={c.shortcutPreview}>
          <OrganizationBrand compact draft={{ ...draft, displayName: name, displayNameZh: name }} showDefault />
          <p className="mt-2 text-xs text-muted-foreground">{c.shortcutSnapshot}</p>
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        <Button
          size="sm"
          disabled={disabled || running || !selected.length || !validBrandShortcutName(name)}
          onClick={() => void create(selected)}
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
          <Button size="sm" variant="secondary" disabled={disabled || running} onClick={() => void create(failed)}>
            {c.shortcutRetry}
          </Button>
        )}
      </div>
      <div className="mt-3 grid gap-2 text-sm" aria-live="polite" aria-busy={running}>
        {error && (
          <p role="alert" className="text-destructive">
            {error}
          </p>
        )}
        {results.map(row => (
          <div key={row.destination} className="flex flex-wrap items-center gap-2">
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
                size="sm"
                variant="ghost"
                disabled={running}
                onClick={() =>
                  void bridge.reveal({ name: resultName, destination: row.destination }).catch(() => {
                    if (alive.current) {
                      setError(c.shortcutRevealFailed)
                    }
                  })
                }
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
