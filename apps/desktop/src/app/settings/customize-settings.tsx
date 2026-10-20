import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { OrganizationBrand, SemantierAttribution } from '@/components/organization-brand'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { decodeBrandingImage } from '@/lib/branding-image'
import {
  Check,
  Download,
  FileImage,
  Info,
  MessageCircle,
  Monitor,
  Network,
  Palette,
  RefreshCw,
  Settings2,
  Upload,
  X
} from '@/lib/icons'
import {
  $brandingLoadFailed,
  $desktopBranding,
  loadDesktopBranding,
  saveDesktopBranding
} from '@/store/desktop-branding'

import { DEFAULT_DESKTOP_BRANDING, type DesktopBranding } from '../../../electron/desktop-branding-types'

import { BrandingShortcuts } from './branding-shortcuts'

import { NavLink, SectionHeading, SettingsContent } from './primitives'
import { SETTING_IDS, settingElementId } from './settings-manifest'

const PLACEMENTS = [
  ['welcome', MessageCircle],
  ['sidebar', Monitor],
  ['customize', Palette],
  ['about', Info],
  ['update', Download],
  ['install', Settings2],
  ['remote', Network],
  ['onboarding', FileImage]
] as const

export type BrandPlacement = (typeof PLACEMENTS)[number][0]

/** Pure presentation: shares identity fragments with production; never mounts live workflows. */
export function BrandingPreview({ draft, placement }: { draft: DesktopBranding; placement: BrandPlacement }) {
  const { t } = useI18n()
  const c = t.branding

  return (
    <section
      aria-label={c.preview}
      className="flex min-h-72 min-w-0 flex-col gap-6 rounded-lg border border-(--ui-stroke-secondary) bg-background p-5"
    >
      <p className="text-xs text-muted-foreground">Hermes · {c[placement]}</p>
      <div className="flex flex-1 flex-col justify-center gap-5">
        <OrganizationBrand
          attribution
          compact={placement === 'sidebar' || placement === 'customize'}
          draft={draft}
          showDefault
        />
        {placement === 'welcome' && (
          <p className="text-center text-sm text-muted-foreground">{t.intro.stock.none?.[0] ?? c.welcome}</p>
        )}
        {(placement === 'about' || placement === 'customize') && (
          <dl className="grid gap-2 text-sm">
            <dt>{c.organizationId}</dt>
            <dd className="text-muted-foreground">{c.unbound}</dd>
          </dl>
        )}
        {placement === 'update' && <p className="text-center text-sm">{t.settings.about.updates}</p>}
        {placement === 'install' && <p className="text-center text-sm">{t.install.settingUpTitle}</p>}
        {placement === 'remote' && (
          <p className="text-center text-sm text-muted-foreground">{t.install.remoteSetupDesc}</p>
        )}
        {placement === 'onboarding' && <p className="text-center text-sm">{c.onboarding}</p>}
      </div>
      <p className="text-xs text-muted-foreground">{c.sample}</p>
    </section>
  )
}

export function CustomizeSettings() {
  const { t } = useI18n()
  const c = t.branding
  const committed = useStore($desktopBranding)
  const loadFailed = useStore($brandingLoadFailed)
  const [draft, setDraft] = useState<DesktopBranding>(committed)
  const [placement, setPlacement] = useState<BrandPlacement>('welcome')
  const [decoding, setDecoding] = useState(false)
  const [saving, setSaving] = useState(false)
  const [shortcutBusy, setShortcutBusy] = useState(false)
  const busy = saving || shortcutBusy
  const [error, setError] = useState<'invalidImage' | 'saveFailed' | null>(null)
  const [saved, setSaved] = useState(false)
  const dirty = useRef(false)
  const generation = useRef(0)
  const fileInput = useRef<HTMLInputElement>(null)
  useEffect(() => {
    if (!dirty.current) {
      setDraft(committed)
    }
  }, [committed])
  useEffect(
    () => () => {
      generation.current++
    },
    []
  )

  const edit = (next: DesktopBranding) => {
    dirty.current = true
    setDraft(next)
    setSaved(false)
    setError(null)
  }

  const invalidateImage = () => {
    generation.current++
    setDecoding(false)
  }

  const upload = async (file: File) => {
    const token = ++generation.current
    setDecoding(true)
    setError(null)
    setSaved(false)

    try {
      const logo = await decodeBrandingImage(file)

      if (token === generation.current) {
        dirty.current = true
        setDraft(value => ({ ...value, logo }))
      }
    } catch {
      if (token === generation.current) {
        setError('invalidImage')
      }
    } finally {
      if (token === generation.current) {
        setDecoding(false)
      }
    }
  }

  const apply = async () => {
    invalidateImage()
    setSaving(true)
    setError(null)
    setSaved(false)

    try {
      const value = await saveDesktopBranding(draft)
      dirty.current = false
      setDraft(value)
      setSaved(true)
      return value
    } catch {
      setError('saveFailed')
      return null
    } finally {
      setSaving(false)
    }
  }

  const available = Boolean(window.hermesDesktop?.branding)

  return (
    <SettingsContent>
      <div id={settingElementId(SETTING_IDS.appearance.customize)}>
        <SectionHeading icon={Palette} title={c.title} />
        <p className="mb-6 text-sm text-muted-foreground">{c.scope}</p>
        {!available && (
          <p className="mb-4 text-sm" role="status">
            {c.nativeUnavailable}
          </p>
        )}
        {loadFailed && (
          <div className="mb-4 text-sm" role="alert">
            <p>{c.loadFailed}</p>
            <Button onClick={() => void loadDesktopBranding()} size="sm" variant="secondary">
              {c.retry}
            </Button>
          </div>
        )}
        <div className="grid min-w-0 gap-8 lg:grid-cols-2">
          <div className="flex min-w-0 flex-col gap-5">
            <label className="grid gap-2 text-sm" htmlFor="branding-display-name-zh">
              {c.nameZh}
              <Input
                disabled={busy}
                id="branding-display-name-zh"
                lang="zh"
                maxLength={120}
                onChange={event => edit({ ...draft, displayNameZh: event.target.value })}
                placeholder="司序科技"
                value={draft.displayNameZh ?? ''}
              />
            </label>
            <label className="grid gap-2 text-sm" htmlFor="branding-display-name">
              {c.nameEn}
              <Input
                disabled={busy}
                id="branding-display-name"
                lang="en"
                maxLength={120}
                onChange={event => edit({ ...draft, displayName: event.target.value })}
                placeholder="Semantier"
                value={draft.displayName}
              />
            </label>
            <div className="grid gap-2">
              <p className="text-sm">{c.logo}</p>
              <OrganizationBrand compact draft={draft} showDefault />
              <input
                accept="image/png,image/jpeg,image/webp"
                aria-label={c.logo}
                className="sr-only"
                disabled={busy}
                onChange={event => {
                  const file = event.target.files?.[0]
                  event.target.value = ''

                  if (file) {
                    void upload(file)
                  }
                }}
                ref={fileInput}
                type="file"
              />
              <div className="flex flex-wrap gap-2">
                <Button disabled={busy} onClick={() => fileInput.current?.click()} size="sm" variant="secondary">
                  <Upload className="size-4" />
                  {c.upload}
                </Button>
                <Button
                  disabled={busy || (!draft.logo && !decoding)}
                  onClick={() => {
                    invalidateImage()
                    edit({ ...draft, logo: null })
                  }}
                  size="sm"
                  variant="ghost"
                >
                  <X className="size-4" />
                  {c.remove}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">{c.imageHelp}</p>
            </div>
            <dl className="grid gap-1 text-sm">
              <dt>{c.organizationId}</dt>
              <dd className="text-muted-foreground">{c.unbound}</dd>
            </dl>
            <SemantierAttribution />
            <div>
              <SectionHeading icon={Monitor} title={c.placements} />
              <nav aria-label={c.placements} className="grid gap-1">
                {PLACEMENTS.map(([key, icon]) => (
                  <NavLink
                    active={key === placement}
                    icon={icon}
                    key={key}
                    label={c[key]}
                    onClick={() => setPlacement(key)}
                  />
                ))}
              </nav>
            </div>
            <p className="text-xs text-muted-foreground">{c.nativeIdentity}</p>
          </div>
          <div className="min-w-0">
            <SectionHeading icon={FileImage} title={c.preview} />
            <BrandingPreview draft={draft} placement={placement} />
          </div>
        </div>
        <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-(--ui-stroke-secondary) pt-4">
          <p
            aria-live="polite"
            className={error ? 'text-sm text-destructive' : 'text-sm text-muted-foreground'}
            role={error ? 'alert' : 'status'}
          >
            {error ? c[error] : decoding ? c.decoding : saving ? c.saving : saved ? c.saved : c.draft}
          </p>
          <div className="flex flex-wrap gap-2">
            <Button
              disabled={busy}
              onClick={() => {
                invalidateImage()
                edit({ ...DEFAULT_DESKTOP_BRANDING })
              }}
              size="sm"
              variant="ghost"
            >
              <RefreshCw className="size-4" />
              {c.reset}
            </Button>
            <Button disabled={busy || decoding || !available} onClick={() => void apply()} size="sm">
              <Check className="size-4" />
              {saving ? c.saving : c.apply}
            </Button>
          </div>
        </div>
        <BrandingShortcuts
          draft={draft}
          dirty={JSON.stringify(draft) !== JSON.stringify(committed)}
          disabled={saving || decoding || loadFailed || !available}
          onApply={apply}
          onBusy={setShortcutBusy}
        />
      </div>
    </SettingsContent>
  )
}
