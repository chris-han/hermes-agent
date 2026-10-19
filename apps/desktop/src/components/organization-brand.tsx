import { useStore } from '@nanostores/react'

import semantierLogo from '@/assets/semantier-logo.png'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $desktopBranding } from '@/store/desktop-branding'

import { desktopBrandingName, type DesktopBranding } from '../../electron/desktop-branding-types'


export function SemantierAttribution() {
  const { t } = useI18n()

  return <p className="text-xs text-muted-foreground">{t.branding.attribution}</p>
}

export function OrganizationBrand({
  draft,
  compact = false,
  attribution = false
}: {
  draft?: DesktopBranding
  compact?: boolean
  attribution?: boolean
  showDefault?: boolean
}) {
  const { locale } = useI18n()
  const saved = useStore($desktopBranding)
  const value = draft ?? saved
  const name = desktopBrandingName(value, locale)

  return (
    <div
      className={cn('flex min-w-0 flex-col gap-2', compact ? '' : 'items-center text-center')}
      data-organization-brand=""
    >
      <div className={cn('flex min-w-0 gap-3', compact ? 'items-center' : 'flex-col items-center')}>
        <img
          alt=""
          className={cn('shrink-0 object-contain', compact ? 'size-7' : 'size-14')}
          src={value.logo ?? semantierLogo}
        />
        <span
          className={cn(
            'min-w-0 break-words [overflow-wrap:anywhere]',
            compact ? 'text-sm font-medium' : 'text-xl font-semibold'
          )}
        >
          {name}
        </span>
      </div>
      {attribution && <SemantierAttribution />}
    </div>
  )
}
