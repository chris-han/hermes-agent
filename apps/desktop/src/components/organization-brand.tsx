import { useStore } from '@nanostores/react'

import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $desktopBranding } from '@/store/desktop-branding'

import type { DesktopBranding } from '../../electron/desktop-branding-types'

import { BrandMark } from './brand-mark'

export function SemantierAttribution() {
  const { t } = useI18n()

  return <p className="text-xs text-muted-foreground">{t.branding.attribution}</p>
}

export function OrganizationBrand({
  draft,
  compact = false,
  attribution = false,
  showDefault = false
}: {
  draft?: DesktopBranding
  compact?: boolean
  attribution?: boolean
  showDefault?: boolean
}) {
  const saved = useStore($desktopBranding)
  const value = draft ?? saved

  if (!showDefault && !value.displayName && !value.logo) {
    return attribution ? <SemantierAttribution /> : null
  }

  return (
    <div
      className={cn('flex min-w-0 flex-col gap-2', compact ? '' : 'items-center text-center')}
      data-organization-brand=""
    >
      <div className={cn('flex min-w-0 gap-3', compact ? 'items-center' : 'flex-col items-center')}>
        {value.logo ? (
          <img alt="" className={cn('shrink-0 object-contain', compact ? 'size-7' : 'size-14')} src={value.logo} />
        ) : value.displayName ? (
          <span
            aria-hidden="true"
            className={cn(
              'inline-flex shrink-0 items-center justify-center rounded-md bg-(--ui-bg-tertiary) font-medium',
              compact ? 'size-7 text-xs' : 'size-14 text-xl'
            )}
          >
            {Array.from(value.displayName).slice(0, 2).join('')}
          </span>
        ) : (
          <BrandMark className={compact ? 'size-7' : 'size-14'} />
        )}
        <span
          className={cn(
            'min-w-0 break-words [overflow-wrap:anywhere]',
            compact ? 'text-sm font-medium' : 'text-xl font-semibold'
          )}
        >
          {value.displayName || 'Hermes'}
        </span>
      </div>
      {attribution && <SemantierAttribution />}
    </div>
  )
}
