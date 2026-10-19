/** Device-local display preference; never authenticated organization identity. */
export interface DesktopBranding {
  version: 1
  /** English name; retains the existing persisted field for compatibility. */
  displayName: string
  displayNameZh?: string
  logo: string | null
}
export const DEFAULT_DESKTOP_BRANDING: DesktopBranding = { version: 1, displayName: '', displayNameZh: '', logo: null }

export function desktopBrandingName(value: DesktopBranding, locale: string): string {
  const english = value.displayName.trim()
  const chinese = value.displayNameZh?.trim() ?? ''
  return /^zh(?:-|$)/i.test(locale)
    ? chinese || english || '司序科技'
    : english || chinese || 'Semantier'
}
