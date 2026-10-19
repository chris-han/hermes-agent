/** Device-local display preference; never authenticated organization identity. */
export interface DesktopBranding {
  version: 1
  displayName: string
  logo: string | null
}
export const DEFAULT_DESKTOP_BRANDING: DesktopBranding = { version: 1, displayName: '', logo: null }
