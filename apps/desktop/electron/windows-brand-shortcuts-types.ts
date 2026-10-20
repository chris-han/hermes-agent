import type { DesktopBranding } from './desktop-branding-types'

export type ShortcutDestination = 'desktop' | 'startMenu'
export interface BrandShortcutRequest {
  name: string
  destinations: ShortcutDestination[]
  branding: DesktopBranding
}
export interface BrandShortcutResult {
  destination: ShortcutDestination
  status: 'created' | 'updated' | 'conflict' | 'failed'
}
export interface BrandShortcutBridge {
  available: boolean
  create: (request: BrandShortcutRequest) => Promise<BrandShortcutResult[]>
  reveal: (request: { name: string; destination: ShortcutDestination }) => Promise<void>
}

export function validBrandShortcutName(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    Boolean(value.trim()) &&
    value === value.trim() &&
    value.length <= 120 &&
    !/[<>:"/\\|?*\x00-\x1f\x7f]/.test(value) &&
    !/[. ]$/.test(value) &&
    !/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(value)
  )
}
