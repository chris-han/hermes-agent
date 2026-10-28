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
    !/[<>:"/\\|?*]/.test(value) &&
    !Array.from(value).some(character => character.charCodeAt(0) < 32 || character.charCodeAt(0) === 127) &&
    !/[. ]$/.test(value) &&
    !/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(value)
  )
}
