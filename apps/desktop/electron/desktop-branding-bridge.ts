import { ipcRenderer } from 'electron'

import type { DesktopBranding } from './desktop-branding-types'
import type { BrandShortcutBridge } from './windows-brand-shortcuts-types'

/** Kept sandbox-safe: no Node/filesystem dependencies in the preload. */
export function createBrandingBridge() {
  return {
    shortcuts: {
      available: process.platform === 'win32',
      create: request => ipcRenderer.invoke('hermes:branding:shortcuts:create', request),
      reveal: request => ipcRenderer.invoke('hermes:branding:shortcuts:reveal', request)
    } satisfies BrandShortcutBridge,
    initial: ipcRenderer.sendSync('hermes:branding:initial') as DesktopBranding,
    get: (): Promise<DesktopBranding> => ipcRenderer.invoke('hermes:branding:get'),
    set: (value: DesktopBranding): Promise<DesktopBranding> => ipcRenderer.invoke('hermes:branding:set', value),
    onChanged(callback: (value: DesktopBranding) => void) {
      const listener = (_event: Electron.IpcRendererEvent, value: DesktopBranding) => callback(value)
      ipcRenderer.on('hermes:branding:changed', listener)

      return () => ipcRenderer.removeListener('hermes:branding:changed', listener)
    }
  }
}
