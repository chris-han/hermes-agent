import { BrowserWindow, ipcMain, type IpcMainEvent, type IpcMainInvokeEvent } from 'electron'

import { createDesktopBrandingStore } from './desktop-branding'
import { DEFAULT_DESKTOP_BRANDING } from './desktop-branding-types'
import { defaultBrandShortcutService } from './windows-brand-shortcuts'
import type { BrandShortcutRequest, ShortcutDestination } from './windows-brand-shortcuts-types'

export function installDesktopBranding(file: string, createShortcuts = defaultBrandShortcutService) {
  const windows = new Set<BrowserWindow>()

  const store = createDesktopBrandingStore(file, value => {
    for (const win of windows) {
      if (!win.isDestroyed()) {
        win.webContents.send('hermes:branding:changed', value)
      }
    }
  })

  const shortcuts = process.platform === 'win32' ? createShortcuts(() => store.get()) : null

  const allowed = (event: IpcMainEvent | IpcMainInvokeEvent) => {
    const win = BrowserWindow.fromWebContents(event.sender)

    return win && windows.has(win) && event.senderFrame === event.sender.mainFrame
  }

  const assertSender = (event: IpcMainInvokeEvent) => {
    if (!allowed(event)) {
      throw new Error('Untrusted branding sender')
    }
  }

  ipcMain.on('hermes:branding:initial', event => {
    event.returnValue = allowed(event) ? store.initial() : DEFAULT_DESKTOP_BRANDING
  })
  ipcMain.handle('hermes:branding:get', event => {
    assertSender(event)

    return store.get()
  })
  ipcMain.handle('hermes:branding:set', (event, value: unknown) => {
    assertSender(event)

    return store.set(value)
  })

  ipcMain.handle('hermes:branding:shortcuts:create', (event, request: BrandShortcutRequest) => {
    assertSender(event)

    if (!shortcuts) {
      throw new Error('Windows shortcuts unavailable')
    }

    return shortcuts.create(request)
  })
  ipcMain.handle(
    'hermes:branding:shortcuts:reveal',
    (event, request: { name: string; destination: ShortcutDestination }) => {
      assertSender(event)

      if (!shortcuts) {
        throw new Error('Windows shortcuts unavailable')
      }

      shortcuts.reveal(request?.name, request?.destination)
    }
  )

  return {
    registerWindow(win: BrowserWindow) {
      windows.add(win)
      win.on('closed', () => windows.delete(win))
    }
  }
}
