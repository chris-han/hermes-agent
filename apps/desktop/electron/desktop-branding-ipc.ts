import { BrowserWindow, ipcMain, type IpcMainEvent, type IpcMainInvokeEvent } from 'electron'

import { createDesktopBrandingStore } from './desktop-branding'
import { DEFAULT_DESKTOP_BRANDING } from './desktop-branding-types'

export function installDesktopBranding(file: string) {
  const windows = new Set<BrowserWindow>()

  const store = createDesktopBrandingStore(file, value => {
    for (const win of windows) {
      if (!win.isDestroyed()) {
        win.webContents.send('hermes:branding:changed', value)
      }
    }
  })

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

  return {
    registerWindow(win: BrowserWindow) {
      windows.add(win)
      win.on('closed', () => windows.delete(win))
    }
  }
}
