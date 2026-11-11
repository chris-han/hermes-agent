import { app, BrowserWindow, ipcMain, session, webContents } from 'electron'
import { registerPreviewFramesIpc } from '../electron/preview-frame-access.ts'
import { notePreviewGuestHidden } from '../electron/preview-guest-offscreen.ts'

app.commandLine.appendSwitch('site-per-process')
void app.whenReady().then(async () => {
  ipcMain.on('hermes:preview-guest-hidden', (event, { webContentsId, hidden }) => notePreviewGuestHidden(event.sender, webContents.fromId(webContentsId), hidden))
  registerPreviewFramesIpc(ipcMain, id => webContents.fromId(id) || undefined, () => session.fromPartition('persist:hermes-preview'))
  const win = new BrowserWindow({ width: 1100, height: 850, webPreferences: { webviewTag: true, contextIsolation: true, preload: process.env.PREVIEW_FIXTURE_PRELOAD } })
  await win.loadURL(process.env.PREVIEW_FIXTURE_HOST)
})
