const { contextBridge, ipcRenderer } = require('electron')
contextBridge.exposeInMainWorld('hermesDesktop', { previewFrames: payload => ipcRenderer.invoke('hermes:preview-frames', payload), setPreviewGuestHidden: (webContentsId, hidden) => ipcRenderer.send('hermes:preview-guest-hidden', { webContentsId, hidden }) })
