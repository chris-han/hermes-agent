import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { app, BrowserWindow, nativeImage } from 'electron'

import { installDesktopBranding } from '../desktop-branding-ipc'
app.setPath('userData', process.argv[2])
const mode = process.argv[3]
const file = path.join(app.getPath('userData'), 'desktop-branding.v1.json')
const controller = installDesktopBranding(file)

const watchdog = setTimeout(() => {
  console.error('Branding fixture timeout')
  app.exit(1)
}, 30000)

async function run() {
  await app.whenReady()

  const windows = [0, 1].map(() => {
    const win = new BrowserWindow({
      show: false,
      webPreferences: {
        sandbox: true,
        contextIsolation: true,
        preload: path.join(path.dirname(process.argv[1]), 'preload.cjs')
      }
    })

    controller.registerWindow(win)

    return win
  })

  for (const win of windows) {
    await win.loadURL('data:text/html,<p>Real native branding acceptance</p>')
  }

  const [primary, peer] = windows
  const read = () => primary.webContents.executeJavaScript('window.hermesDesktop.branding.get()')

  if (mode === 'commit') {
    assert.equal((await read()).displayName, '')
    await peer.webContents.executeJavaScript(
      'window.changes=[]; window.hermesDesktop.branding.onChanged(value=>window.changes.push(value)); true'
    )
    // Real Electron decoder and a valid raster with alpha, 64 x 64.
    const image = nativeImage.createFromBitmap(Buffer.alloc(64 * 64 * 4, 128), { width: 64, height: 64 })
    const logo = 'data:image/png;base64,' + image.toPNG().toString('base64')
    await primary.webContents.executeJavaScript(
      `window.hermesDesktop.branding.set(${JSON.stringify({ version: 1, displayName: 'Native Brand', displayNameZh: '原生品牌', logo })})`
    )
    assert.equal(
      (await peer.webContents.executeJavaScript('window.hermesDesktop.branding.get()')).displayName,
      'Native Brand'
    )
    assert.equal(await peer.webContents.executeJavaScript('window.changes.length'), 1)
    assert.equal(
      (await peer.webContents.executeJavaScript('window.hermesDesktop.branding.get()')).displayNameZh,
      '原生品牌'
    )

    for (const bad of ['data:image/png;base64,PHN2Zz4=', 'data:image/svg+xml;base64,PHN2Zy8+']) {
      await assert.rejects(
        primary.webContents.executeJavaScript(
          `window.hermesDesktop.branding.set(${JSON.stringify({ version: 1, displayName: 'bad', logo: bad })})`
        )
      )
    }

    fs.mkdirSync(file + '.tmp')
    await assert.rejects(
      primary.webContents.executeJavaScript(
        'window.hermesDesktop.branding.set({version:1,displayName:"lost",logo:null})'
      )
    )
    assert.equal((await read()).displayName, 'Native Brand')
    fs.rmdirSync(file + '.tmp')
  } else if (mode === 'remove') {
    assert.equal((await read()).displayName, 'Native Brand')
    assert.equal((await read()).displayNameZh, '原生品牌')
    assert.ok((await read()).logo)
    await primary.webContents.executeJavaScript(
      'window.hermesDesktop.branding.set({version:1,displayName:"Native Brand",displayNameZh:"原生品牌",logo:null})'
    )
  } else if (mode === 'reset') {
    assert.deepEqual(await read(), { version: 1, displayName: 'Native Brand', displayNameZh: '原生品牌', logo: null })
    await primary.webContents.executeJavaScript(
      'window.hermesDesktop.branding.set({version:1,displayName:"",logo:null})'
    )
  } else {
    assert.deepEqual(await read(), { version: 1, displayName: '', displayNameZh: '', logo: null })
  }

  console.log(
    JSON.stringify({
      ok: true,
      platform: process.platform,
      mode,
      checks: 'real preload/IPC/raster/native disk/peer publication/cold restart'
    })
  )
  clearTimeout(watchdog)
  app.exit(0)
}

run().catch(error => {
  console.error(error)
  app.exit(1)
})
