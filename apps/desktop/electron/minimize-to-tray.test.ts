import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import type { BrowserWindow } from 'electron'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

const native = vi.hoisted(() => ({
  app: { on: vi.fn(), quit: vi.fn(), dock: { show: vi.fn(), hide: vi.fn() } },
  ipc: new Map<string, (...args: any[]) => any>(),
  windows: [] as any[],
  trays: [] as any[],
  fail: false,
  confirmClose: vi.fn(() => 0)
}))

vi.mock('electron', () => ({
  app: native.app,
  dialog: { showMessageBoxSync: native.confirmClose },
  ipcMain: { handle: (name: string, fn: (...args: any[]) => any) => native.ipc.set(name, fn) },
  BrowserWindow: { getAllWindows: () => native.windows },
  Menu: { buildFromTemplate: (items: unknown[]) => items },
  nativeImage: { createFromPath: () => ({ isEmpty: () => false, resize: () => ({}) }) },
  Tray: class extends EventEmitter {
    destroyed = false
    menu: any[] = []
    constructor() {
      super()

      if (native.fail) {
        throw new Error('Unavailable tray')
      }

      native.trays.push(this)
    }
    setToolTip() {}
    setContextMenu(menu: any[]) {
      this.menu = menu
    }
    destroy() {
      this.destroyed = true
    }
    isDestroyed() {
      return this.destroyed
    }
  }
}))
vi.mock('./tray-host', () => ({ watchLinuxTrayHost: async () => () => {} }))

import { createMinimizeToTray } from './minimize-to-tray'

class Window extends EventEmitter {
  visible = true
  minimized = false
  destroyed = false
  skipped = false
  focused = false
  webContents = { send: vi.fn(), mainFrame: {} }
  isDestroyed() {
    return this.destroyed
  }
  isVisible() {
    return this.visible
  }
  isMinimized() {
    return this.minimized
  }
  setSkipTaskbar(on: boolean) {
    this.skipped = on
  }
  blur() {
    this.focused = false
  }
  hide() {
    // A real Win32 window does NOT release keyboard focus by being hidden
    // (#126570); only an explicit blur does.
    this.visible = false
    this.emit('hide')
  }
  show() {
    this.visible = true
    this.emit('show')
  }
  focus() {
    this.focused = true
  }
  showInactive() {
    this.visible = true
    this.emit('show')
  }
  restore() {
    this.minimized = false
    this.visible = true
    this.emit('restore')
  }
  minimize() {
    this.minimized = true
    this.emit('minimize')
  }
  close() {
    const event = { preventDefault: vi.fn() }
    this.emit('close', event)

    if (!event.preventDefault.mock.calls.length) {
      this.destroyed = true
      this.emit('closed')
    }

    return event
  }
}

let home: string
beforeEach(() => {
  home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-tray-'))
  native.windows = []
  native.trays = []
  native.fail = false
  vi.clearAllMocks()
  native.confirmClose.mockReturnValue(0)
})
afterEach(() => fs.rmSync(home, { recursive: true, force: true }))

function setup() {
  const main = new Window()
  const peer = new Window()
  native.windows.push(main, peer)
  let handoff = false

  const controller = createMinimizeToTray({
    preferencesPath: path.join(home, 'minimize-to-tray.json'),
    getIconPath: () => 'icon.png',
    restoreMainWindow: () => main.showInactive(),
    isQuittingForHandoff: () => handoff,
    log: vi.fn()
  })

  controller.registerWindow(main as unknown as BrowserWindow, { closeToTray: true })
  controller.registerWindow(peer as unknown as BrowserWindow)

  return {
    controller,
    main,
    peer,
    handoff: () => {
      handoff = true
    }
  }
}

function flushDeferredHide() {
  return new Promise<void>(resolve => setImmediate(resolve))
}

test.runIf(process.platform === 'win32').each([false, true])(
  'closing asks before quitting with minimize-to-tray enabled=%s',
  async enabled => {
    const { controller, main } = setup()
    await controller.setEnabled(enabled)
    expect(main.close().preventDefault).toHaveBeenCalledOnce()
    await answerClose(main, 'quit')
    expect(native.app.quit).toHaveBeenCalledOnce()
    expect(main.visible).toBe(true)
    expect(main.destroyed).toBe(false)

    controller.beginQuit()
    main.close()
    expect(main.destroyed).toBe(true)
    expect(native.confirmClose).not.toHaveBeenCalled()
  }
)

test.runIf(process.platform === 'win32').each([false, true])(
  'cancelling close and choosing background work with minimize-to-tray enabled=%s',
  async enabled => {
    const { controller, main } = setup()
    await controller.setEnabled(enabled)
    expect(main.close().preventDefault).toHaveBeenCalledOnce()
    await answerClose(main, 'cancel')
    expect(main.visible).toBe(true)
    expect(native.app.quit).not.toHaveBeenCalled()

    main.close()
    main.close()
    await answerClose(main, 'background')
    await flushDeferredHide()
    expect(native.confirmClose).not.toHaveBeenCalled()
    expect(main.visible).toBe(false)
    expect(main.destroyed).toBe(false)
    expect(controller.status().available).toBe(true)
    expect(JSON.parse(fs.readFileSync(path.join(home, 'minimize-to-tray.json'), 'utf8')).enabled).toBe(enabled)
  }
)

const originalPlatform = Object.getOwnPropertyDescriptor(process, 'platform')

function setPlatform(platform: string) {
  Object.defineProperty(process, 'platform', { value: platform })
}

function restorePlatform() {
  if (originalPlatform) {
    Object.defineProperty(process, 'platform', originalPlatform)
  }
}

test('tray hide defers past the minimize dispatch (#119252)', async () => {
  const { controller, main } = setup()
  await controller.start()
  await controller.setEnabled(true)

  // Hiding synchronously inside the minimize dispatch wedges the native
  // minimized flag on Windows; hide must run after dispatch returns.
  main.minimize()
  expect(main.visible).toBe(true)
  await flushDeferredHide()
  expect(main.visible).toBe(false)
})

test('Windows restore re-activates the window instead of showInactive (#119252)', async () => {
  setPlatform('win32')

  try {
    const { controller, main } = setup()
    await controller.start()
    await controller.setEnabled(true)

    main.minimize()
    await flushDeferredHide()
    controller.restore()
    expect(main.visible).toBe(true)
    expect(main.minimized).toBe(false)
    expect(main.skipped).toBe(false)
    expect(main.focused).toBe(true)
  } finally {
    restorePlatform()
  }
})

test('Windows restore of a tray-hidden window by any other path activates it (#127349)', async () => {
  setPlatform('win32')

  try {
    const { controller, main } = setup()
    await controller.start()
    await controller.setEnabled(true)

    main.minimize()
    await flushDeferredHide()

    const show = vi.spyOn(main, 'show')
    const focus = vi.spyOn(main, 'focus')

    // A relaunch, deep link or notification click restores without the tray.
    // restore() alone leaves a painted window that is not the OS foreground
    // window, so it drops all input. It needs show() then focus().
    main.restore()
    await flushDeferredHide()
    expect(show).toHaveBeenCalledOnce()
    expect(focus).toHaveBeenCalledOnce()
    expect(main.skipped).toBe(false)
    expect(show.mock.invocationCallOrder[0]).toBeLessThan(focus.mock.invocationCallOrder[0])
  } finally {
    restorePlatform()
  }
})

test('a restore of a window the tray never hid is left alone', async () => {
  setPlatform('win32')

  try {
    const { main } = setup()
    const show = vi.spyOn(main, 'show')
    const focus = vi.spyOn(main, 'focus')

    main.minimize()
    main.restore()
    await flushDeferredHide()
    expect(show).not.toHaveBeenCalled()
    expect(focus).not.toHaveBeenCalled()
  } finally {
    restorePlatform()
  }
})

test('Windows tray hide releases keyboard focus before hiding (#126570)', async () => {
  setPlatform('win32')

  try {
    const { controller, main } = setup()
    await controller.start()
    await controller.setEnabled(true)

    // The close-to-tray path hides a focused window; a real hidden Win32
    // window keeps the UI thread's keyboard focus (the fake above models
    // that: hide does not touch focus), so the shared hide path must blur
    // explicitly, before the hide itself -- after it, it no longer takes.
    main.focused = true
    const blur = vi.spyOn(main, 'blur')
    const hide = vi.spyOn(main, 'hide')
    main.close()
    await answerClose(main, 'background')
    expect(main.visible).toBe(false)
    expect(main.focused).toBe(false)
    expect(blur).toHaveBeenCalledOnce()
    expect(blur.mock.invocationCallOrder[0]).toBeLessThan(hide.mock.invocationCallOrder[0])
  } finally {
    restorePlatform()
  }
})

test('a restore before the deferred hide fires cancels the stale hide', async () => {
  const { controller, main } = setup()
  await controller.start()
  await controller.setEnabled(true)

  main.minimize()
  // User restores (taskbar/shortcut) before the deferred hide fires.
  main.restore()
  await flushDeferredHide()
  expect(main.visible).toBe(true)
  expect(main.minimized).toBe(false)
})

test('opt-in minimize and primary Close preserve windows while explicit Quit still exits', async () => {
  const { controller, main, peer } = setup()
  expect(await controller.start()).toEqual({ enabled: false, available: false })
  main.minimize()
  expect(main.visible).toBe(true)
  main.restore()
  await native.ipc.get('hermes:minimize-to-tray:set')!(null, true)
  expect(native.ipc.get('hermes:minimize-to-tray:get')!()).toEqual({ enabled: true, available: true })
  main.minimize()
  await flushDeferredHide()
  expect(main.destroyed).toBe(false)
  expect(main.visible).toBe(false)
  expect(peer.visible).toBe(true)

  if (process.platform === 'darwin') {
    expect(native.app.dock.hide).not.toHaveBeenCalled()
  }

  peer.minimize()
  await flushDeferredHide()
  expect(peer.visible).toBe(false)

  if (process.platform === 'darwin') {
    expect(native.app.dock.hide).toHaveBeenCalled()
  }

  if (process.platform === 'win32') {
    expect(main.skipped && peer.skipped).toBe(true)
  }

  native.trays[0].menu[0].click()
  expect(main.visible && peer.visible).toBe(true)
  expect(main.minimized || peer.minimized).toBe(false)
  expect(main.skipped || peer.skipped).toBe(false)
  peer.close()
  expect(peer.destroyed).toBe(true)
  main.minimize()
  native.trays[0].menu[2].click()
  expect(native.app.quit).toHaveBeenCalledOnce()
  // A cancelled guard doesn't call beginQuit; hide remains enabled.
  controller.restore()
  main.minimize()
  await flushDeferredHide()
  expect(main.destroyed).toBe(false)
  expect(main.visible).toBe(false)
  // X/Alt+F4 hides the primary, but an accepted explicit quit closes it.
  expect(main.close().preventDefault).toHaveBeenCalledOnce()
  expect(main.destroyed).toBe(false)
  expect(main.visible).toBe(false)
  expect(native.trays[0].destroyed).toBe(false)
  native.trays[0].menu[0].click()
  expect(main.visible).toBe(true)
  controller.beginQuit()
  expect(main.close().preventDefault).not.toHaveBeenCalled()
  expect(main.destroyed).toBe(true)
  native.app.on.mock.calls.find(([event]) => event === 'will-quit')![1]()
  expect(native.trays[0].destroyed).toBe(true)
})

test('linux reuses one tray when minimize-to-tray is toggled off and on (#126353)', async () => {
  setPlatform('linux')

  try {
    const { controller, main } = setup()
    await controller.start()
    await controller.setEnabled(true)
    expect(native.trays).toHaveLength(1)
    main.minimize()
    await flushDeferredHide()
    expect(main.visible).toBe(false)

    await controller.setEnabled(false)
    expect(main.visible).toBe(true)
    expect(native.trays).toHaveLength(1)
    expect(native.trays[0].destroyed).toBe(false)

    await controller.setEnabled(true)
    expect(native.trays).toHaveLength(1)
    expect(native.trays[0].destroyed).toBe(false)
    main.minimize()
    await flushDeferredHide()
    expect(main.visible).toBe(false)
    native.trays[0].menu[0].click()
    expect(main.visible).toBe(true)
  } finally {
    restorePlatform()
  }
})

test('persistence, disabling, failed tray creation, and handoff never strand hidden windows', async () => {
  const first = setup()
  await first.controller.start()
  await first.controller.setEnabled(true)
  first.main.minimize()
  await first.controller.setEnabled(false)
  expect(first.main.visible).toBe(true)
  // Linux parks the tray so a later enable can reuse it (#126353).
  expect(native.trays[0].destroyed).toBe(process.platform !== 'linux')
  await first.controller.setEnabled(true)
  const restarted = setup()
  expect(await restarted.controller.start()).toEqual({ enabled: true, available: true })
  restarted.handoff()
  restarted.main.close()
  expect(restarted.main.destroyed).toBe(true)
  const failed = setup()
  native.fail = true
  expect(await failed.controller.start()).toEqual({ enabled: true, available: false })
  failed.main.minimize()
  expect(failed.main.visible).toBe(true)
  failed.main.close()
  await flushDeferredHide()
  expect(failed.main.destroyed).toBe(process.platform !== 'win32')

  if (process.platform === 'win32') {
    expect(failed.main.visible).toBe(true)
    expect(await answerClose(failed.main, 'background')).toEqual({ ok: false, reason: 'background-unavailable' })
    expect(native.confirmClose).not.toHaveBeenCalled()
  }

  fs.writeFileSync(path.join(home, 'minimize-to-tray.json'), 'bad json')
  const disabled = setup()
  expect(await disabled.controller.start()).toEqual({ enabled: false, available: false })
  native.confirmClose.mockReturnValue(2)
  disabled.main.close()
  expect(disabled.main.destroyed).toBe(process.platform !== 'win32')
  expect(native.app.quit).not.toHaveBeenCalled()
})

const closeRequest = (win: Window) =>
  native.ipc.get('hermes:minimize-to-tray:close-request')!({
    sender: win.webContents,
    senderFrame: win.webContents.mainFrame
  })

const answerClose = (win: Window, action: string, id = closeRequest(win)?.id) =>
  native.ipc.get('hermes:minimize-to-tray:close-answer')!(
    { sender: win.webContents, senderFrame: win.webContents.mainFrame },
    id,
    action
  )

test('Windows close uses a scoped renderer choice, coalesces close clicks and cancels without mutation', async () => {
  setPlatform('win32')

  try {
    const { controller, main, peer } = setup()
    await controller.start()
    main.close()
    main.close()
    const request = closeRequest(main)
    expect(request.id).toEqual(expect.any(String))
    expect(main.webContents.send).toHaveBeenCalledWith('hermes:minimize-to-tray:close-requested', request)
    expect(main.webContents.send.mock.calls.filter(([event]) => event.endsWith('close-requested'))).toHaveLength(1)
    expect(closeRequest(peer)).toBeNull()
    expect(await answerClose(peer, 'quit', request.id)).toEqual({ ok: false, reason: 'invalid-request' })
    expect(await answerClose(main, 'quit', 'old-request')).toEqual({ ok: false, reason: 'invalid-request' })
    expect(await answerClose(main, 'unknown')).toEqual({ ok: false, reason: 'invalid-request' })
    expect(
      await native.ipc.get('hermes:minimize-to-tray:close-answer')!(
        { sender: main.webContents, senderFrame: {} },
        request.id,
        'quit'
      )
    ).toEqual({ ok: false, reason: 'invalid-request' })
    expect(await answerClose(main, 'cancel')).toEqual({ ok: true })
    expect(closeRequest(main)).toBeNull()
    expect(main.visible).toBe(true)
    expect(native.app.quit).not.toHaveBeenCalled()
    expect(native.confirmClose).not.toHaveBeenCalled()
  } finally {
    restorePlatform()
  }
})

test('Windows renderer background/quit choices retain lifecycle ownership and surface tray failure', async () => {
  setPlatform('win32')

  try {
    const { controller, main } = setup()
    await controller.setEnabled(false)
    main.close()
    native.fail = true
    expect(await answerClose(main, 'background')).toEqual({ ok: false, reason: 'background-unavailable' })
    expect(main.visible).toBe(true)
    expect(closeRequest(main)).toBeTruthy()
    native.fail = false
    const background = answerClose(main, 'background')
    expect(await answerClose(main, 'background')).toEqual({ ok: false, reason: 'busy' })
    expect(await background).toEqual({ ok: true })
    expect(main.visible).toBe(false)
    expect(closeRequest(main)).toBeNull()
    expect(JSON.parse(fs.readFileSync(path.join(home, 'minimize-to-tray.json'), 'utf8')).enabled).toBe(false)
    controller.restore()
    main.close()
    expect(await answerClose(main, 'quit')).toEqual({ ok: true })
    expect(native.app.quit).toHaveBeenCalledOnce()
    expect(native.confirmClose).not.toHaveBeenCalled()
  } finally {
    restorePlatform()
  }
})
