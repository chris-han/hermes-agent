import { afterEach, describe, expect, it, vi } from 'vitest'

import { ownedPreviewGuest, PreviewFrameAccess, registerPreviewFramesIpc } from './preview-frame-access'
import { PreviewFrames } from './preview-frames'

afterEach(() => vi.restoreAllMocks())

describe('preview frame ownership', () => {
  it('accepts only a live preview guest owned by the requesting renderer and partition', () => {
    const host = {} as never
    const partition = {} as never
    const guest = { isDestroyed: () => false, getType: () => 'webview', hostWebContents: host, session: partition }
    expect(ownedPreviewGuest(host, guest as never, partition)).toBe(true)
    expect(ownedPreviewGuest({} as never, guest as never, partition)).toBe(false)
    expect(ownedPreviewGuest(host, guest as never, {} as never)).toBe(false)
    expect(ownedPreviewGuest(host, { ...guest, isDestroyed: () => true } as never, partition)).toBe(false)
    expect(ownedPreviewGuest(host, { ...guest, getType: () => 'window' } as never, partition)).toBe(false)
  })
})

it('rejects references after a loader change and across guest instances', async () => {
  let loaderId = 'old'
  vi.spyOn(PreviewFrames.prototype, 'snapshot').mockImplementation(async () => [
    { id: 'root', contextId: 'context', loaderId }
  ])
  vi.spyOn(PreviewFrames.prototype, 'mapPoint').mockResolvedValue({ visible: true, x: 0, y: 0 })
  vi.spyOn(PreviewFrames.prototype, 'evaluate').mockResolvedValue('ok')
  vi.spyOn(PreviewFrames.prototype, 'dispose').mockImplementation(() => {})
  const first = new PreviewFrameAccess({} as never)
  const second = new PreviewFrameAccess({} as never)
  const token = (await first.list())[0].token
  await expect(second.run(token, 'x')).rejects.toThrow('STALE_FRAME')
  expect(await first.run(token, 'x')).toBe('ok')
  loaderId = 'new'
  await expect(first.run(token, 'x')).rejects.toThrow('STALE_FRAME')
  await expect(first.input(token, { type: 'char', keyCode: 'x' })).rejects.toThrow('STALE_FRAME')
  expect((await first.list())[0].token).not.toBe(token)
  first.dispose()
  second.dispose()
})

it('registers IPC before Electron is ready without creating the session', () => {
  const ipc = { handle: vi.fn() }
  const getPartition = vi.fn()
  registerPreviewFramesIpc(ipc as never, () => undefined, getPartition)
  expect(ipc.handle).toHaveBeenCalledOnce()
  expect(getPartition).not.toHaveBeenCalled()
})

it('rechecks visibility after asynchronous frame inspection before executing', async () => {
  let allowed = true
  vi.spyOn(PreviewFrames.prototype, 'snapshot').mockResolvedValue([
    { id: 'root', contextId: 'context', loaderId: 'loader' }
  ])
  vi.spyOn(PreviewFrames.prototype, 'mapPoint').mockImplementation(async () => ({ visible: true, x: 0, y: 0 }))
  const evaluate = vi.spyOn(PreviewFrames.prototype, 'evaluate').mockResolvedValue('secret')
  vi.spyOn(PreviewFrames.prototype, 'dispose').mockImplementation(() => {})
  const access = new PreviewFrameAccess({} as never, () => {
    if (!allowed) {
      throw new Error('PANE_UNAVAILABLE')
    }
  })
  const token = (await access.list())[0].token
  vi.mocked(PreviewFrames.prototype.mapPoint).mockImplementation(async () => {
    allowed = false
    return { visible: true, x: 0, y: 0 }
  })
  await expect(access.run(token, 'x')).rejects.toThrow('PANE_UNAVAILABLE')
  expect(evaluate).not.toHaveBeenCalled()
  access.dispose()
})

it('does not return frame content if the guest becomes hidden during execution', async () => {
  let allowed = true
  vi.spyOn(PreviewFrames.prototype, 'snapshot').mockResolvedValue([
    { id: 'root', contextId: 'context', loaderId: 'loader' }
  ])
  vi.spyOn(PreviewFrames.prototype, 'mapPoint').mockResolvedValue({ visible: true, x: 0, y: 0 })
  vi.spyOn(PreviewFrames.prototype, 'evaluate').mockImplementation(async () => {
    allowed = false
    return 'secret'
  })
  vi.spyOn(PreviewFrames.prototype, 'dispose').mockImplementation(() => {})
  const access = new PreviewFrameAccess({} as never, () => {
    if (!allowed) {
      throw new Error('PANE_UNAVAILABLE')
    }
  })
  const token = (await access.list())[0].token
  await expect(access.run(token, 'x')).rejects.toThrow('PANE_UNAVAILABLE')
  access.dispose()
})

it('disposes a guest attachment when the owned webview is destroyed', async () => {
  let handler: any
  let destroyed: (() => void) | undefined
  const partition = {} as never
  const sender = { mainFrame: {} }
  const guest = {
    id: 781234,
    debugger: {},
    hostWebContents: sender,
    session: partition,
    isDestroyed: () => false,
    getType: () => 'webview',
    once: (_event: string, callback: () => void) => {
      destroyed = callback
    }
  }
  const dispose = vi.spyOn(PreviewFrameAccess.prototype, 'dispose').mockImplementation(() => {})
  vi.spyOn(PreviewFrameAccess.prototype, 'list').mockResolvedValue([])
  registerPreviewFramesIpc(
    {
      handle: (_name: string, callback: unknown) => {
        handler = callback
      }
    } as never,
    () => guest as never,
    () => partition
  )
  await handler({ sender, senderFrame: sender.mainFrame }, { webContentsId: guest.id, operation: 'list' })
  expect(dispose).not.toHaveBeenCalled()
  expect(destroyed).toBeTypeOf('function')
  destroyed!()
  expect(dispose).toHaveBeenCalledOnce()
})
