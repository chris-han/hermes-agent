import { randomUUID } from 'node:crypto'

import type { IpcMain, Session, WebContents } from 'electron'

import { type PreviewFrame, PreviewFrames } from './preview-frames'
import { isPreviewGuestHidden } from './preview-guest-offscreen'

/** Only a renderer's own preview guest may be addressed. Session/tab selection
 * remains in the renderer's existing owner-checked preview registry. */
export function ownedPreviewGuest(
  sender: WebContents,
  guest: WebContents | null | undefined,
  partition: Electron.Session
): guest is WebContents {
  return (
    !!guest &&
    !guest.isDestroyed() &&
    guest.getType() === 'webview' &&
    guest.hostWebContents === sender &&
    guest.session === partition
  )
}

export class PreviewFrameAccess {
  private readonly frames: PreviewFrames
  private readonly handles = new Map<string, PreviewFrame>()
  private readonly tokens = new Map<string, string>()

  constructor(
    debugger_: Electron.Debugger,
    private readonly assertAllowed: () => void = () => {}
  ) {
    this.frames = new PreviewFrames(debugger_)
  }

  private key(frame: PreviewFrame) {
    return `${frame.sessionId || ''}:${frame.id}:${frame.loaderId}:${frame.contextId}`
  }

  async list(): Promise<{ token: string }[]> {
    const frames = await this.frames.snapshot()
    this.assertAllowed()
    const current = new Set(frames.map(frame => this.key(frame)))
    for (const [key, token] of this.tokens) {
      if (!current.has(key)) {
        this.tokens.delete(key)
        this.handles.delete(token)
      }
    }
    const result: { token: string }[] = []
    for (const frame of frames) {
      if (!(await this.frames.mapPoint(frame, frames)).visible) {
        continue
      }
      this.assertAllowed()
      const key = this.key(frame)
      let token = this.tokens.get(key)
      if (!token) {
        token = randomUUID()
        this.tokens.set(key, token)
      }
      this.handles.set(token, frame)
      result.push({ token })
    }
    this.assertAllowed()
    return result
  }

  private resolve(token: string): PreviewFrame {
    const frame = this.handles.get(token)
    if (!frame) {
      throw new Error('STALE_FRAME')
    }
    return frame
  }

  async run(token: string, expression: string) {
    const frame = this.resolve(token)
    const snapshot = await this.frames.snapshot()
    if (!snapshot.some(candidate => this.key(candidate) === this.key(frame))) {
      throw new Error('STALE_FRAME')
    }
    if (!(await this.frames.mapPoint(frame, snapshot)).visible) {
      throw new Error('FRAME_NOT_VISIBLE')
    }
    this.assertAllowed()
    const result = await this.frames.evaluate(frame, expression)
    this.assertAllowed()
    return result
  }

  async input(token: string, event: any, point?: { x: number; y: number }) {
    const frame = this.resolve(token)
    const snapshot = await this.frames.snapshot()
    if (!snapshot.some(candidate => this.key(candidate) === this.key(frame))) {
      throw new Error('STALE_FRAME')
    }
    if (event?.type === 'mouseDown') {
      if (!Number.isFinite(point?.x) || !Number.isFinite(point?.y)) {
        throw new Error('INVALID_POINT')
      }
      const mapped = await this.frames.mapPoint(frame, snapshot, point)
      if (!mapped.visible || Math.abs(mapped.x - event.x) > 1 || Math.abs(mapped.y - event.y) > 1) {
        throw new Error('STALE_FRAME')
      }
    }
    this.assertAllowed()
    await this.frames.input(frame, event)
    this.assertAllowed()
  }

  async mapPoint(token: string, point: { x: number; y: number }) {
    if (!Number.isFinite(point?.x) || !Number.isFinite(point?.y)) {
      throw new Error('INVALID_POINT')
    }
    const frame = this.resolve(token)
    const snapshot = await this.frames.snapshot()
    if (!snapshot.some(candidate => this.key(candidate) === this.key(frame))) {
      throw new Error('STALE_FRAME')
    }
    const result = await this.frames.mapPoint(frame, snapshot, point)
    this.assertAllowed()
    return result
  }

  dispose() {
    this.handles.clear()
    this.tokens.clear()
    this.frames.dispose()
  }
}

/** IPC remains private to the embedding renderer and its own preview guest. */
export function registerPreviewFramesIpc(
  ipc: IpcMain,
  fromId: (id: number) => WebContents | undefined,
  getPartition: () => Session
) {
  const accessByGuest = new WeakMap<WebContents, PreviewFrameAccess>()
  ipc.handle('hermes:preview-frames', async (event, payload) => {
    const guest = fromId(Number(payload?.webContentsId))
    if (event.senderFrame !== event.sender.mainFrame || !ownedPreviewGuest(event.sender, guest, getPartition())) {
      throw new Error('WRONG_OWNER')
    }
    const assertAllowed = () => {
      if (!ownedPreviewGuest(event.sender, guest, getPartition())) {
        throw new Error('WRONG_OWNER')
      }
      if (isPreviewGuestHidden(guest)) {
        throw new Error('PANE_UNAVAILABLE')
      }
    }
    if (payload?.operation !== 'dispose') {
      assertAllowed()
    }
    let access = accessByGuest.get(guest)
    if (payload?.operation === 'dispose') {
      access?.dispose()
      accessByGuest.delete(guest)
      return
    }
    if (!access) {
      access = new PreviewFrameAccess(guest.debugger, assertAllowed)
      accessByGuest.set(guest, access)
      const ownedAccess = access
      guest.once('destroyed', () => {
        ownedAccess.dispose()
        accessByGuest.delete(guest)
      })
    }
    try {
      if (payload?.operation === 'list') {
        return await access.list()
      }
      if (typeof payload?.token !== 'string') {
        throw new Error('STALE_FRAME')
      }
      if (payload.operation === 'run' && typeof payload.code === 'string') {
        return await access.run(payload.token, payload.code)
      }
      if (payload.operation === 'mapPoint') {
        return await access.mapPoint(payload.token, payload.point)
      }
      if (payload.operation === 'input') {
        return await access.input(payload.token, payload.event, payload.point)
      }
      throw new Error('INVALID_PREVIEW_OPERATION')
    } catch (error) {
      // A later list may safely start a new attachment after a transient CDP
      // setup failure; all previous frame handles are retired with it.
      if (error instanceof Error && error.message === 'PREVIEW_FRAME_UNAVAILABLE') {
        access.dispose()
        accessByGuest.delete(guest)
      }
      throw error
    }
  })
}
