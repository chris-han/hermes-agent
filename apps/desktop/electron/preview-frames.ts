/** Private CDP routing for one embedded preview guest; no port is opened. */
import type { Debugger } from 'electron'

export interface PreviewFrame {
  id: string
  parentId?: string
  loaderId: string
  sessionId?: string
  contextId: string
}

type FrameTree = { frame: { id: string; parentId?: string; loaderId: string }; childFrames?: FrameTree[] }

export class PreviewFrames {
  private readonly sessions = new Map<string, string>()
  private readonly contexts = new Map<string, string>()
  private readonly contextNumbers = new Map<string, number>()
  private readonly pending = new Set<Promise<void>>()
  private started: Promise<void> | undefined
  private ownsDebugger = false
  private disposed = false
  private childFailure = false

  constructor(private readonly debugger_: Debugger) {}
  private key(sessionId?: string, frameId?: string) {
    return `${sessionId || ''}:${frameId || ''}`
  }

  private onMessage = (_event: unknown, method: string, params: any, sessionId?: string) => {
    if (this.disposed) {
      return
    }
    if (method === 'Target.attachedToTarget' && params.targetInfo?.type === 'iframe') {
      const child = String(params.sessionId)
      this.sessions.set(child, String(params.targetInfo.targetId))
      const work = this.initialize(child).catch(() => {
        this.sessions.delete(child)
        this.childFailure = true
      })
      this.pending.add(work)
      void work.finally(() => this.pending.delete(work))
    } else if (method === 'Target.detachedFromTarget') {
      const child = String(params.sessionId)
      this.sessions.delete(child)
      for (const key of this.contexts.keys()) {
        if (key.startsWith(`${child}:`)) {
          this.contexts.delete(key)
          this.contextNumbers.delete(key)
        }
      }
    } else if (method === 'Runtime.executionContextCreated' && params.context?.auxData?.isDefault) {
      const context = params.context
      const key = this.key(sessionId, context.auxData.frameId)
      this.contexts.set(key, context.uniqueId)
      this.contextNumbers.set(key, context.id)
    } else if (method === 'Runtime.executionContextDestroyed') {
      for (const [key, uniqueId] of this.contexts) {
        if (
          key.startsWith(`${sessionId || ''}:`) &&
          (uniqueId === params.executionContextUniqueId ||
            (params.executionContextUniqueId == null && this.contextNumbers.get(key) === params.executionContextId))
        ) {
          this.contexts.delete(key)
          this.contextNumbers.delete(key)
        }
      }
    } else if (method === 'Runtime.executionContextsCleared') {
      for (const key of this.contexts.keys()) {
        if (key.startsWith(`${sessionId || ''}:`)) {
          this.contexts.delete(key)
          this.contextNumbers.delete(key)
        }
      }
    }
  }
  private onDetach = () => {
    this.ownsDebugger = false
    this.started = undefined
    this.debugger_.removeListener('message', this.onMessage)
    this.debugger_.removeListener('detach', this.onDetach)
    this.contexts.clear()
    this.contextNumbers.clear()
    this.sessions.clear()
  }

  private async initialize(sessionId?: string) {
    await this.debugger_.sendCommand(
      'Target.setAutoAttach',
      { autoAttach: true, waitForDebuggerOnStart: false, flatten: true },
      sessionId
    )
    await this.debugger_.sendCommand('Runtime.enable', {}, sessionId)
    await this.debugger_.sendCommand('Page.enable', {}, sessionId)
  }

  private async start() {
    if (this.disposed) {
      throw new Error('PREVIEW_DEBUGGER_CLOSED')
    }
    if (!this.started) {
      if (this.debugger_.isAttached()) {
        throw new Error('PREVIEW_DEBUGGER_BUSY')
      }
      this.started = (async () => {
        this.debugger_.attach('1.3')
        this.ownsDebugger = true
        this.debugger_.on('message', this.onMessage)
        this.debugger_.on('detach', this.onDetach)
        try {
          await this.initialize()
        } catch (error) {
          this.dispose()
          throw new Error('PREVIEW_FRAME_UNAVAILABLE', { cause: error })
        }
      })()
    }
    await this.started
    while (this.pending.size) {
      await Promise.all([...this.pending])
    }
    if (this.childFailure) {
      throw new Error('PREVIEW_FRAME_UNAVAILABLE')
    }
    if (!this.ownsDebugger || this.disposed) {
      throw new Error('PREVIEW_DEBUGGER_CLOSED')
    }
  }

  async snapshot(): Promise<PreviewFrame[]> {
    await this.start()
    const frames: PreviewFrame[] = []
    for (const sessionId of [undefined, ...this.sessions.keys()]) {
      const result = (await this.debugger_.sendCommand('Page.getFrameTree', {}, sessionId)) as { frameTree: FrameTree }
      const visit = (tree: FrameTree) => {
        const contextId = this.contexts.get(this.key(sessionId, tree.frame.id))
        if (!contextId) {
          throw new Error('PREVIEW_FRAME_UNAVAILABLE')
        }
        frames.push({
          id: tree.frame.id,
          parentId: tree.frame.parentId,
          loaderId: tree.frame.loaderId,
          sessionId,
          contextId
        })
        for (const child of tree.childFrames || []) {
          visit(child)
        }
      }
      visit(result.frameTree)
    }
    if (!frames.some(frame => !frame.parentId)) {
      throw new Error('PANE_UNAVAILABLE')
    }
    return frames
  }

  async evaluate(frame: PreviewFrame, expression: string): Promise<unknown> {
    await this.start()
    const check = () => {
      if (this.contexts.get(this.key(frame.sessionId, frame.id)) !== frame.contextId) {
        throw new Error('STALE_FRAME')
      }
    }
    check()
    const result = await this.debugger_.sendCommand(
      'Runtime.evaluate',
      { expression, uniqueContextId: frame.contextId, awaitPromise: true, returnByValue: true },
      frame.sessionId
    )
    check()
    if (result.exceptionDetails) {
      throw new Error('PREVIEW_FRAME_SCRIPT_FAILED')
    }
    return result.result?.value
  }

  /** Dispatch to the guest's browser input pipeline. Electron's webview
   * sendInputEvent stops at OOPIF boundaries; CDP routes through Chromium's
   * compositor and still produces trusted events in the focused document. */
  async input(frame: PreviewFrame, event: any): Promise<void> {
    await this.start()
    const verify = () => {
      if (this.contexts.get(this.key(frame.sessionId, frame.id)) !== frame.contextId) {
        throw new Error('STALE_FRAME')
      }
    }
    verify()
    if (event?.type === 'mouseMove' || event?.type === 'mouseDown' || event?.type === 'mouseUp') {
      if (!Number.isFinite(event.x) || !Number.isFinite(event.y)) {
        throw new Error('INVALID_INPUT')
      }
      const type =
        event.type === 'mouseMove' ? 'mouseMoved' : event.type === 'mouseDown' ? 'mousePressed' : 'mouseReleased'
      await this.debugger_.sendCommand('Input.dispatchMouseEvent', {
        type,
        x: event.x,
        y: event.y,
        button: event.type === 'mouseMove' ? 'none' : 'left',
        clickCount: event.clickCount || 0
      })
    } else if (event?.type === 'keyDown' || event?.type === 'keyUp' || event?.type === 'char') {
      const key = event.keyCode
      if (typeof key !== 'string' || !key || key.length > 32) {
        throw new Error('INVALID_INPUT')
      }
      if (
        (await this.evaluate(
          frame,
          'document.hasFocus() && !!window.__hermesActHolder?.aimed && document.activeElement === window.__hermesActHolder.aimed && document.contains(window.__hermesActHolder.aimed)'
        )) !== true
      ) {
        throw new Error('FRAME_NOT_FOCUSED')
      }
      const modifierBits: Record<string, number> = { alt: 1, control: 2, meta: 4, shift: 8 }
      const modifiers = Array.isArray(event.modifiers)
        ? event.modifiers.reduce((value: number, name: string) => value | (modifierBits[name] || 0), 0)
        : 0
      const special: Record<string, number> = {
        Backspace: 8,
        Tab: 9,
        Enter: 13,
        Escape: 27,
        ' ': 32,
        End: 35,
        Home: 36,
        ArrowLeft: 37,
        ArrowUp: 38,
        ArrowRight: 39,
        ArrowDown: 40,
        Delete: 46
      }
      const singleCharacter = Array.from(key).length === 1
      const windowsVirtualKeyCode =
        special[key] || (singleCharacter && key.codePointAt(0)! < 128 ? key.toUpperCase().charCodeAt(0) : 0)
      if (event.type === 'char' && !singleCharacter && key !== 'Enter') {
        return
      }
      const type = event.type === 'keyDown' ? 'rawKeyDown' : event.type === 'keyUp' ? 'keyUp' : 'char'
      const text = event.type === 'char' ? (key === 'Enter' ? '\r' : key) : undefined
      const code = /^[a-z]$/i.test(key) ? `Key${key.toUpperCase()}` : key
      await this.debugger_.sendCommand('Input.dispatchKeyEvent', {
        type,
        key,
        code,
        windowsVirtualKeyCode,
        nativeVirtualKeyCode: windowsVirtualKeyCode,
        modifiers,
        ...(text === undefined ? {} : { text, unmodifiedText: text })
      })
    } else {
      throw new Error('INVALID_INPUT')
    }
    verify()
  }

  /** Map child-frame viewport coordinates into the guest viewport. A null
   * point only checks that the frame is visibly embedded. */
  async mapPoint(
    frame: PreviewFrame,
    frames: PreviewFrame[],
    point?: { x: number; y: number }
  ): Promise<{ visible: boolean; x: number; y: number }> {
    let current = frame
    let x = point?.x ?? 0
    let y = point?.y ?? 0
    const visited = new Set<string>()
    while (current.parentId) {
      if (visited.has(current.id)) {
        throw new Error('STALE_FRAME')
      }
      visited.add(current.id)
      const parent = frames.find(candidate => candidate.id === current.parentId)
      if (!parent) {
        throw new Error('STALE_FRAME')
      }
      const verify = () => {
        if (
          this.contexts.get(this.key(current.sessionId, current.id)) !== current.contextId ||
          this.contexts.get(this.key(parent.sessionId, parent.id)) !== parent.contextId
        ) {
          throw new Error('STALE_FRAME')
        }
      }
      verify()
      const owner = await this.debugger_.sendCommand('DOM.getFrameOwner', { frameId: current.id }, parent.sessionId)
      const node = await this.debugger_.sendCommand(
        'DOM.resolveNode',
        { backendNodeId: owner.backendNodeId },
        parent.sessionId
      )
      const objectId = node.object?.objectId
      if (!objectId) {
        throw new Error('STALE_FRAME')
      }
      let result: any
      try {
        result = await this.debugger_.sendCommand(
          'Runtime.callFunctionOn',
          {
            objectId,
            returnByValue: true,
            arguments: [{ value: point ? { x, y } : null }],
            functionDeclaration: `function(point) {
            var r = this.getBoundingClientRect();
            var sx = r.width / this.offsetWidth, sy = r.height / this.offsetHeight;
            var left = Math.max(0, r.left), top = Math.max(0, r.top), right = Math.min(innerWidth, r.right), bottom = Math.min(innerHeight, r.bottom);
            var visible = !!(this.isConnected && this.offsetWidth && this.offsetHeight && right > left && bottom > top);
            for (var e = this; e && visible; e = e.parentElement) {
              var s = getComputedStyle(e), t = s.transform;
              if (s.display === 'none' || s.visibility !== 'visible' || Number(s.opacity) === 0 || (s.clipPath && s.clipPath !== 'none')) visible = false;
              if (e !== this) {
                var box = e.getBoundingClientRect();
                if (/^(hidden|clip|scroll|auto)$/.test(s.overflowX)) { left = Math.max(left, box.left); right = Math.min(right, box.right); }
                if (/^(hidden|clip|scroll|auto)$/.test(s.overflowY)) { top = Math.max(top, box.top); bottom = Math.min(bottom, box.bottom); }
                if (right <= left || bottom <= top) visible = false;
              }
              if (t && t !== 'none') { try { var m = new DOMMatrixReadOnly(t); if (!m.is2D || Math.abs(m.b) > 0.0001 || Math.abs(m.c) > 0.0001 || m.a <= 0 || m.d <= 0) visible = false; } catch (_) { visible = false; } }
            }
            var px = r.left + (this.clientLeft + (point ? point.x : 0)) * sx;
            var py = r.top + (this.clientTop + (point ? point.y : 0)) * sy;
            if (point) {
              if (point.x < 0 || point.y < 0 || point.x >= this.clientWidth || point.y >= this.clientHeight || px < 0 || py < 0 || px >= innerWidth || py >= innerHeight || document.elementFromPoint(px, py) !== this) visible = false;
            }
            return { visible: visible, x: px, y: py };
          }`
          },
          parent.sessionId
        )
      } finally {
        await this.debugger_.sendCommand('Runtime.releaseObject', { objectId }, parent.sessionId).catch(() => {})
      }
      verify()
      const mapped = result.result?.value
      if (!mapped || !mapped.visible || !Number.isFinite(mapped.x) || !Number.isFinite(mapped.y)) {
        return { visible: false, x, y }
      }
      x = mapped.x
      y = mapped.y
      current = parent
    }
    return { visible: true, x, y }
  }

  dispose() {
    if (this.disposed) {
      return
    }
    this.disposed = true
    this.debugger_.removeListener('message', this.onMessage)
    this.debugger_.removeListener('detach', this.onDetach)
    if (this.ownsDebugger && this.debugger_.isAttached()) {
      this.debugger_.detach()
    }
    this.ownsDebugger = false
    this.contexts.clear()
    this.contextNumbers.clear()
    this.sessions.clear()
  }
}
