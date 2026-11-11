import { EventEmitter } from 'node:events'

import { describe, expect, it, vi } from 'vitest'

import { PreviewFrames } from './preview-frames'

class Debugger extends EventEmitter {
  attached = false
  attach = vi.fn(() => {
    this.attached = true
  })
  detach = vi.fn(() => {
    this.attached = false
    this.emit('detach', {}, 'detach')
  })
  isAttached = () => this.attached
  sendCommand = vi.fn(async (method: string, _params?: unknown, session?: string) => {
    if (method === 'Target.setAutoAttach' && !session) {
      this.emit(
        'message',
        {},
        'Target.attachedToTarget',
        { sessionId: 'child-session', targetInfo: { targetId: 'child', type: 'iframe' } },
        ''
      )
    }

    if (method === 'Runtime.enable') {
      const id = session ? 'child' : 'top'
      this.emit(
        'message',
        {},
        'Runtime.executionContextCreated',
        { context: { id: 1, uniqueId: `${id}-context`, auxData: { frameId: id, isDefault: true } } },
        session || ''
      )
    }

    if (method === 'Page.getFrameTree') {
      const id = session ? 'child' : 'top'

      return { frameTree: { frame: { id, loaderId: `${id}-loader`, ...(session ? { parentId: 'top' } : {}) } } }
    }

    if (method === 'Runtime.evaluate') {
      return { result: { value: String((_params as any)?.expression).startsWith('document.hasFocus()') ? true : 'ok' } }
    }

    return {}
  })
}

describe('preview frame debugger', () => {
  it('routes child execution to its own session and rejects replaced contexts', async () => {
    const dbg = new Debugger()
    const frames = new PreviewFrames(dbg as never)
    const snapshot = await frames.snapshot()
    expect(snapshot.map(frame => frame.id)).toEqual(['top', 'child'])
    const child = snapshot[1]
    expect(await frames.evaluate(child, '1')).toBe('ok')
    expect(dbg.sendCommand).toHaveBeenCalledWith(
      'Runtime.evaluate',
      expect.objectContaining({ uniqueContextId: 'child-context' }),
      'child-session'
    )
    dbg.emit(
      'message',
      {},
      'Runtime.executionContextDestroyed',
      { executionContextUniqueId: 'child-context', executionContextId: 1 },
      'child-session'
    )
    await expect(frames.evaluate(child, '1')).rejects.toThrow('STALE_FRAME')
    frames.dispose()
    expect(dbg.detach).toHaveBeenCalledOnce()
    expect(dbg.listenerCount('message')).toBe(0)
  })

  it('retires a context when Chromium reports only its numeric id', async () => {
    const dbg = new Debugger()
    const frames = new PreviewFrames(dbg as never)
    const child = (await frames.snapshot())[1]
    dbg.emit('message', {}, 'Runtime.executionContextDestroyed', { executionContextId: 1 }, 'child-session')
    await expect(frames.evaluate(child, '1')).rejects.toThrow('STALE_FRAME')
    frames.dispose()
  })

  it('sends trusted input through the guest root and checks frame focus', async () => {
    const dbg = new Debugger()
    const frames = new PreviewFrames(dbg as never)
    const child = (await frames.snapshot())[1]
    await frames.input(child, { type: 'mouseDown', x: 23, y: 42, button: 'left', clickCount: 1 })
    expect(dbg.sendCommand).toHaveBeenCalledWith(
      'Input.dispatchMouseEvent',
      expect.objectContaining({ type: 'mousePressed', x: 23, y: 42, button: 'left' })
    )
    await frames.input(child, { type: 'keyDown', keyCode: 'a', modifiers: ['control'] })
    expect(dbg.sendCommand).toHaveBeenCalledWith(
      'Input.dispatchKeyEvent',
      expect.objectContaining({ type: 'rawKeyDown', key: 'a', windowsVirtualKeyCode: 65, modifiers: 2 })
    )
    frames.dispose()
  })

  it('refuses keyboard input after focus leaves the targeted control', async () => {
    const dbg = new Debugger()
    const original = dbg.sendCommand.getMockImplementation()!
    dbg.sendCommand.mockImplementation(async (method, params, session) => {
      if (method === 'Runtime.evaluate') {
        return { result: { value: false } }
      }

      return original(method, params, session)
    })
    const frames = new PreviewFrames(dbg as never)
    const child = (await frames.snapshot())[1]
    await expect(frames.input(child, { type: 'char', keyCode: 'x' })).rejects.toThrow('FRAME_NOT_FOCUSED')
    expect(dbg.sendCommand).not.toHaveBeenCalledWith('Input.dispatchKeyEvent', expect.anything())
    frames.dispose()
  })

  it('reports child-session initialization failures instead of returning a partial tree', async () => {
    const dbg = new Debugger()
    const original = dbg.sendCommand.getMockImplementation()!
    dbg.sendCommand.mockImplementation(async (method, params, session) => {
      if (method === 'Runtime.enable' && session === 'child-session') {
        throw new Error('child failed')
      }

      return original(method, params, session)
    })
    const frames = new PreviewFrames(dbg as never)
    await expect(frames.snapshot()).rejects.toThrow('PREVIEW_FRAME_UNAVAILABLE')
    frames.dispose()
  })

  it('does not silently omit a frame whose document context is still loading', async () => {
    const dbg = new Debugger()
    const original = dbg.sendCommand.getMockImplementation()!
    dbg.sendCommand.mockImplementation(async (method, params, session) => {
      if (method === 'Runtime.enable' && session === 'child-session') {
        return {}
      }

      return original(method, params, session)
    })
    const frames = new PreviewFrames(dbg as never)
    await expect(frames.snapshot()).rejects.toThrow('PREVIEW_FRAME_UNAVAILABLE')
    frames.dispose()
  })

  it('never steals or detaches another debugger', async () => {
    const dbg = new Debugger()
    dbg.attached = true
    const frames = new PreviewFrames(dbg as never)
    await expect(frames.snapshot()).rejects.toThrow('PREVIEW_DEBUGGER_BUSY')
    expect(dbg.detach).not.toHaveBeenCalled()
    dbg.attached = false
    expect((await frames.snapshot()).length).toBe(2)
    frames.dispose()
    expect(dbg.detach).toHaveBeenCalledOnce()
  })
})

it('rejects a frame when the execution context was replaced under the same frame id', async () => {
  const dbg = new Debugger()
  const frames = new PreviewFrames(dbg as never)
  const child = (await frames.snapshot())[1]
  dbg.emit(
    'message',
    {},
    'Runtime.executionContextCreated',
    { context: { uniqueId: 'replacement', auxData: { frameId: 'child', isDefault: true } } },
    'child-session'
  )
  await expect(frames.evaluate(child, '1')).rejects.toThrow('STALE_FRAME')
  frames.dispose()
})
