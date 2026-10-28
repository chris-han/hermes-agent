import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

import type { IpcMain, IpcMainInvokeEvent } from 'electron'
import { expect, test } from 'vitest'

import { DeliverySources } from './delivery-source'
import { registerDeliverySourceIpc } from './delivery-source-ipc'
import { UpdateOperation } from './operation'

test('real native source owner rejects guest/busy writes and the next check uses the saved origin', async () => {
  const root = mkdtempSync(path.join(tmpdir(), 'delivery-ipc-'))

  try {
    const sources = new DeliverySources(path.join(root, 'source.json'), path.join(root, 'config.yaml'), 'http://127.0.0.1:8926')
    const handlers = new Map<string, (event: IpcMainInvokeEvent, source?: unknown) => unknown>()
    const trusted = {} as IpcMainInvokeEvent
    let busy = false
    const operation = new UpdateOperation(async () => null)
    registerDeliverySourceIpc({ handle: (key, callback) => { handlers.set(key, callback) } } as Pick<IpcMain, 'handle'>, {
      sources, operation,
      assertViewer: event => { if (event !== trusted) {throw new Error('main frame')} },
      eligible: () => true,
      busy: () => busy,
      changed: () => {},
      check: async () => ({ supported: true, message: sources.base() })
    })
    const select = handlers.get('hermes:updates:delivery:set')!
    await expect(select({} as IpcMainInvokeEvent, 'channel2')).rejects.toThrow('main frame')
    busy = true
    await expect(select(trusted, 'channel2')).rejects.toThrow('progress')
    expect(sources.status().selected).toBe('channel3')
    busy = false
    expect(await select(trusted, 'channel2')).toMatchObject({ status: { message: 'https://s3.semantier.com' } })
    expect(new DeliverySources(path.join(root, 'source.json'), path.join(root, 'config.yaml'), '').status().selected).toBe('channel2')
  } finally { rmSync(root, { force: true, recursive: true }) }
})

test('a successful save and failed check retain the new source and return a distinct outcome', async () => {
  const root = mkdtempSync(path.join(tmpdir(), 'delivery-check-failure-'))

  try {
    const sources = new DeliverySources(path.join(root, 'source.json'), path.join(root, 'config.yaml'), 'http://127.0.0.1:8926')
    const handlers = new Map<string, (event: IpcMainInvokeEvent, source?: unknown) => unknown>()
    const changed: string[] = []
    registerDeliverySourceIpc({ handle: (key, callback) => { handlers.set(key, callback) } } as Pick<IpcMain, 'handle'>, {
      sources, operation: new UpdateOperation(async () => null), assertViewer: () => {},
      eligible: () => true, busy: () => false,
      changed: selection => { changed.push(selection.selected) },
      check: async () => { throw new Error('external endpoint unavailable') }
    })
    const result = await handlers.get('hermes:updates:delivery:set')!({} as IpcMainInvokeEvent, 'channel2')
    expect(result).toMatchObject({ saved: true, check_status: 'failed', selection: { selected: 'channel2' }, status: { error: 'check-failed' } })
    expect(sources.status().selected).toBe('channel2')
    expect(changed).toContain('channel2')
  } finally { rmSync(root, { force: true, recursive: true }) }
})


test('manual retry clears the saved-source failure through the native operation owner', async () => {
  const root = mkdtempSync(path.join(tmpdir(), 'delivery-retry-'))

  try {
    const sources = new DeliverySources(path.join(root, 'source.json'), path.join(root, 'config.yaml'), '')
    const operation = new UpdateOperation(async () => null)
    const handlers = new Map<string, (event: IpcMainInvokeEvent, source?: unknown) => unknown>()
    registerDeliverySourceIpc({ handle: (key, callback) => { handlers.set(key, callback) } } as Pick<IpcMain, 'handle'>, {
      sources, operation, assertViewer: () => {}, eligible: () => true, busy: () => false,
      changed: () => {}, check: async () => { throw new Error('offline') }
    })
    const event = {} as IpcMainInvokeEvent
    await handlers.get('hermes:updates:delivery:set')!(event, 'channel2')
    expect(handlers.get('hermes:updates:delivery:get')!(event)).toMatchObject({ selected: 'channel2', checkStatus: 'failed' })
    await operation.check(async () => ({ supported: true, updateAvailable: false }))
    expect(handlers.get('hermes:updates:delivery:get')!(event)).toMatchObject({ selected: 'channel2', checkStatus: 'completed' })
  } finally { rmSync(root, { force: true, recursive: true }) }
})
