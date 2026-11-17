import { EventEmitter } from 'node:events'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { expect, it, vi } from 'vitest'
const mocks = vi.hoisted(() => ({ fork: vi.fn() }))
vi.mock('node:child_process', () => ({ fork: mocks.fork }))
import { FileSyncSupervisor } from './file-sync-supervisor'

it('does not treat a post-spawn IPC/kill error as proof that the engine stopped', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'filex-stop-proof-'))
  const folder = path.join(root, 'documents')
  await fs.mkdir(folder)
  const children: any[] = []
  mocks.fork.mockImplementation(() => {
    const child = Object.assign(new EventEmitter(), {
      pid: 100 + children.length,
      connected: true,
      stdout: new EventEmitter(),
      kill: vi.fn(),
      disconnect: vi.fn()
    }) as any

    child.disconnect.mockImplementation(() => {
      child.connected = false
      child.emit('error', Object.assign(new Error('OS refused stop'), { code: 'EPERM' }))
    })

    child.send = ({ args }: any, callback: () => void) => {
      callback()

      if (args[1] === 'list') {child.stdout.emit('data', '[]\n')}

      if (args[1] === 'run') {child.stdout.emit('data', '{"event":"pass","params":{"failed":0}}\n')}
      else {child.emit('exit', 0)}
    }

    children.push(child)

    return child
  })

  const supervisor = new FileSyncSupervisor({
    stateDir: path.join(root, 'private'),
    binary: 'pinned',
    helper: 'helper',
    identity: async () => ({ userId: 'a', workspaceId: 'wa' }),
    binding: async () => ({
      userId: 'a',
      workspaceId: 'wa',
      remoteRoot: 'private://',
      token: 'fixture-token',
      enginePath: '/api/file-sync/engine',
      version: '0.61.1'
    }),
    seal: () => 'encrypted',
    verifyBinary: async () => {}
  })

  try {
    await supervisor.enable('https://cloud.example', folder)
    expect(supervisor.status().state).toBe('Synced')
    vi.useFakeTimers()
    const stopping = expect(supervisor.shutdown()).rejects.toThrow('not stopped')
    await vi.advanceTimersByTimeAsync(8001)
    await stopping
    const child = children.at(-1)
    expect(child.disconnect).toHaveBeenCalled()
    expect(() => child.emit('error', new Error('second stop refused'))).not.toThrow()
    child.emit('exit', 0)
    await expect(supervisor.shutdown()).resolves.toBeUndefined()
  } finally {
    vi.useRealTimers()
    await fs.rm(root, { recursive: true, force: true })
  }
})
