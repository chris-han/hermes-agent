import { fork } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { fileURLToPath } from 'node:url'

import { expect, it, vi } from 'vitest'

import { superviseFilexChild } from './filex-child.mjs'

it('keeps account transition blocked when an OS tree-stop command fails, until child exit is observed', () => {
  const parent = Object.assign(new EventEmitter(), { env: {}, stdout: {} })

  const child = Object.assign(new EventEmitter(), {
    pid: 123,
    exitCode: null,
    stdout: { pipe: vi.fn() },
    stderr: { resume: vi.fn() }
  })

  const exit = vi.fn()
  const stop = vi.fn((_cmd, _args, _options, callback) => callback(new Error('OS denied termination')))
  superviseFilexChild(parent, { platform: 'win32', spawn: () => child, execFile: stop, killGroup: vi.fn(), exit })
  parent.emit('message', { binary: 'pinned-filex', args: [] })
  parent.emit('disconnect')
  expect(stop).toHaveBeenCalledTimes(1)
  expect(exit).not.toHaveBeenCalled()
  parent.emit('SIGTERM')
  expect(stop).toHaveBeenCalledTimes(2)
  expect(exit).not.toHaveBeenCalled()
  child.emit('exit', 1)
  expect(exit).toHaveBeenCalledWith(1)
})
it('stops its real portable child after parent IPC disconnects', async () => {
  const helper = fork(fileURLToPath(new URL('./filex-child.mjs', import.meta.url)), [], {
    stdio: ['ignore', 'pipe', 'ignore', 'ipc']
  })

  const pid = await new Promise<number>((resolve, reject) => {
    helper.once('error', reject)
    helper.stdout?.once('data', bytes => resolve(Number(String(bytes))))
    helper.send({
      binary: process.execPath,
      args: ['-e', 'process.stdout.write(String(process.pid));setInterval(()=>{},1000)']
    })
  })

  const stopped = new Promise(resolve => helper.once('exit', resolve))
  helper.disconnect()
  await stopped
  expect(() => process.kill(pid, 0)).toThrow()
})
