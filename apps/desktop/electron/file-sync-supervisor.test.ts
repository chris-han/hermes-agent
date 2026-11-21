import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

import { FileSyncSupervisor } from './file-sync-supervisor'

const roots: string[] = []
afterEach(async () => {
  for (const root of roots.splice(0)) {await fs.rm(root, { recursive: true, force: true })}
})

async function setup() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'semantier-pair-'))
  roots.push(root)
  const folder = path.join(root, 'documents')
  await fs.mkdir(folder)
  let identity = { userId: 'a', workspaceId: 'wa' }
  let pairs: unknown[] = []

  const supervisor = new FileSyncSupervisor({
    stateDir: path.join(root, 'private'),
    binary: '/missing',
    helper: '/missing',
    identity: async () => identity,
    binding: async () => ({
      ...identity,
      remoteRoot: 'private://',
      token: 'scoped-secret',
      enginePath: '/api/file-sync/engine',
      version: '0.61.1'
    }),
    seal: () => 'encrypted',
    verifyBinary: async () => {},
    run: async (args, _env, _event, signal) => {
      if (args[1] === 'list') {return { code: 0, stdout: JSON.stringify(pairs) }}

      if (args[1] === 'add') {pairs = [{ local: args[2], remote: args[3], account: args[5] }]}

      if (args[1] === 'run' && !signal.aborted) {
        await new Promise<void>(resolve => signal.addEventListener('abort', () => resolve(), { once: true }))
      }

      return { code: 0, events: [] }
    }
  })

  return {
    root,
    folder,
    supervisor,
    switchTo: (userId: string, workspaceId: string) => {
      identity = { userId, workspaceId }
    }
  }
}

it('keeps a durable A folder owner across pause, restart and fresh B pairing', async () => {
  const { folder, supervisor, switchTo } = await setup()
  await supervisor.enable('https://cloud.example', folder)
  await fs.writeFile(path.join(folder, 'notes.md'), 'A private notes')
  await supervisor.pause()
  switchTo('b', 'wb')
  await expect(new FileSyncSupervisor(supervisor.options).enable('https://cloud.example', folder)).rejects.toThrow(
    /another account/
  )
  const registry = JSON.parse(await fs.readFile(supervisor.registryPath, 'utf8'))
  expect(registry.bindings[0]).toMatchObject({
    userId: 'a',
    workspaceId: 'wa',
    canonicalLocalFolder: await fs.realpath(folder)
  })
  expect(JSON.stringify(registry)).not.toContain('scoped-secret')
  expect(await fs.readFile(path.join(folder, 'notes.md'), 'utf8')).toBe('A private notes')
  switchTo('a', 'wa')
  await supervisor.enable('https://cloud.example', folder)
  await supervisor.pause()
})
it('refuses overlapping and symlink aliases for a foreign account', async () => {
  const { root, folder, supervisor, switchTo } = await setup()
  await supervisor.enable('https://cloud.example', folder)
  await supervisor.pause()
  const nested = path.join(folder, 'nested')
  await fs.mkdir(nested)
  switchTo('b', 'wb')
  await expect(supervisor.enable('https://cloud.example', nested)).rejects.toThrow(/another account/)
  const alias = path.join(root, 'alias')
  await fs.symlink(folder, alias, 'dir')
  await expect(supervisor.enable('https://cloud.example', alias)).rejects.toThrow(/another account/)
})
it('fences an in-flight enable before it may launch after account change', async () => {
  const { folder, supervisor } = await setup()

  let finish: () => void = () => {}

  const wait = new Promise<void>(resolve => {
    finish = resolve
  })

  supervisor.options.binding = async () => {
    await wait

    return {
      userId: 'a',
      workspaceId: 'wa',
      remoteRoot: 'private://',
      token: 'secret',
      enginePath: '/api/file-sync/engine',
      version: '0.61.1'
    }
  }

  const enabling = supervisor.enable('https://cloud.example', folder)
  await new Promise(resolve => setTimeout(resolve, 20))
  await supervisor.accountChanging('https://cloud.example')
  supervisor.accountChanged()
  finish()
  await expect(enabling).rejects.toThrow(/account changed/)
  expect(supervisor.status().state).not.toBe('Synced')
})

it('adopts an interrupted exact engine pair without re-adding or resetting its baseline', async () => {
  const { folder, supervisor } = await setup()
  await supervisor.enable('https://cloud.example', folder)
  await supervisor.pause()
  const registry = JSON.parse(await fs.readFile(supervisor.registryPath, 'utf8'))
  registry.bindings[0].paired = false
  await fs.writeFile(supervisor.registryPath, JSON.stringify(registry))
  const original = supervisor.options.run!
  let added = 0

  supervisor.options.run = async (...args) => {
    if (args[0][1] === 'add') {added++}

    return original(...args)
  }

  const restarted = new FileSyncSupervisor(supervisor.options)
  await restarted.enable('https://cloud.example', folder)
  await restarted.pause()
  expect(added).toBe(0)
})
it('fences a different origin even when another origin was previously current', async () => {
  const { folder, supervisor } = await setup()
  await supervisor.enable('https://old.example', folder)
  await supervisor.pause()

  let finish: () => void = () => {}

  const wait = new Promise<void>(r => {
    finish = r
  })

  const previous = supervisor.options.identity

  supervisor.options.identity = async origin => {
    await wait

    return previous(origin)
  }

  const operation = supervisor.enable('https://new.example', folder)
  await new Promise(r => setTimeout(r, 20))
  await supervisor.accountChanging('https://new.example')
  supervisor.accountChanged()
  finish()
  await expect(operation).rejects.toThrow(/account changed/)
})
it('does not recreate a missing pair or launch when the selected drive is absent', async () => {
  const { root, folder, supervisor } = await setup()
  await supervisor.enable('https://cloud.example', folder)
  await supervisor.pause()
  const original = supervisor.options.run!
  let adds = 0

  supervisor.options.run = async (...args) => {
    if (args[0][1] === 'list') {return { code: 0, stdout: '[]' }}

    if (args[0][1] === 'add') {adds++}

    return original(...args)
  }

  await expect(supervisor.enable('https://cloud.example', folder)).rejects.toThrow(/missing/)
  expect(adds).toBe(0)
  await fs.rename(folder, path.join(root, 'unmounted'))
  await expect(supervisor.enable('https://cloud.example', folder)).rejects.toThrow()
  expect(adds).toBe(0)
  expect(supervisor.status()).toMatchObject({ state: 'Error', folder })
})
it('can pause and retry after an unavailable helper without poisoning the lifecycle', async () => {
  const { folder, supervisor } = await setup()
  supervisor.options.run = undefined
  await expect(supervisor.enable('https://cloud.example', folder)).rejects.toThrow(/pairing state/)
  await expect(supervisor.pause()).resolves.toBeUndefined()
  await expect(supervisor.enable('https://cloud.example', folder)).rejects.toThrow(/pairing state/)
})

it('does not silently resume an enabled folder after offline restoration fails', async () => {
  const { folder, supervisor } = await setup()
  await supervisor.enable('https://cloud.example', folder)
  await supervisor.shutdown()
  let identityCalls = 0
  const options = { ...supervisor.options, identity: async () => {
    identityCalls++
    throw new Error('Account unavailable offline')
  } }
  const offline = new FileSyncSupervisor(options)
  await expect(offline.restore()).rejects.toThrow(/offline/)
  expect((await offline.owners())[0].enabled).toBe(false)
  const recovered = new FileSyncSupervisor(options)
  await expect(recovered.restore()).resolves.toBeUndefined()
  expect(identityCalls).toBe(1)
  expect(recovered.status().state).toBe('Paused')
})


it('rejects residual files with no local owner before obtaining credentials or running the engine', async () => {
  const { folder, supervisor, switchTo } = await setup()
  await fs.writeFile(path.join(folder, 'notes.md'), 'Account A residual bytes')
  switchTo('b', 'wb')
  let bindingCalls = 0
  let engineCalls = 0
  const binding = supervisor.options.binding

  supervisor.options.binding = async origin => {
    bindingCalls++

    return binding(origin)
  }

  supervisor.options.run = async () => {
    engineCalls++

    return { code: 0, stdout: '[]' }
  }

  await expect(supervisor.enable('https://cloud.example', folder)).rejects.toThrow(/empty folder/)
  expect(bindingCalls).toBe(0)
  expect(engineCalls).toBe(0)
  expect(await supervisor.owners()).toEqual([])
  expect(await fs.readFile(path.join(folder, 'notes.md'), 'utf8')).toBe('Account A residual bytes')
})

it('persists an unresolved device stop and refuses reconnect after restart', async () => {
  const { folder, supervisor } = await setup()
  const token = 'fxd_' + 'a'.repeat(43)
  supervisor.options.binding = async () => ({ userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' })
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => 'PENDING'
  await supervisor.enable('https://cloud.example', folder)
  await supervisor.pause()
  const registry = JSON.parse(await fs.readFile(supervisor.registryPath, 'utf8'))
  expect(registry.bindings[0]).toMatchObject({ enabled: false, deviceStop: 'PENDING' })
  const restarted = new FileSyncSupervisor(supervisor.options)
  await restarted.restore()
  expect((await restarted.statusFor('https://cloud.example', { userId: 'a', workspaceId: 'wa' })).state).toBe('Error')
  await expect(restarted.enable('https://cloud.example', folder)).rejects.toThrow(/stop.*pending/i)
  expect((await restarted.owners())[0].sealedToken).toBe('sealed:' + token)
})

it('records an unknown device stop before contacting the server', async () => {
  const { folder, supervisor } = await setup()
  const token = 'fxd_' + 'b'.repeat(43)
  supervisor.options.binding = async () => ({ userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' })
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => {
    const registry = JSON.parse(await fs.readFile(supervisor.registryPath, 'utf8'))
    expect(registry.bindings[0]).toMatchObject({ enabled: false, deviceStop: 'UNKNOWN' })
    throw new Error('network lost')
  }
  await supervisor.enable('https://cloud.example', folder)
  await supervisor.pause()
  expect((await supervisor.owners())[0]).toMatchObject({ enabled: false, deviceStop: 'UNKNOWN' })
  expect(supervisor.status().state).toBe('Error')
})

it('fences a previously enabled device grant before restoring after a crash', async () => {
  const { folder, supervisor } = await setup()
  const token = 'fxd_' + 'c'.repeat(43)
  let issued = 0
  let stopped = 0
  supervisor.options.binding = async () => {
    issued++
    return { userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' }
  }
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => { stopped++; return 'PENDING' }
  await supervisor.enable('https://cloud.example', folder)
  const restarted = new FileSyncSupervisor(supervisor.options)
  await restarted.restore()
  expect(stopped).toBe(1)
  expect(issued).toBe(1)
  expect((await restarted.owners())[0]).toMatchObject({ enabled: false, deviceStop: 'PENDING' })
  await supervisor.pause()
})

it('fences a device grant on shutdown while preserving enabled intent when confirmed stopped', async () => {
  const { folder, supervisor } = await setup()
  const token = 'fxd_' + 'd'.repeat(43)
  let stopped = 0
  supervisor.options.binding = async () => ({ userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' })
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => { stopped++; return 'STOPPED' }
  await supervisor.enable('https://cloud.example', folder)
  await supervisor.shutdown()
  expect(stopped).toBe(1)
  expect((await supervisor.owners())[0]).toMatchObject({ enabled: true, deviceStop: 'STOPPED' })
  const restarted = new FileSyncSupervisor(supervisor.options)
  await restarted.restore()
  expect(restarted.status().state).toBe('Syncing')
  await restarted.pause()
})

it('retires the active device grant before replacing a running binding', async () => {
  const { folder, supervisor } = await setup()
  const token = 'fxd_' + 'e'.repeat(43)
  let issued = 0
  supervisor.options.binding = async () => { issued++; return { userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' } }
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => 'PENDING'
  await supervisor.enable('https://cloud.example', folder)
  await expect(supervisor.enable('https://cloud.example', folder)).rejects.toThrow(/stop.*pending/i)
  expect(issued).toBe(1)
  expect((await supervisor.owners())[0]).toMatchObject({ enabled: false, deviceStop: 'PENDING' })
})

it('retires the device grant when the watcher exits unexpectedly', async () => {
  const { folder, supervisor } = await setup()
  const token = 'fxd_' + 'f'.repeat(43)
  const original = supervisor.options.run!
  let exit: () => void = () => {}
  supervisor.options.binding = async () => ({ userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' })
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => 'PENDING'
  supervisor.options.run = async (...args) => {
    if (args[0][1] !== 'run') {return original(...args)}
    await new Promise<void>(resolve => { exit = resolve })
    return { code: 1 }
  }
  await supervisor.enable('https://cloud.example', folder)
  exit()
  await vi.waitFor(async () => expect((await supervisor.owners())[0]).toMatchObject({ enabled: false, deviceStop: 'PENDING' }))
  expect(supervisor.status().state).toBe('Error')
})

it('shares an in-flight device stop across pause and account change', async () => {
  const { folder, supervisor } = await setup()
  const token = 'fxd_' + 'g'.repeat(43)
  let calls = 0
  let release: () => void = () => {}
  let started: () => void = () => {}
  const waiting = new Promise<void>(resolve => { started = resolve })
  const hold = new Promise<void>(resolve => { release = resolve })
  supervisor.options.binding = async () => ({ userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' })
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => { calls++; started(); await hold; return 'PENDING' }
  await supervisor.enable('https://cloud.example', folder)
  const pause = supervisor.pause()
  await waiting
  const changing = supervisor.accountChanging('https://cloud.example')
  release()
  await Promise.all([pause, changing])
  supervisor.accountChanged()
  expect(calls).toBe(1)
  expect((await supervisor.owners())[0]).toMatchObject({ enabled: false, deviceStop: 'PENDING' })
})

it('retries a pending stop before an explicit resume and issues only after confirmation', async () => {
  const { folder, supervisor } = await setup()
  const token = 'fxd_' + 'h'.repeat(43)
  let issued = 0
  let calls = 0
  supervisor.options.binding = async () => { issued++; return { userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' } }
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => ++calls === 1 ? 'PENDING' : 'STOPPED'
  await supervisor.enable('https://cloud.example', folder)
  await supervisor.pause()
  const restarted = new FileSyncSupervisor(supervisor.options)
  await restarted.enable('https://cloud.example', folder)
  expect(calls).toBe(2)
  expect(issued).toBe(2)
  await restarted.pause()
})

it('does not restore enabled intent on shutdown after an explicit pause', async () => {
  const { folder, supervisor } = await setup()
  const token = 'fxd_' + 'i'.repeat(43)
  supervisor.options.binding = async () => ({ userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' })
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => 'STOPPED'
  await supervisor.enable('https://cloud.example', folder)
  await supervisor.pause()
  await supervisor.shutdown()
  expect((await supervisor.owners())[0]).toMatchObject({ enabled: false, deviceStop: 'STOPPED' })
})

it('waits for an ownership commit and fences its grant during account change', async () => {
  const { folder, supervisor } = await setup()
  const token = 'fxd_' + 'j'.repeat(43)
  let release: () => void = () => {}
  let started: () => void = () => {}
  const waiting = new Promise<void>(resolve => { started = resolve })
  const hold = new Promise<void>(resolve => { release = resolve })
  let stops = 0
  supervisor.options.binding = async () => ({ userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' })
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => { stops++; return 'STOPPED' }
  const internals = supervisor as unknown as { write: (...args: any[]) => Promise<void> }
  const write = internals.write.bind(supervisor)
  let first = true
  internals.write = async (...args) => {
    if (first) { first = false; started(); await hold }
    await write(...args)
  }
  const enabling = supervisor.enable('https://cloud.example', folder)
  const rejected = expect(enabling).rejects.toThrow(/account changed/i)
  await waiting
  const changing = supervisor.accountChanging('https://cloud.example')
  release()
  await Promise.all([rejected, changing])
  supervisor.accountChanged()
  expect(stops).toBe(1)
  expect((await supervisor.owners())[0]).toMatchObject({ enabled: false, deviceStop: 'STOPPED' })
})

it('fences the device grant when pairing fails after ownership is committed', async () => {
  const { folder, supervisor } = await setup()
  const token = 'fxd_' + 'k'.repeat(43)
  let stops = 0
  supervisor.options.binding = async () => ({ userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' })
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => { stops++; return 'PENDING' }
  supervisor.options.run = async () => ({ code: 1 })
  await expect(supervisor.enable('https://cloud.example', folder)).rejects.toThrow(/pairing state/i)
  expect(stops).toBe(1)
  expect((await supervisor.owners())[0]).toMatchObject({ enabled: false, deviceStop: 'PENDING' })
})

it('does not re-enable a stopped grant when shutdown races account change', async () => {
  const { folder, supervisor } = await setup()
  const token = 'fxd_' + 'l'.repeat(43)
  let release: () => void = () => {}
  let started: () => void = () => {}
  const waiting = new Promise<void>(resolve => { started = resolve })
  const hold = new Promise<void>(resolve => { release = resolve })
  supervisor.options.binding = async () => ({ userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' })
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => 'STOPPED'
  await supervisor.enable('https://cloud.example', folder)
  const internals = supervisor as unknown as { write: (...args: any[]) => Promise<void> }
  const write = internals.write.bind(supervisor)
  internals.write = async (...args) => {
    if (args[0].some((record: any) => record.enabled && record.deviceStop === 'STOPPED')) { started(); await hold }
    await write(...args)
  }
  const shutdown = supervisor.shutdown()
  await waiting
  let changed = false
  const changing = supervisor.accountChanging('https://cloud.example').then(() => { changed = true })
  await new Promise(resolve => setTimeout(resolve, 20))
  expect(changed).toBe(false)
  release()
  await Promise.all([shutdown, changing])
  supervisor.accountChanged()
  expect((await supervisor.owners())[0]).toMatchObject({ enabled: false, deviceStop: 'STOPPED' })
})

it('does not bypass an unresolved stop by selecting another folder for the same account', async () => {
  const { root, folder, supervisor } = await setup()
  const other = path.join(root, 'other')
  await fs.mkdir(other)
  const token = 'fxd_' + 'm'.repeat(43)
  let issued = 0
  supervisor.options.binding = async () => { issued++; return { userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' } }
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => 'UNKNOWN'
  await supervisor.enable('https://cloud.example', folder)
  await supervisor.pause()
  await expect(supervisor.enable('https://cloud.example', other)).rejects.toThrow(/stop.*unknown/i)
  expect(issued).toBe(1)
})

it('keeps an unresolved stop scoped to its account when another account selects a separate folder', async () => {
  const { root, folder, supervisor, switchTo } = await setup()
  const other = path.join(root, 'other')
  await fs.mkdir(other)
  const token = 'fxd_' + 'n'.repeat(43)
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => 'UNKNOWN'
  supervisor.options.binding = async () => ({ userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' })
  await supervisor.enable('https://cloud.example', folder)
  await supervisor.accountChanging('https://cloud.example')
  supervisor.accountChanged()
  switchTo('b', 'wb')
  supervisor.options.binding = async () => ({ userId: 'b', workspaceId: 'wb', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' })
  // Each account has separate engine pairing state in the real client.
  supervisor.options.run = async (args, _env, _event, signal) => {
    if (args[1] === 'list') {return { code: 0, stdout: '[]' }}
    if (args[1] === 'run' && !signal.aborted) {await new Promise<void>(resolve => signal.addEventListener('abort', () => resolve(), { once: true }))}
    return { code: 0 }
  }
  await supervisor.enable('https://cloud.example', other)
  expect((await supervisor.owners()).find(owner => owner.userId === 'a')).toMatchObject({ enabled: false, deviceStop: 'UNKNOWN' })
  await supervisor.pause()
})

it('stops the local watcher even if the device stop fence cannot be written', async () => {
  const { folder, supervisor } = await setup()
  const token = 'fxd_' + 'o'.repeat(43)
  let watchSignal: AbortSignal | undefined
  const run = supervisor.options.run!
  supervisor.options.run = (...args) => {
    if (args[0][1] === 'run') {watchSignal = args[3]}
    return run(...args)
  }
  supervisor.options.binding = async () => ({ userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' })
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => 'STOPPED'
  await supervisor.enable('https://cloud.example', folder)
  const internals = supervisor as unknown as { write: (...args: any[]) => Promise<void> }
  const write = internals.write.bind(supervisor)
  internals.write = async () => { throw new Error('fixture disk failure') }
  try {
    await expect(supervisor.pause()).rejects.toThrow('fixture disk failure')
    expect(watchSignal?.aborted).toBe(true)
    expect(supervisor.status().state).toBe('Error')
  } finally {
    internals.write = write
    await supervisor.pause()
  }
})

it('retires an enabled device grant if account change precedes restoration', async () => {
  const { folder, supervisor } = await setup()
  const token = 'fxd_' + 'q'.repeat(43)
  let stops = 0
  supervisor.options.binding = async () => ({ userId: 'a', workspaceId: 'wa', remoteRoot: 'private://', token, enginePath: '/api/file-sync/engine', version: '0.61.1' })
  supervisor.options.seal = value => 'sealed:' + value
  supervisor.options.unseal = value => value.slice(7)
  supervisor.options.stopDevice = async () => { stops++; return 'PENDING' }
  await supervisor.enable('https://cloud.example', folder)
  const restarted = new FileSyncSupervisor(supervisor.options)
  await restarted.accountChanging('https://cloud.example')
  restarted.accountChanged()
  expect(stops).toBe(1)
  expect((await restarted.owners())[0]).toMatchObject({ enabled: false, deviceStop: 'PENDING' })
  await supervisor.pause()
})
