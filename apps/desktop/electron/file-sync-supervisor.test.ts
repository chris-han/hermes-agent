import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, it } from 'vitest'

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
    run: async args => {
      if (args[1] === 'list') {return { code: 0, stdout: JSON.stringify(pairs) }}

      if (args[1] === 'add') {pairs = [{ local: args[2], remote: args[3], account: args[5] }]}

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
