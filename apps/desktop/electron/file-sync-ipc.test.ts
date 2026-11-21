import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

const mock = vi.hoisted(() => ({ handlers: new Map<string, (...args: any[]) => any>(), root: '', user: 'a' }))
vi.mock('electron', () => ({
  app: { isPackaged: false, getAppPath: () => mock.root, getPath: () => mock.root },
  ipcMain: { handle: (name: string, handler: any) => mock.handlers.set(name, handler) },
  dialog: { showOpenDialog: vi.fn() },
  safeStorage: { isEncryptionAvailable: () => true, getSelectedStorageBackend: () => 'secret_service', encryptString: () => Buffer.from('sealed') }
}))
vi.mock('./semantier-workspace-ipc', () => ({ workspace: () => ({ client: {
  identity: async () => ({ authenticated: true, user: { user_id: mock.user, workspace_id: 'w' + mock.user }, workspace: { id: 'w' + mock.user } }),
  fileSyncBinding: async () => ({ userId: mock.user, workspaceId: 'w' + mock.user, remoteRoot: 'private://', token: 'secret', enginePath: '/api/file-sync/engine', version: '0.61.1' })
} }) }))
import { createFileSyncIpc } from './file-sync-ipc'

const supervisors: ReturnType<typeof createFileSyncIpc>[] = []
afterEach(async () => {
  for (const s of supervisors.splice(0)) { await s.shutdown() }

  if (mock.root) { await fs.rm(mock.root, { recursive: true, force: true }) }
  mock.handlers.clear()
  mock.user = 'a'
})

async function setup() {
  mock.root = await fs.mkdtemp(path.join(os.tmpdir(), 'filex-ipc-'))
  const s = createFileSyncIpc(() => 'https://cloud.example')
  supervisors.push(s)
  s.options.forbiddenRoots = []

  s.options.verifyBinary = async () => {}
  const pairs = new Map<string, unknown[]>()

  let emitEvent: (event: any) => void = () => {}

  s.options.run = async (args, env, emit, signal) => {
    const key = env.FILEX_SYNC_DIR!

    if (args[1] === 'list') { return { code: 0, stdout: JSON.stringify(pairs.get(key) || []) } }

    if (args[1] === 'add') { pairs.set(key, [{ local: args[2], remote: args[3], account: args[5] }]);

 return { code: 0 } }

    emitEvent = emit
    emit({ event: 'pass', params: { failed: 0 } })
    await new Promise<void>(resolve => signal.addEventListener('abort', () => resolve(), { once: true }))

    return { code: 0 }
  }

  const old = path.join(mock.root, 'old'), current = path.join(mock.root, 'current')
  await fs.mkdir(old); await fs.mkdir(current)
  // The app state must not overlap either selected folder.
  s.options.stateDir = path.join(mock.root, 'file-sync')
  await s.enable('https://cloud.example', old)
  await fs.writeFile(path.join(old, 'retained.md'), 'old bytes')
  await s.pause()
  await s.enable('https://cloud.example', current)
  await fs.writeFile(path.join(current, 'retained.md'), 'current bytes')
  const invoke = (action: string) => mock.handlers.get('hermes:file-sync:' + action)!({ sender: { id: 1 } })

  return { s, old, current, invoke, emit: (event: any) => emitEvent(event) }
}

it('reports and resumes the running current folder with a paused historical binding retained', async () => {
  const { s, old, current, invoke } = await setup()
  const before = await s.owners()
  expect(await invoke('status')).toMatchObject({ state: 'Synced', folder: current })
  expect(await invoke('resume')).toMatchObject({ folder: current })
  expect(await s.owners()).toEqual(before)
  expect(await fs.readFile(path.join(old, 'retained.md'), 'utf8')).toBe('old bytes')
  expect(await fs.readFile(path.join(current, 'retained.md'), 'utf8')).toBe('current bytes')
})

it('restores the unique enabled folder and resumes the persisted selection after a paused restart', async () => {
  const { s, current, invoke } = await setup()
  await s.shutdown()
  const restarted = createFileSyncIpc(() => 'https://cloud.example')
  supervisors.push(restarted)
  Object.assign(restarted.options, s.options)
  await restarted.restore()
  expect(await invoke('status')).toMatchObject({ state: 'Synced', folder: current })
  await invoke('pause')
  await restarted.shutdown()
  const paused = createFileSyncIpc(() => 'https://cloud.example')
  supervisors.push(paused)
  Object.assign(paused.options, s.options)
  expect(await invoke('status')).toEqual({ state: 'Paused', folder: current })
  expect(await invoke('resume')).toMatchObject({ state: 'Synced', folder: current })
  expect(await paused.owners()).toHaveLength(2)
})

it('refuses restoration when the authenticated account no longer owns the retained binding', async () => {
  const { s } = await setup()
  await s.shutdown()
  const before = await s.owners()
  mock.user = 'b'
  const restarted = createFileSyncIpc(() => 'https://cloud.example')
  supervisors.push(restarted)
  Object.assign(restarted.options, s.options)
  await expect(restarted.restore()).rejects.toThrow('belongs to another account')
  expect(await restarted.owners()).toEqual(before)
})

it('isolates A to B to A, including resume and pause', async () => {
  const { s, current, invoke } = await setup()
  await s.accountChanging('https://cloud.example')
  mock.user = 'b'
  expect(await invoke('status')).toEqual({ state: 'Paused' })
  await expect(invoke('resume')).rejects.toThrow(/Choose a folder/)
  await expect(invoke('pause')).rejects.toThrow(/No active folder/)
  const b = path.join(mock.root, 'b')
  await fs.mkdir(b)
  await s.enable('https://cloud.example', b)
  expect(await invoke('status')).toMatchObject({ state: 'Synced', folder: b })
  await s.accountChanging('https://cloud.example')
  mock.user = 'a'
  expect(await invoke('status')).toEqual({ state: 'Paused', folder: current })
  expect(await invoke('resume')).toMatchObject({ state: 'Synced', folder: current })
  expect(await s.owners()).toHaveLength(3)
})

it.each(['ambiguous', 'multiple-enabled', 'wrong-selection', 'missing-selection', 'duplicate-id', 'duplicate-folder', 'invalid-enabled'])('refuses %s registry state without rewriting records', async kind => {
  const { s, invoke } = await setup()
  await s.shutdown()
  const registry = JSON.parse(await fs.readFile(s.registryPath, 'utf8'))

  if (kind === 'ambiguous') { registry.bindings.forEach((x: any) => { x.enabled = false }); delete registry.selectedBindingIds }

  if (kind === 'multiple-enabled') { registry.bindings.forEach((x: any) => { x.enabled = true }) }

  if (kind === 'wrong-selection') { registry.selectedBindingIds = [registry.bindings[0].id] }

  if (kind === 'missing-selection') { registry.selectedBindingIds = ['missing'] }

  if (kind === 'duplicate-folder') { registry.bindings[0].canonicalLocalFolder = registry.bindings[1].canonicalLocalFolder }

  if (kind === 'invalid-enabled') { registry.bindings[1].enabled = 'false' }

  if (kind === 'duplicate-id') { registry.bindings[0].id = registry.bindings[1].id }
  await fs.writeFile(s.registryPath, JSON.stringify(registry))
  const before = await fs.readFile(s.registryPath, 'utf8')
  const restarted = createFileSyncIpc(() => 'https://cloud.example')
  supervisors.push(restarted)
  Object.assign(restarted.options, s.options)
  await expect(invoke('status')).rejects.toThrow(/Ambiguous|Inconsistent|Invalid/)
  await expect(invoke('resume')).rejects.toThrow(/Ambiguous|Inconsistent|Invalid/)

  if (kind === 'multiple-enabled' || kind === 'wrong-selection') { await expect(restarted.restore()).rejects.toThrow(/Inconsistent/) }
  expect(await fs.readFile(s.registryPath, 'utf8')).toBe(before)
})

it('rejects a status read delayed across an account generation change', async () => {
  const { s, invoke } = await setup()
  const read = s.owners.bind(s)
  let release!: () => void
  let entered!: () => void
  const started = new Promise<void>(r => { entered = r })
  const wait = new Promise<void>(r => { release = r })
  vi.spyOn(s, 'owners').mockImplementationOnce(async () => { entered(); await wait;

 return read() })
  const status = invoke('status')
  const rejected = expect(status).rejects.toThrow(/account changed/)
  await started
  await s.accountChanging('https://cloud.example')
  mock.user = 'b'
  release()
  await rejected
  expect(await invoke('status')).toEqual({ state: 'Paused' })
})

it('rejects resume when authentication changes after selection but before enable', async () => {
  const { s, invoke } = await setup()
  const select = s.selectedBinding.bind(s)
  vi.spyOn(s, 'selectedBinding').mockImplementationOnce(async (...args) => {
    const record = await select(...args)
    mock.user = 'b'

    return record
  })
  const before = await fs.readFile(s.registryPath, 'utf8')
  await expect(invoke('resume')).rejects.toThrow(/account changed/)
  expect(await fs.readFile(s.registryPath, 'utf8')).toBe(before)
})

it('reports real Error, Conflict, Syncing and Paused states for the selected folder', async () => {
  const { current, invoke, emit } = await setup()

  for (const [event, state] of [
    [{ event: 'error' }, 'Error'],
    [{ event: 'pass', params: { conflicts: 1 } }, 'Conflict'],
    [{ event: 'pass', params: { failed: 1 } }, 'Error'],
    [{ event: 'progress' }, 'Syncing']
  ] as const) {
    emit(event)
    expect(await invoke('status')).toMatchObject({ state, folder: current })
  }

  await invoke('pause')
  expect(await invoke('status')).toEqual({ state: 'Paused', folder: current })
})

it('adopts a legacy unique enabled binding without using historical array order', async () => {
  const { s, current, invoke } = await setup()
  await s.shutdown()
  const registry = JSON.parse(await fs.readFile(s.registryPath, 'utf8'))
  delete registry.selectedBindingIds
  await fs.writeFile(s.registryPath, JSON.stringify(registry))
  const restarted = createFileSyncIpc(() => 'https://cloud.example')
  supervisors.push(restarted)
  Object.assign(restarted.options, s.options)
  await restarted.restore()
  expect(await invoke('status')).toEqual({ state: 'Synced', folder: current })
  expect(await invoke('resume')).toEqual({ state: 'Synced', folder: current })
})

it('refuses missing current ownership and does not expose a folder for another origin or workspace', async () => {
  const { s, invoke } = await setup()
  expect(await s.statusFor('https://other.example', { userId: 'a', workspaceId: 'wa' })).toEqual({ state: 'Paused' })
  expect(await s.statusFor('https://cloud.example', { userId: 'a', workspaceId: 'other' })).toEqual({ state: 'Paused' })
  await expect(s.resumeFor('https://other.example', { userId: 'a', workspaceId: 'wa' })).rejects.toThrow(/Choose a folder/)
  await fs.writeFile(s.registryPath, JSON.stringify({ version: 1, bindings: [] }))
  await expect(invoke('status')).rejects.toThrow(/Inconsistent/)
  await expect(invoke('resume')).rejects.toThrow(/Inconsistent/)
})

it('keeps an actual failed resume Error visible even when the persisted folder is paused', async () => {
  const { s, current, invoke } = await setup()
  await invoke('pause')

  s.options.verifyBinary = async () => { throw new Error('Pinned engine unavailable') }
  await expect(invoke('resume')).rejects.toThrow(/Pinned engine unavailable/)
  expect(await invoke('status')).toEqual({ state: 'Error', folder: current, message: 'Pinned engine unavailable' })
})
