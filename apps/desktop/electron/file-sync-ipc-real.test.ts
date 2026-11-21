import { createCipheriv, createDecipheriv, randomBytes, randomUUID } from 'node:crypto'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { expect, it, vi } from 'vitest'

const mock = vi.hoisted(() => ({
  handlers: new Map<string, (...args: any[]) => any>(), root: '', folder: '',
  client: null as any, seal: (_value: string) => Buffer.alloc(0), unseal: (_value: Buffer): string => '',
  picker: async () => ({ canceled: false, filePaths: [mock.folder] })
}))
vi.mock('electron', () => ({
  app: { isPackaged: false, getAppPath: () => mock.root, getPath: () => mock.root, getLocale: () => 'en' },
  ipcMain: { handle: (name: string, handler: any) => mock.handlers.set(name, handler) },
  dialog: { showOpenDialog: () => mock.picker(),
    showMessageBox: async () => ({ response: 1 }) },
  safeStorage: { isEncryptionAvailable: () => true, getSelectedStorageBackend: () => 'secret_service',
    encryptString: (value: string) => mock.seal(value), decryptString: (value: Buffer) => mock.unseal(value) }
}))
vi.mock('./semantier-workspace-ipc', () => ({ workspace: () => ({ client: mock.client }) }))

import { createFileSyncIpc } from './file-sync-ipc'
import { createSemantierWorkspaceClient } from './semantier-workspace-client'

const fixture = process.env.FILEX_PORTABLE_FIXTURE
it.skipIf(!fixture)('uses Desktop File Sync IPC with real account and FileX owners', async () => {
  const f = JSON.parse(await fs.readFile(fixture!, 'utf8'))
  const services = JSON.parse(await fs.readFile(path.resolve(path.dirname(fixture!), '..', 'services.json'), 'utf8'))
  const origin = services.containerWeb || services.web
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'semantier-filex-ipc-real-'))
  mock.root = path.join(root, 'app'); mock.folder = path.join(root, 'folder')
  await fs.mkdir(mock.root); await fs.mkdir(mock.folder)
  const key = randomBytes(32)
  mock.seal = value => {
    const iv = randomBytes(12), cipher = createCipheriv('aes-256-gcm', key, iv)
    return Buffer.concat([iv, cipher.update(value), cipher.final(), cipher.getAuthTag()])
  }
  mock.unseal = bytes => {
    const decipher = createDecipheriv('aes-256-gcm', key, bytes.subarray(0, 12))
    decipher.setAuthTag(bytes.subarray(-16))
    return Buffer.concat([decipher.update(bytes.subarray(12, -16)), decipher.final()]).toString()
  }
  let cookie = ''
  async function login(user: string) {
    const response = await fetch(origin + '/auth/password/login', { method: 'POST',
      headers: { 'content-type': 'application/json', Origin: origin },
      body: JSON.stringify({ login: user, password: f.password }) })
    expect(response.status).toBe(200)
    cookie = response.headers.getSetCookie().map(value => value.split(';')[0]).join('; ')
  }
  await login(f.login)
  mock.client = createSemantierWorkspaceClient(origin, (url, init) =>
    fetch(url, { ...init, headers: { ...init?.headers, ...(init?.credentials === 'omit' ? {} : { cookie }) } }))
  const identity = await mock.client.identity()
  const firstDevice = await mock.client.fileSyncDeviceBinding()
  const secondDevice = await mock.client.fileSyncDeviceBinding()
  expect(firstDevice.userId).toBe(identity.user.user_id)
  expect(firstDevice.token).not.toBe(secondDevice.token)
  expect(await mock.client.fileSyncDeviceStop(firstDevice.token)).toBe('STOPPED')
  const secondList = await fetch(origin + secondDevice.enginePath + '/api/files/manager?action=index&path=' +
    encodeURIComponent(secondDevice.remoteRoot), { headers: { Authorization: 'Bearer ' + secondDevice.token } })
  expect(secondList.status).toBe(200)
  expect(await mock.client.fileSyncDeviceStop(secondDevice.token)).toBe('STOPPED')
  let account = { origin, scope: { userId: identity.user.user_id, workspaceId: identity.workspace.id,
    organizationId: identity.organization_id } }
  const supervisor = createFileSyncIpc(() => account)
  supervisor.options.binary = process.env.FILEX_TEST_BINARY!
  supervisor.options.helper = fileURLToPath(new URL('./filex-child.mjs', import.meta.url))
  const invoke = (name: string) => mock.handlers.get('hermes:file-sync:' + name)!({ sender: { id: 1 } })
  try {
    expect(await invoke('status')).toEqual({ state: 'Paused' })
    expect(await supervisor.owners()).toEqual([])
    const ownerA = account
    let pickerOpened!: () => void
    let finishPicker!: (value: { canceled: boolean; filePaths: string[] }) => void
    const opened = new Promise<void>(resolve => {pickerOpened = resolve})
    mock.picker = () => new Promise(resolve => {finishPicker = resolve; pickerOpened()})
    const pendingChoice = invoke('choose')
    const rejectedChoice = expect(pendingChoice).rejects.toThrow(/account changed/i)
    await opened
    await supervisor.accountChanging(origin)
    await login(f.other_login)
    const switched = await mock.client.identity()
    account = { origin, scope: { userId: switched.user.user_id, workspaceId: switched.workspace.id,
      organizationId: switched.organization_id } }
    supervisor.accountChanged()
    finishPicker({ canceled: false, filePaths: [mock.folder] })
    await rejectedChoice
    expect(await supervisor.owners()).toEqual([])
    expect(await fs.readdir(mock.folder)).toEqual([])
    await supervisor.accountChanging(origin)
    await login(f.login)
    account = ownerA
    supervisor.accountChanged()
    mock.picker = async () => ({ canceled: false, filePaths: [mock.folder] })
    expect(await invoke('choose')).toMatchObject({ folder: await fs.realpath(mock.folder) })
    if (process.env.FILEX_DEVICE_SUPERVISOR === '1') {
      expect((await supervisor.owners())[0]).toMatchObject({ enabled: true, deviceGrant: true })
    }
    const filename = 'ipc-upload-' + randomUUID() + '.md'
    await fs.writeFile(path.join(mock.folder, filename), 'IPC fixture bytes\n')
    await expect.poll(async () => {
      const response = await fetch(origin + '/api/file-sync/files?download=true&path=' + filename, {
        headers: { cookie, 'X-Semantier-User-Id': account.scope.userId,
          'X-Semantier-Workspace-Id': account.scope.workspaceId } })
      return response.status === 206 ? response.text() : ''
    }, { timeout: 20000 }).toBe('IPC fixture bytes\n')
    await invoke('pause')
    expect((await invoke('status')).state).toBe('Paused')
    if (process.env.FILEX_DEVICE_SUPERVISOR === '1') {
      expect((await supervisor.owners())[0]).toMatchObject({ enabled: false, deviceStop: 'STOPPED' })
    }
    await invoke('resume')
    if (process.env.FILEX_DEVICE_SUPERVISOR === '1') {
      const gate = async (action?: string) => {
        const response = await fetch(origin + '/__fixture/filex-gate', action ? {
          method: 'POST', headers: { cookie, 'content-type': 'application/json' }, body: JSON.stringify({ action })
        } : { headers: { cookie } })
        expect(response.status).toBe(200)
        return response.json()
      }
      await gate('arm')
      await fs.writeFile(path.join(mock.folder, 'held-' + randomUUID() + '.md'), 'held device upload\n')
      await expect.poll(async () => (await gate()).entered, { timeout: 20000 }).toBe(true)
      expect((await invoke('pause')).state).toBe('Error')
      expect((await supervisor.owners())[0]).toMatchObject({ enabled: false, deviceStop: 'PENDING' })
      await gate('release')
      await expect.poll(async () => (await gate()).finished, { timeout: 15000 }).toBe(true)
    }
    await supervisor.accountChanging(origin)
    await login(f.other_login)
    const other = await mock.client.identity()
    account = { origin, scope: { userId: other.user.user_id, workspaceId: other.workspace.id,
      organizationId: other.organization_id } }
    supervisor.accountChanged()
    expect(await invoke('status')).toEqual({ state: 'Paused' })
    await expect(invoke('resume')).rejects.toThrow(/choose a folder/i)
    expect(await fs.readFile(path.join(mock.folder, filename), 'utf8')).toBe('IPC fixture bytes\n')
  } finally {
    await supervisor.shutdown()
    mock.handlers.clear()
    await fs.rm(root, { recursive: true, force: true })
  }
}, 60000)
