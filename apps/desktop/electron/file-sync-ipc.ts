import path from 'node:path'

import { app, dialog, ipcMain, safeStorage } from 'electron'

import { FileSyncSupervisor } from './file-sync-supervisor'
import { workspace } from './semantier-workspace-ipc'

export function createFileSyncIpc(sourceForSender: (senderId: number) => string | null) {
  const assets = app.isPackaged
    ? path.join(process.resourcesPath, 'filex')
    : path.join(app.getAppPath(), 'resources', 'filex')

  async function identity(origin: string) {
    const value = await workspace(origin).client.identity()

    if (
      !value.authenticated ||
      !value.user?.user_id ||
      !value.workspace?.id ||
      value.user.workspace_id !== value.workspace.id
    )
      {throw new Error('Sign in to your personal Workspace first.')}

    return { userId: value.user.user_id, workspaceId: value.workspace.id }
  }

  const supervisor = new FileSyncSupervisor({
    stateDir: path.join(app.getPath('userData'), 'file-sync'),
    binary: path.join(assets, process.platform === 'win32' ? 'filex.exe' : 'filex'),
    helper: path.join(assets, 'filex-child.mjs'),
    identity,
    binding: origin => workspace(origin).client.fileSyncBinding(),
    seal: token => {
      if (
        !safeStorage.isEncryptionAvailable() ||
        (process.platform === 'linux' && safeStorage.getSelectedStorageBackend() === 'basic_text')
      )
        {throw new Error('OS credential protection is unavailable. Unlock your keychain and retry.')}

      return safeStorage.encryptString(token).toString('base64')
    },
    forbiddenRoots: [app.getPath('userData'), process.env.HERMES_HOME || path.join(app.getPath('home'), '.hermes')]
  })

  async function own(senderId: number) {
    const generation = supervisor.ownershipGeneration
    const origin = sourceForSender(senderId)

    if (!origin) {throw new Error('Connect this window to your Semantier Workspace first.')}

    const account = await identity(origin)

    if (generation !== supervisor.ownershipGeneration || sourceForSender(senderId) !== origin) {
      throw new Error('The account changed. Reconnect File Sync.')
    }

    return { origin, identity: account, generation }
  }

  async function unchanged(senderId: number, owner: Awaited<ReturnType<typeof own>>) {
    const current = await own(senderId)

    if (current.generation !== owner.generation || current.origin !== owner.origin || current.identity.userId !== owner.identity.userId || current.identity.workspaceId !== owner.identity.workspaceId) {
      throw new Error('The account changed. Reconnect File Sync.')
    }
  }

  ipcMain.handle('hermes:file-sync:status', async event => {
    const owner = await own(event.sender.id)

    const status = await supervisor.statusFor(owner.origin, owner.identity)
    await unchanged(event.sender.id, owner)

    return status
  })
  ipcMain.handle('hermes:file-sync:choose', async event => {
    const owner = await own(event.sender.id)

    const result = await dialog.showOpenDialog({
      title: 'File Sync — select one folder',
      properties: ['openDirectory']
    })

    if (result.canceled || result.filePaths.length !== 1) {return { state: 'Paused' }}
    const current = await own(event.sender.id)

    if (
      current.generation !== owner.generation ||
      current.origin !== owner.origin ||
      current.identity.userId !== owner.identity.userId ||
      current.identity.workspaceId !== owner.identity.workspaceId
    )
      {throw new Error('The account changed. Choose the folder again after reconnecting.')}

    return supervisor.enable(owner.origin, result.filePaths[0], owner.identity)
  })
  ipcMain.handle('hermes:file-sync:pause', async event => {
    const owner = await own(event.sender.id)
    const record = await supervisor.selectedBinding(owner.origin, owner.identity)
    await unchanged(event.sender.id, owner)

    if (!record || supervisor.status().folder !== record.canonicalLocalFolder) {throw new Error('No active folder for this account.')}
    await supervisor.pause()

    return supervisor.status()
  })
  ipcMain.handle('hermes:file-sync:resume', async event => {
    const owner = await own(event.sender.id)

    await unchanged(event.sender.id, owner)

    return supervisor.resumeFor(owner.origin, owner.identity)
  })

  return supervisor
}
