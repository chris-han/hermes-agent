import path from 'node:path'

import { app, dialog, ipcMain, safeStorage } from 'electron'

import { FileSyncSupervisor } from './file-sync-supervisor'
import { assertWorkspaceScope, type WorkspaceScope } from './semantier-workspace-client'
import { workspace } from './semantier-workspace-ipc'

export type FileSyncAccount = { origin: string; scope: WorkspaceScope }

export function createFileSyncIpc(accountForDevice: () => FileSyncAccount | null) {
  const assets = app.isPackaged
    ? path.join(process.resourcesPath, 'filex')
    : path.join(app.getAppPath(), 'resources', 'filex')

  async function identity(origin: string) {
    const selected = accountForDevice()
    if (!selected || selected.origin !== origin) {throw new Error('Select your Semantier account first.')}
    const value = await workspace(origin).client.identity()
    assertWorkspaceScope(value, selected.scope)
    if (JSON.stringify(accountForDevice()) !== JSON.stringify(selected)) {throw new Error('The account changed. Reconnect File Sync.')}

    if (
      !value.authenticated ||
      !value.user?.user_id ||
      !value.workspace?.id ||
      value.user.workspace_id !== value.workspace.id
    )
      {throw new Error('Sign in to your personal Workspace first.')}

    return { userId: value.user.user_id, workspaceId: value.workspace.id, organizationId: selected.scope.organizationId }
  }

  const supervisor = new FileSyncSupervisor({
    stateDir: path.join(app.getPath('userData'), 'file-sync'),
    binary: path.join(assets, process.platform === 'win32' ? 'filex.exe' : 'filex'),
    helper: path.join(assets, 'filex-child.mjs'),
    identity,
    binding: origin => workspace(origin).client.fileSyncDeviceBinding(),
    stopDevice: (origin, token) => workspace(origin).client.fileSyncDeviceStop(token),
    seal: token => {
      if (
        !safeStorage.isEncryptionAvailable() ||
        (process.platform === 'linux' && safeStorage.getSelectedStorageBackend() === 'basic_text')
      )
        {throw new Error('OS credential protection is unavailable. Unlock your keychain and retry.')}

      return safeStorage.encryptString(token).toString('base64')
    },
    unseal: token => {
      if (
        !safeStorage.isEncryptionAvailable() ||
        (process.platform === 'linux' && safeStorage.getSelectedStorageBackend() === 'basic_text')
      )
        {throw new Error('OS credential protection is unavailable. Unlock your keychain and retry.')}

      return safeStorage.decryptString(Buffer.from(token, 'base64'))
    },
    forbiddenRoots: [app.getPath('userData'), process.env.HERMES_HOME || path.join(app.getPath('home'), '.hermes')]
  })

  async function own() {
    const generation = supervisor.ownershipGeneration
    const selected = accountForDevice()

    if (!selected) {throw new Error('Sign in to your Semantier account first.')}
    const { origin } = selected

    const account = await identity(origin)

    if (generation !== supervisor.ownershipGeneration || JSON.stringify(accountForDevice()) !== JSON.stringify(selected)) {
      throw new Error('The account changed. Reconnect File Sync.')
    }

    return { origin, identity: account, generation }
  }

  async function unchanged(owner: Awaited<ReturnType<typeof own>>) {
    const current = await own()

    if (current.generation !== owner.generation || current.origin !== owner.origin || current.identity.userId !== owner.identity.userId || current.identity.workspaceId !== owner.identity.workspaceId || current.identity.organizationId !== owner.identity.organizationId) {
      throw new Error('The account changed. Reconnect File Sync.')
    }
  }

  ipcMain.handle('hermes:file-sync:status', async () => {
    const owner = await own()

    const status = await supervisor.statusFor(owner.origin, owner.identity)
    await unchanged(owner)

    return status
  })
  ipcMain.handle('hermes:file-sync:choose', async () => {
    const owner = await own()

    const result = await dialog.showOpenDialog({
      title: 'File Sync — select one folder',
      properties: ['openDirectory']
    })

    if (result.canceled || result.filePaths.length !== 1) {
      await unchanged(owner)
      return supervisor.statusFor(owner.origin, owner.identity)
    }
    const zh = app.getLocale().toLowerCase().startsWith('zh')
    const confirmation = await dialog.showMessageBox({
      type: 'question',
      title: zh ? '启用文件同步' : 'Enable File Sync',
      message: zh ? '同步所选文件夹？' : 'Sync the selected folder?',
      detail: result.filePaths[0],
      buttons: zh ? ['取消', '启用同步'] : ['Cancel', 'Enable Sync'],
      defaultId: 0, cancelId: 0, noLink: true
    })
    if (confirmation.response !== 1) {
      await unchanged(owner)
      return supervisor.statusFor(owner.origin, owner.identity)
    }
    const current = await own()

    if (
      current.generation !== owner.generation ||
      current.origin !== owner.origin ||
      current.identity.userId !== owner.identity.userId ||
      current.identity.workspaceId !== owner.identity.workspaceId ||
      current.identity.organizationId !== owner.identity.organizationId
    )
      {throw new Error('The account changed. Choose the folder again after reconnecting.')}

    return supervisor.enable(owner.origin, result.filePaths[0], owner.identity, true)
  })
  ipcMain.handle('hermes:file-sync:pause', async () => {
    const owner = await own()
    const record = await supervisor.selectedBinding(owner.origin, owner.identity)
    await unchanged(owner)

    if (!record || supervisor.status().folder !== record.canonicalLocalFolder) {throw new Error('No active folder for this account.')}
    await supervisor.pause()

    return supervisor.status()
  })
  ipcMain.handle('hermes:file-sync:resume', async () => {
    const owner = await own()

    await unchanged(owner)

    return supervisor.resumeFor(owner.origin, owner.identity)
  })

  return supervisor
}
