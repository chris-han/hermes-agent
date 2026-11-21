import { execFile } from 'node:child_process'
import { createCipheriv, randomBytes, randomUUID } from 'node:crypto'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

import { expect, it } from 'vitest'

import { FileSyncSupervisor } from './file-sync-supervisor'

const fixture = process.env.FILEX_PORTABLE_FIXTURE
it.skipIf(!fixture)(
  'runs the portable production supervisor through real login, scoped Web transport and durable CLI state',
  async () => {
    const f = JSON.parse(await fs.readFile(fixture!, 'utf8'))
    const services = JSON.parse(await fs.readFile(path.resolve(path.dirname(fixture!), '..', 'services.json'), 'utf8'))
    const origin = services.containerWeb || services.web

    const login = await fetch(origin + '/auth/password/login', {
      method: 'POST',
      headers: { 'content-type': 'application/json', Origin: origin },
      body: JSON.stringify({ login: f.login, password: f.password })
    })

    expect(login.status).toBe(200)

    const cookie = login.headers
      .getSetCookie()
      .map(value => value.split(';')[0])
      .join('; ')

    async function request(url: string, init?: RequestInit) {
      return fetch(origin + url, { ...init, headers: { cookie, 'X-Semantier-User-Id':f.user_id,'X-Semantier-Workspace-Id':f.workspace_id, ...init?.headers } })
    }

    const root = await fs.mkdtemp(path.join(os.tmpdir(), 'semantier-filex-real-'))
    const folder = path.join(root, 'documents')
    await fs.mkdir(folder)
    const secondFolder = path.join(root, 'second-device')
    await fs.mkdir(secondFolder)
    const secondOnly = 'second-device-off-' + randomUUID() + '.md'
    await fs.writeFile(path.join(secondFolder, secondOnly), 'Must stay on second device\n')
    const key = randomBytes(32)
    const name = 'portable-supervisor-' + randomUUID() + '.md'

    const options = {
      stateDir: path.join(root, 'private'),
      binary: process.env.FILEX_TEST_BINARY!,
      helper: fileURLToPath(new URL('./filex-child.mjs', import.meta.url)),
      identity: async () => {
        const response = await request('/auth/context')
        expect(response.status).toBe(200)
        const value = await response.json()

        return { userId: value.user.user_id, workspaceId: value.workspace.id, organizationId: value.organization_id }
      },
      binding: async () => {
        const response = await request('/api/file-sync/binding', {
          method: 'POST',
          headers: { Origin: origin, 'x-semantier-csrf': '1' }
        })

        expect(response.status).toBe(200)

        return response.json()
      },
      // Test boundary only: production uses Electron safeStorage, qualified on Windows.
      seal: (value: string) => {
        const iv = randomBytes(12)
        const cipher = createCipheriv('aes-256-gcm', key, iv)

        return Buffer.concat([iv, cipher.update(value), cipher.final(), cipher.getAuthTag()]).toString('base64')
      }
    }

    let supervisor = new FileSyncSupervisor(options)
    const secondDevice = new FileSyncSupervisor({ ...options, stateDir: path.join(root, 'second-private') })

    try {
      await secondDevice.restore()
      expect(await secondDevice.owners()).toEqual([])
      expect((await secondDevice.statusFor(origin, await options.identity())).state).toBe('Paused')
      await supervisor.enable(origin, folder)
      await fs.writeFile(path.join(folder, name), 'Portable supervisor upload\n')
      await expect.poll(() => supervisor.status().state, { timeout: 20000 }).toBe('Synced')
      const bytes = await request('/api/file-sync/files?download=true&path=' + name)
      expect(bytes.status).toBe(206)
      expect(await bytes.text()).toBe('Portable supervisor upload\n')
      const listing = await request('/api/file-sync/files')
      expect(listing.status).toBe(200)
      expect((await listing.json()).files.map((file: { name: string }) => file.name)).not.toContain(secondOnly)
      expect(await fs.readFile(path.join(secondFolder, secondOnly), 'utf8')).toBe('Must stay on second device\n')
      const activeOwner = (await supervisor.owners())[0]
      const credential = await options.binding()
      let lockCode = 0

      try {
        await promisify(execFile)(options.binary, ['sync', 'run', '--json', '--account', activeOwner.id], {
          env: {
            ...process.env,
            FILEX_URL: origin + credential.enginePath,
            FILEX_TOKEN: credential.token,
            FILEX_SYNC_DIR: path.join(options.stateDir, activeOwner.id),
            FILEX_CLI_CONFIG: path.join(options.stateDir, activeOwner.id, 'cli.yaml'),
            FILEX_LANG: 'en'
          },
          timeout: 15000
        })
      } catch (error) {
        lockCode = Number((error as { code?: number }).code)
      }

      expect(lockCode).toBe(4)
      await supervisor.pause()
      const oldOwners = await supervisor.owners()
      const baseline = await fs.readFile(path.join(options.stateDir, oldOwners[0].id, 'baseline', 'pair-1.json'))
      supervisor = new FileSyncSupervisor(options)
      await supervisor.enable(origin, folder)
      await expect.poll(() => supervisor.status().state, { timeout: 20000 }).toBe('Synced')
      expect((await supervisor.owners())[0].id).toBe(oldOwners[0].id)
      expect(await fs.readFile(path.join(options.stateDir, oldOwners[0].id, 'baseline', 'pair-1.json'))).toEqual(
        baseline
      )

      const saved = await request('/api/file-sync/files', {
        method: 'POST',
        headers: { Origin: origin, 'x-semantier-csrf': '1', 'content-type': 'application/json' },
        body: JSON.stringify({ path: name, content: 'Web returned portable supervisor bytes\n' })
      })

      expect(saved.status).toBe(200)
      await expect
        .poll(() => fs.readFile(path.join(folder, name), 'utf8'), { timeout: 15000 })
        .toBe('Web returned portable supervisor bytes\n')
    } finally {
      await supervisor.shutdown()
      await secondDevice.shutdown()
      await fs.rm(root, { recursive: true, force: true })
    }
  },
  60000
)
