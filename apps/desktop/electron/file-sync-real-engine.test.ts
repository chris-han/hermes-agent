import { execFile } from 'node:child_process'
import { createCipheriv, createDecipheriv, randomBytes, randomUUID } from 'node:crypto'
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

    let cookie = login.headers
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
        const response = await request(process.env.FILEX_DEVICE_SUPERVISOR === '1' ? '/api/file-sync/device-binding' : '/api/file-sync/binding', {
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
      },
      unseal: (value: string) => {
        const bytes = Buffer.from(value, 'base64')
        const decipher = createDecipheriv('aes-256-gcm', key, bytes.subarray(0, 12))
        decipher.setAuthTag(bytes.subarray(-16))
        return Buffer.concat([decipher.update(bytes.subarray(12, -16)), decipher.final()]).toString()
      },
      stopDevice: async (_origin: string, token: string): Promise<'STOPPED' | 'PENDING' | 'UNKNOWN'> => {
        const response = await fetch(origin + '/api/file-sync/device-stop', {
          method: 'POST', headers: { Authorization: 'Bearer ' + token }, credentials: 'omit'
        })
        if (!response.ok) {throw new Error('Device stop unavailable')}
        const result = await response.json()
        if (!['STOPPED', 'PENDING', 'UNKNOWN'].includes(result.state)) {throw new Error('Invalid device stop state')}
        return result.state
      }
    }

    let supervisor = new FileSyncSupervisor(options)
    const secondDevice = new FileSyncSupervisor({ ...options, stateDir: path.join(root, 'second-private') })

    try {
      const selected = await options.identity()
      const expectedScope = { user_id: selected.userId, workspace_id: selected.workspaceId,
        organization_id: selected.organizationId }
      const readWinman = async () => {
        const response = await request('/api/winman/capability')
        expect(response.status).toBe(200)
        return response.json()
      }
      const setWinman = async (enabled: boolean) => {
        const response = await request('/api/winman/capability', { method: 'PUT',
          headers: { Origin: origin, 'x-semantier-csrf': '1', 'content-type': 'application/json' },
          body: JSON.stringify({ enabled, expected_scope: expectedScope }) })
        expect(response.status).toBe(200)
        return response.json()
      }
      const target = await (await request('/__fixture/winman-target')).json()
      const admitWinman = async (requestId: string) => {
        const response = await request('/api/winman/messages/' + target.session_id, { method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ text: 'fixture', request_id: requestId, binding_id: target.binding_id }) })
        const receipt = response.status === 202 ? await response.json() : null
        const state = await (await request('/__fixture/winman-target')).json()
        return { status: receipt?.receipt?.status || (response.status === 403 ? 'denied' : response.status),
          admissions: state.admissions }
      }
      expect(await supervisor.owners()).toEqual([])
      expect(await readWinman()).toMatchObject({ desired: { enabled: false }, execution: 'unavailable' })
      expect(await admitWinman('files-off-winman-off')).toEqual({ status: 'denied', admissions: 0 })
      if (process.env.FILEX_DEVICE_SUPERVISOR === '1') {
        expect((await request('/__fixture/winman-execute', { method: 'POST' })).status).toBe(403)
      }
      await secondDevice.restore()
      expect(await secondDevice.owners()).toEqual([])
      expect((await secondDevice.statusFor(origin, await options.identity())).state).toBe('Paused')
      expect(await setWinman(true)).toMatchObject({ desired: { enabled: true }, execution: 'unavailable' })
      expect(await supervisor.owners()).toEqual([])
      expect(await admitWinman('files-off-winman-on')).toEqual({ status: 'queued', admissions: 1 })
      if (process.env.FILEX_DEVICE_SUPERVISOR === '1') {
        const execution = await request('/__fixture/winman-execute', { method: 'POST' })
        expect(execution.status).toBe(200)
        expect(await execution.json()).toEqual({ requests: ['files-off-winman-on'], completed: 1 })
      }
      const secondLogin = await fetch(origin + '/auth/password/login', {
        method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ login: f.login, password: f.password })
      })
      expect(secondLogin.status).toBe(200)
      const secondCookie = secondLogin.headers.getSetCookie().map(value => value.split(';')[0]).join('; ')
      const secondView = await fetch(origin + '/api/winman/capability', { headers: { cookie: secondCookie } })
      expect(await secondView.json()).toMatchObject({ desired: { enabled: true }, execution: 'unavailable' })
      await supervisor.enable(origin, folder)
      expect((await supervisor.owners())[0].enabled).toBe(true)
      if (process.env.FILEX_DEVICE_SUPERVISOR === '1') {
        expect((await supervisor.owners())[0].deviceGrant).toBe(true)
      }
      expect(await readWinman()).toMatchObject({ desired: { enabled: true }, execution: 'unavailable' })
      expect(await admitWinman('files-on-winman-on')).toEqual({ status: 'queued', admissions: 2 })
      if (process.env.FILEX_DEVICE_SUPERVISOR === '1') {
        const execution = await request('/__fixture/winman-execute', { method: 'POST' })
        expect(execution.status).toBe(200)
        expect(await execution.json()).toEqual({ requests: ['files-off-winman-on', 'files-on-winman-on'], completed: 2 })
      }
      await fs.writeFile(path.join(folder, name), 'Portable supervisor upload\n')
      await expect.poll(async () => {
        const bytes = await request('/api/file-sync/files?download=true&path=' + name)
        return bytes.status === 206 ? bytes.text() : ''
      }, { timeout: 20000 }).toBe('Portable supervisor upload\n')
      const listing = await request('/api/file-sync/files')
      expect(listing.status).toBe(200)
      expect((await listing.json()).files.map((file: { name: string }) => file.name)).not.toContain(secondOnly)
      expect(await fs.readFile(path.join(secondFolder, secondOnly), 'utf8')).toBe('Must stay on second device\n')
      const gate = async (action: string) => {
        const response = await request('/__fixture/filex-gate', { method: 'POST',
          headers: { 'content-type': 'application/json' }, body: JSON.stringify({ action }) })
        expect(response.status).toBe(200)
      }
      const afterOutage = 'offline-' + randomUUID() + '.md'
      await gate('offline')
      await fs.writeFile(path.join(folder, afterOutage), 'Explicit reconnect required\n')
      await expect.poll(async () => (await supervisor.owners())[0].enabled, { timeout: 30000 }).toBe(false)
      expect(supervisor.status().state).toBe('Error')
      await gate('online')
      await new Promise(resolve => setTimeout(resolve, 6500))
      const beforeResume = await request('/api/file-sync/files?download=true&path=' + afterOutage)
      expect(beforeResume.status).not.toBe(206)
      await supervisor.enable(origin, folder)
      await expect.poll(async () => {
        const response = await request('/api/file-sync/files?download=true&path=' + afterOutage)
        return response.status === 206 ? response.text() : ''
      }, { timeout: 20000 }).toBe('Explicit reconnect required\n')
      const afterPartialFailure = 'partial-' + randomUUID() + '.md'
      await gate('upload-offline')
      await fs.writeFile(path.join(folder, afterPartialFailure), 'Explicit partial recovery\n')
      await expect.poll(async () => (await supervisor.owners())[0].enabled, { timeout: 30000 }).toBe(false)
      expect(supervisor.status().state).toBe('Error')
      await gate('upload-online')
      await new Promise(resolve => setTimeout(resolve, 6500))
      expect((await request('/api/file-sync/files?download=true&path=' + afterPartialFailure)).status).not.toBe(206)
      await supervisor.enable(origin, folder)
      await expect.poll(async () => {
        const response = await request('/api/file-sync/files?download=true&path=' + afterPartialFailure)
        return response.status === 206 ? response.text() : ''
      }, { timeout: 20000 }).toBe('Explicit partial recovery\n')
      expect(await setWinman(false)).toMatchObject({ desired: { enabled: false }, execution: 'unavailable' })
      expect((await supervisor.owners())[0].enabled).toBe(true)
      expect(await admitWinman('files-on-winman-off')).toEqual({ status: 'denied', admissions: 2 })
      if (process.env.FILEX_DEVICE_SUPERVISOR === '1') {
        expect((await request('/__fixture/winman-execute', { method: 'POST' })).status).toBe(403)
      }
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
      // The disposable account token is deliberately revoked; reconnect must
      // preserve the exact local pair and baseline when the server reissues it.
      const revoked = await request('/api/file-sync/binding', { method: 'DELETE',
        headers: { Origin: origin, 'x-semantier-csrf': '1' } })
      expect(revoked.status).toBe(204)
      supervisor = new FileSyncSupervisor(options)
      await supervisor.enable(origin, folder)
      await expect.poll(() => supervisor.status().state, { timeout: 20000 }).toBe('Synced')
      const replacement = await options.binding()
      expect(replacement.token).not.toBe(credential.token)
      expect(replacement.remoteRoot).toBe(credential.remoteRoot)
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

      const secondChosen = path.join(root, 'second-chosen')
      await fs.mkdir(secondChosen)
      await secondDevice.enable(origin, secondChosen, await options.identity())
      await expect
        .poll(() => fs.readFile(path.join(secondChosen, name), 'utf8').catch(() => ''), { timeout: 20000 })
        .toBe('Web returned portable supervisor bytes\n')
      expect((await secondDevice.owners())[0].id).not.toBe((await supervisor.owners())[0].id)
      const secondUpload = 'second-device-upload-' + randomUUID() + '.md'
      await fs.writeFile(path.join(secondChosen, secondUpload), 'Second device bytes\n')
      await expect
        .poll(() => fs.readFile(path.join(folder, secondUpload), 'utf8').catch(() => ''), { timeout: 25000 })
        .toBe('Second device bytes\n')
      await secondDevice.pause()
      if (process.env.FILEX_DEVICE_SUPERVISOR === '1') {
        expect((await secondDevice.owners())[0]).toMatchObject({ deviceGrant: true, enabled: false, deviceStop: 'STOPPED' })
        expect((await supervisor.owners())[0]).toMatchObject({ deviceGrant: true, enabled: true })
      }
      const pausedOnly = 'paused-device-' + randomUUID() + '.md'
      await fs.writeFile(path.join(secondChosen, pausedOnly), 'Paused device bytes\n')
      const afterPause = await request('/api/file-sync/files')
      expect((await afterPause.json()).files.map((file: { name: string }) => file.name)).not.toContain(pausedOnly)
      expect(await fs.readFile(path.join(secondChosen, pausedOnly), 'utf8')).toBe('Paused device bytes\n')
      expect((await secondDevice.statusFor(origin, await options.identity())).state).toBe('Paused')

      if (f.other_login) {
        const gate = async (action?: string) => {
          const response = await request('/__fixture/filex-gate', action ? { method: 'POST',
            headers: { 'content-type': 'application/json' }, body: JSON.stringify({ action }) } : undefined)
          expect(response.status).toBe(200)
          return response.json()
        }
        await gate('arm')
        const inFlight = 'in-flight-' + randomUUID() + '.md'
        await fs.writeFile(path.join(folder, inFlight), 'Owner A upload already in flight\n')
        await expect.poll(async () => (await gate()).entered, { timeout: 20000 }).toBe(true)
        await supervisor.accountChanging(origin)
        const heldAfterSwitch = 'account-switch-paused-' + randomUUID() + '.md'
        await fs.writeFile(path.join(folder, heldAfterSwitch), 'Held until account A resumes\n')
        await expect(supervisor.resumeFor(origin, selected)).rejects.toThrow(/account change/i)
        const otherLogin = await fetch(origin + '/auth/password/login', {
          method: 'POST', headers: { 'content-type': 'application/json', Origin: origin },
          body: JSON.stringify({ login: f.other_login, password: f.password })
        })
        expect(otherLogin.status).toBe(200)
        cookie = otherLogin.headers.getSetCookie().map(value => value.split(';')[0]).join('; ')
        supervisor.accountChanged()
        await gate('release')
        await expect.poll(async () => (await gate()).finished, { timeout: 15000 }).toBe(true)
        const otherIdentity = await options.identity()
        expect(otherIdentity.userId).not.toBe(activeOwner.userId)
        expect(await readWinman()).toMatchObject({ desired: { enabled: false }, execution: 'unavailable' })
        const foreignToggle = await request('/api/winman/capability', { method: 'PUT',
          headers: { Origin: origin, 'x-semantier-csrf': '1', 'content-type': 'application/json' },
          body: JSON.stringify({ enabled: true, expected_scope: expectedScope }) })
        expect(foreignToggle.status).toBe(403)
        const otherScope = { user_id: otherIdentity.userId, workspace_id: otherIdentity.workspaceId,
          organization_id: otherIdentity.organizationId }
        const otherToggle = await request('/api/winman/capability', { method: 'PUT',
          headers: { Origin: origin, 'x-semantier-csrf': '1', 'content-type': 'application/json' },
          body: JSON.stringify({ enabled: true, expected_scope: otherScope }) })
        expect(otherToggle.status).toBe(200)
        expect(await otherToggle.json()).toMatchObject({ desired: { enabled: true }, execution: 'unavailable' })
        const foreignAdmission = await request('/api/winman/messages/' + target.session_id, { method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ text: 'fixture', request_id: 'foreign', binding_id: target.binding_id }) })
        expect(foreignAdmission.status).toBe(403)
        const ownerWinman = await fetch(origin + '/api/winman/capability', { headers: { cookie: secondCookie } })
        expect(await ownerWinman.json()).toMatchObject({ desired: { enabled: false }, execution: 'unavailable' })
        expect(await supervisor.statusFor(origin, otherIdentity)).toEqual({ state: 'Paused' })
        await expect(supervisor.enable(origin, folder, otherIdentity)).rejects.toThrow(/another account|folder binding/i)
        expect(await fs.readFile(path.join(folder, name), 'utf8')).toBe('Web returned portable supervisor bytes\n')
        const otherBinding = await fetch(origin + '/api/file-sync/binding', { method: 'POST', headers: { cookie,
          Origin: origin, 'x-semantier-csrf': '1', 'X-Semantier-User-Id': otherIdentity.userId,
          'X-Semantier-Workspace-Id': otherIdentity.workspaceId } })
        expect(otherBinding.status).toBe(200)
        const otherFiles = await fetch(origin + '/api/file-sync/files', { headers: { cookie,
          'X-Semantier-User-Id': otherIdentity.userId, 'X-Semantier-Workspace-Id': otherIdentity.workspaceId } })
        expect(otherFiles.status, await otherFiles.clone().text()).toBe(200)
        expect((await otherFiles.json()).files.map((file: { name: string }) => file.name)).not.toContain(inFlight)
        // Observe longer than the real engine's five-second watch interval.
        await new Promise(resolve => setTimeout(resolve, 6200))
        const ownerFiles = await fetch(origin + '/api/file-sync/files', { headers: { cookie: secondCookie,
          'X-Semantier-User-Id': selected.userId, 'X-Semantier-Workspace-Id': selected.workspaceId } })
        expect(ownerFiles.status).toBe(200)
        expect((await ownerFiles.json()).files.map((file: { name: string }) => file.name)).not.toContain(heldAfterSwitch)
        const gateOutcome = await gate()
        if (gateOutcome.kind === 'direct' && gateOutcome.status >= 200 && gateOutcome.status < 300) {
          const committed = await fetch(origin + '/api/file-sync/files?download=true&path=' + inFlight, { headers: {
            cookie: secondCookie, 'X-Semantier-User-Id': selected.userId, 'X-Semantier-Workspace-Id': selected.workspaceId } })
          expect(committed.status).toBe(206)
          expect(await committed.text()).toBe('Owner A upload already in flight\n')
        }
        await supervisor.accountChanging(origin)
        cookie = secondCookie
        supervisor.accountChanged()
        await supervisor.shutdown()
        supervisor = new FileSyncSupervisor(options)
        await supervisor.restore()
        expect((await supervisor.statusFor(origin, selected)).state).toBe('Paused')
        expect((await supervisor.owners())[0].enabled).toBe(false)
        await supervisor.resumeFor(origin, selected)
        await expect.poll(async () => {
          const response = await request('/api/file-sync/files?download=true&path=' + heldAfterSwitch)
          return response.status === 206 ? response.text() : ''
        }, { timeout: 20000 }).toBe('Held until account A resumes\n')
      }
    } finally {
      await supervisor.shutdown()
      await secondDevice.shutdown()
      await fs.rm(root, { recursive: true, force: true })
    }
  },
  90000
)
