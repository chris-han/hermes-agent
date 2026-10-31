import { mkdtempSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import type { IpcMain, IpcMainInvokeEvent } from 'electron'
import { afterEach, expect, it } from 'vitest'

import { registerSemantierFeedbackIpc } from './semantier-feedback-ipc'
import { SemantierObservations } from './semantier-observations'
const roots: string[] = []
afterEach(() => {
  for (const r of roots.splice(0)) {
    rmSync(r, { recursive: true, force: true })
  }
})

it('feedback IPC requires an application viewer before any journal operation', () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'semantier-feedback-ipc-'))
  roots.push(dir)
  const owner = new SemantierObservations(dir)
  const trusted = {} as IpcMainInvokeEvent
  let handler: (event: IpcMainInvokeEvent, command: string, value?: unknown) => unknown = () => null
  registerSemantierFeedbackIpc(
    {
      handle: (_name, callback) => {
        handler = callback
      }
    } as Pick<IpcMain, 'handle'>,
    owner,
    event => {
      if (event !== trusted) {
        throw new Error('Application main frame required')
      }
    }
  )
  expect(() => handler({} as IpcMainInvokeEvent, 'consent', { enabled: true, profile: 'semantier-local' })).toThrow(
    'main frame'
  )
  expect(owner.status().enabled).toBe(false)
  expect(() => handler(trusted, 'consent', { enabled: true, profile: '../other' })).toThrow('Original local profile')
  expect(handler(trusted, 'consent', { enabled: true, profile: 'semantier-local' })).toMatchObject({ enabled: true })
  expect(handler(trusted, 'drain', { profile: 'other' })).toEqual([])
})
it('requires explicit consent and exact restart target; preserves pending through handoff', () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'semantier-observation-'))
  roots.push(dir)
  const owner = new SemantierObservations(dir)

  const target = {
    release_id: 'a'.repeat(32),
    target_sha256: 'b'.repeat(64),
    target_version: '0.0.37.0',
    package_family: 'Semantier.Agent.Acceptance_test',
    build_id: 'a'.repeat(32)
  }

  owner.begin(target, '0.0.36.0')
  expect(owner.drain('semantier-local')).toEqual([])
  owner.consent(true, 'semantier-local')
  owner.begin(target, '0.0.36.0')
  expect(owner.drain('other')).toEqual([])
  expect(owner.drain('semantier-local')[0].outcome).toBe('pending')
  const next = new SemantierObservations(dir)
  next.reconcile({ version: '0.0.38.0', family: target.package_family, buildId: target.build_id })
  expect(next.drain('semantier-local').at(-1)?.outcome).toBe('unknown')
  next.reconcile({ version: target.target_version, family: target.package_family, buildId: target.build_id })
  expect(next.drain('semantier-local').at(-1)?.outcome).toBe('unknown')
  next.verified('c'.repeat(64))
  next.reconcile({ version: target.target_version, family: target.package_family, buildId: target.build_id })
  expect(next.drain('semantier-local').at(-1)?.outcome).toBe('unknown')
  next.verified(target.target_sha256)
  const verifiedRestart = new SemantierObservations(dir)
  verifiedRestart.reconcile({ version: target.target_version, family: target.package_family, buildId: target.build_id })
  expect(verifiedRestart.drain('semantier-local').at(-1)?.outcome).toBe('target_confirmed')
  expect(verifiedRestart.drain('semantier-local').at(-1)).not.toHaveProperty('verified_artifact_sha256')
  next.consent(false, 'semantier-local')
  expect(next.drain('semantier-local')).toEqual([])
})
it('a corrupt optional observation journal cannot prevent startup or update', async () => {
  const fs = await import('node:fs')
  const dir = mkdtempSync(path.join(os.tmpdir(), 'semantier-observation-corrupt-'))
  roots.push(dir)
  fs.writeFileSync(path.join(dir, 'semantier-update-observations.json'), '{broken')
  let owner: SemantierObservations | undefined
  expect(() => {
    owner = new SemantierObservations(dir)
  }).not.toThrow()
  expect(owner?.status().storageUnavailable).toBe(true)
})

it('malformed persisted events fail closed instead of throwing during update', async () => {
  const fs = await import('node:fs')
  const dir = mkdtempSync(path.join(os.tmpdir(), 'semantier-invalid-event-'))
  roots.push(dir)
  fs.writeFileSync(
    path.join(dir, 'semantier-update-observations.json'),
    JSON.stringify({ enabled: true, profile: 'semantier-local', queue: [null], pending: null, dropped: 0 })
  )
  const owner = new SemantierObservations(dir)
  expect(owner.status()).toMatchObject({ enabled: false, storageUnavailable: true })
  expect(owner.drain('semantier-local')).toEqual([])
})
