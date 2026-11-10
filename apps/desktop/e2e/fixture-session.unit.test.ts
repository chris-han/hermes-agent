import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test, vi } from 'vitest'

import { writeFixtureOwner } from './fixture-environment'
import {
  buttonLabelPattern,
  type FixtureStageRecord,
  recordFixtureStage,
  sanitizeDiagnostic,
  stopFixtureGateway
} from './fixture-session'

test('failed stages persist bounded actual UI diagnostics before rethrowing the original error', async () => {
  const records: FixtureStageRecord[] = []
  const failure = new Error('Missing button at https://user:pass@example.invalid/path?token=private')

  const page = {
    url: () => 'https://example.invalid/?secret=private',
    locator: () => ({
      ariaSnapshot: async () => '- button "[ Begin ]"\n- textbox "password": private\n' + 'x'.repeat(20_000)
    })
  }

  const save = vi.fn(async () => {})
  await expect(
    recordFixtureStage(
      records,
      'confirm',
      async () => {
        throw failure
      },
      () => page as never,
      save
    )
  ).rejects.toBe(failure)
  expect(save).toHaveBeenCalledOnce()
  expect(records[0]).toMatchObject({ name: 'confirm', status: 'failed' })
  expect(records[0].accessibility).toContain('[ Begin ]')
  expect(records[0].accessibility!.length).toBeLessThanOrEqual(12_000)
  expect(JSON.stringify(records)).not.toContain('private')
  expect(JSON.stringify(records)).not.toContain('user:pass')
})

test('button names match normal and decorated labels without matching another action', () => {
  expect(buttonLabelPattern('Begin').test('[ Begin ]')).toBe(true)
  expect(buttonLabelPattern('Begin').test('Begin')).toBe(true)
  expect(buttonLabelPattern('Begin').test('Begin again')).toBe(false)
  expect(buttonLabelPattern('A+B').test('A+B')).toBe(true)
  expect(sanitizeDiagnostic('Authorization: Bearer synthetic-token')).not.toContain('synthetic-token')
})

test('a real gateway-stop spawn failure retains its home', async () => {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'fixture-stop-')))
  const source = path.join(root, 'source')

  const sandbox = {
    root,
    hermesHome: path.join(root, 'hermes-home'),
    userDataDir: path.join(root, 'electron-user-data'),
    cleanup: vi.fn()
  }

  fs.mkdirSync(sandbox.hermesHome)
  fs.mkdirSync(sandbox.userDataDir)
  fs.mkdirSync(source)
  writeFixtureOwner(sandbox, source)

  const env = {
    HOME: root,
    HERMES_HOME: sandbox.hermesHome,
    HERMES_DESKTOP_ISOLATED_BACKEND: '1',
    HERMES_DESKTOP_PYTHON: path.join(root, 'nonexistent-python')
  }

  try {
    await expect(stopFixtureGateway(source, sandbox, env)).rejects.toThrow(/ENOENT/)
    expect(fs.existsSync(sandbox.hermesHome)).toBe(true)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
