import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test } from 'vitest'

import { assertFixtureOwner, fixturePythonEnv, stripCredentials, writeFixtureOwner } from './fixture-environment'

const roots: string[] = []
afterEach(() => {
  for (const root of roots) {fs.rmSync(root, { recursive: true, force: true })}
})

test('maps the selected Python to the actual Desktop override and strips inherited credentials', () => {
  expect(fixturePythonEnv({ HERMES_PYTHON: '/chosen/python' })).toEqual({
    HERMES_PYTHON: '/chosen/python',
    HERMES_DESKTOP_PYTHON: '/chosen/python'
  })
  expect(fixturePythonEnv({ HERMES_PYTHON: '/other', HERMES_DESKTOP_PYTHON: '/explicit' }).HERMES_DESKTOP_PYTHON).toBe(
    '/explicit'
  )
  expect(
    stripCredentials({ OPENAI_API_KEY: 'synthetic', HERMES_HOME: '/real/home', HERMES_YOLO_MODE: '1', PATH: '/bin' })
  ).toEqual({ PATH: '/bin' })
})

test('validates owned roots before gateway stop or deletion and rejects escaped/symlinked homes', () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'fixture-owner-'))
  roots.push(base)
  const root = path.join(base, 'sandbox')
  fs.mkdirSync(root)

  const sandbox = {
    root,
    hermesHome: path.join(root, 'hermes-home'),
    userDataDir: path.join(root, 'electron-user-data')
  }

  fs.mkdirSync(sandbox.hermesHome)
  fs.mkdirSync(sandbox.userDataDir)
  writeFixtureOwner(sandbox, '/source')
  expect(assertFixtureOwner(root, '/source', base)).toEqual(sandbox)
  expect(() => assertFixtureOwner(root, '/other-source', base)).toThrow(/owner/)
  fs.rmSync(sandbox.hermesHome, { recursive: true })
  fs.symlinkSync(base, sandbox.hermesHome, process.platform === 'win32' ? 'junction' : 'dir')
  expect(() => assertFixtureOwner(root, '/source', base)).toThrow(/symlink/)
})
