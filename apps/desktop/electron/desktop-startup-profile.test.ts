import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { createDesktopProfilePreferences } from './desktop-profile'
import { resolveDesktopStartupProfile } from './desktop-startup-profile'
import { resolveSourcePython } from './source-python'

test('startup refuses unreadable preferences instead of overwriting them with a fallback', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'desktop-startup-preference-'))
  const file = path.join(root, 'active-profile.json')

  try {
    writeFileSync(file, '{broken')
    assert.throws(() => createDesktopProfilePreferences(file).readStartup())
    assert.equal(readFileSync(file, 'utf8'), '{broken')
    writeFileSync(file, JSON.stringify({ profile: 'work', defaultRoute: { connectionId: null, profile: 'work' } }))
    assert.equal(createDesktopProfilePreferences(file).readStartup().defaultRoute?.profile, 'work')
  } finally { rmSync(root, { recursive: true, force: true }) }
})

const sourceRoot = path.resolve('../..')
const python = resolveSourcePython(sourceRoot)

test.skipIf(!python)('native bridge resolves the real Python profile owner without booting a gateway', async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'desktop-startup-owner-'))
  const backend = { command: python!, kind: 'python', root: sourceRoot, env: { PYTHONPATH: sourceRoot } }

  try {
    const first = await resolveDesktopStartupProfile(backend, root, 'registered-retired', false)
    assert.equal(first.profile, 'semantier-local')
    const protectedStore = path.join(root, 'profiles', first.profile, '.cloud-device.bin')
    writeFileSync(protectedStore, 'opaque retained credential')
    const repeated = await Promise.all(Array.from({ length: 3 }, () => resolveDesktopStartupProfile(backend, root, null, false)))
    assert.ok(repeated.every(result => result.profile === first.profile))
    assert.equal(readFileSync(protectedStore, 'utf8'), 'opaque retained credential')
    await assert.rejects(resolveDesktopStartupProfile(backend, root, 'deliberately-missing', true), /Local profile unavailable/)
  } finally { rmSync(root, { recursive: true, force: true }) }
})
