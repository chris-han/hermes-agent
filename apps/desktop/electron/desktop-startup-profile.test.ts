import assert from 'node:assert/strict'
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { createDesktopProfilePreferences } from './desktop-profile'
import { prepareDesktopStartupProfile, resolveDesktopStartupProfile } from './desktop-startup-profile'
import { resolveSourcePython } from './source-python'

test('startup commits resolved local scope before returning, but preserves saved remote intent', async () => {
  const events: string[] = []

  const deps = {
    preference: { profile: 'retired', defaultRoute: null },
    explicitProfile: null,
    isLocalConnection: () => false,
    isRemote: () => false,
    resolve: async () => {
      events.push('resolve')

      return { profile: 'semantier-local', recovered: true }
    },
    remember: (profile: string) => {
      events.push(`remember:${profile}`)
    },
    setDefault: (route: { profile: string }) => {
      events.push(`default:${route.profile}`)
    },
    log: () => {}
  }

  await prepareDesktopStartupProfile(deps)
  assert.deepEqual(events, ['resolve', 'remember:semantier-local', 'default:semantier-local'])
  events.length = 0
  await prepareDesktopStartupProfile({
    ...deps,
    preference: {
      profile: 'remote-work',
      defaultRoute: { connectionId: 'remote', profile: 'remote-work' }
    }
  })
  assert.deepEqual(events, [])
})

test('startup refuses unreadable preferences instead of overwriting them with a fallback', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'desktop-startup-preference-'))
  const file = path.join(root, 'active-profile.json')

  try {
    writeFileSync(file, '{broken')
    assert.throws(() => createDesktopProfilePreferences(file).readStartup())
    assert.equal(readFileSync(file, 'utf8'), '{broken')
    writeFileSync(file, JSON.stringify({ profile: 'work', defaultRoute: { connectionId: null, profile: 'work' } }))
    assert.equal(createDesktopProfilePreferences(file).readStartup().defaultRoute?.profile, 'work')
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
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
    const repeated = await Promise.all(
      Array.from({ length: 3 }, () => resolveDesktopStartupProfile(backend, root, null, false))
    )
    assert.ok(repeated.every(result => result.profile === first.profile))
    assert.equal(readFileSync(protectedStore, 'utf8'), 'opaque retained credential')
    await assert.rejects(
      resolveDesktopStartupProfile(backend, root, 'deliberately-missing', true),
      /Local profile unavailable/
    )
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test.skipIf(!python)('real legacy owner retirement repairs remembered and default routes together', async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'desktop-internal-retirement-'))
  const backend = { command: python!, kind: 'python', root: sourceRoot, env: { PYTHONPATH: sourceRoot } }
  const legacy = path.join(root, 'profiles', 'tender-runtime-expert')
  const file = path.join(root, 'active-profile.json')

  try {
    mkdirSync(legacy, { recursive: true })
    writeFileSync(path.join(legacy, 'config.yaml'), 'display:\n  language: en\n')
    writeFileSync(path.join(legacy, '.env'), 'fixture-secret-retained')
    writeFileSync(
      file,
      JSON.stringify({
        profile: 'tender-runtime-expert',
        defaultRoute: { connectionId: null, profile: 'tender-runtime-expert' }
      })
    )
    const preferences = createDesktopProfilePreferences(file)
    await prepareDesktopStartupProfile({
      preference: preferences.readStartup(),
      explicitProfile: null,
      isLocalConnection: () => true,
      isRemote: () => false,
      resolve: (profile, explicit) => resolveDesktopStartupProfile(backend, root, profile, explicit),
      remember: name => {
        preferences.remember(name)
      },
      setDefault: route => {
        preferences.setDefault(route)
      },
      log: () => {}
    })
    assert.equal(preferences.readStartup().profile, 'semantier-local')
    assert.equal(preferences.readStartup().defaultRoute?.profile, 'semantier-local')
    assert.equal(existsSync(legacy), false)
    const batches = readdirSync(path.join(root, '.desktop-retired-profiles'))
    assert.equal(batches.length, 1)
    assert.equal(
      readFileSync(path.join(root, '.desktop-retired-profiles', batches[0], 'tender-runtime-expert', '.env'), 'utf8'),
      'fixture-secret-retained'
    )
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
