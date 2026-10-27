import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, expect, test } from 'vitest'

import type { ChannelBuild } from './channel-protocol'
import { assertPackagedChannelSelection, packagedChannelSelection, savePackagedChannel } from './semantier-channel'
const root = mkdtempSync(join(tmpdir(), 'semantier-channel-'))
afterEach(() => rmSync(join(root, 'updates.json'), { force: true }))

const build: ChannelBuild = {
  buildId: 'a'.repeat(32), channel: 'semantier-acceptance', sequence: 46,
  repository: 'chris-han/hermes-agent', commit: 'b'.repeat(40), sourceVersion: '1.0.0',
  version: '0.0.46', windowsVersion: '0.0.46.0', publicBase: 'https://updates.example.test', bundleEnv: {},
  identity: { token: 'b0fdc2a7e4b08d16', displayName: 'Semantier Agent Acceptance', appId: 'com.semantier.agent.acceptance',
    appNamePascal: 'SemantierAgentAcceptance', artifactNamePascal: 'SemantierAgent', cliName: 'semantier-agent-acceptance',
    windowsExecutableName: 'Semantier Agent', msixAppIdWithOrg: 'Semantier.Agent.Acceptance' }
}

test('selection persists across restart without mutating installed provenance or other preferences', () => {
  const path = join(root, 'updates.json')
  writeFileSync(path, JSON.stringify({ branch: 'custom', other: 123 }))
  expect(packagedChannelSelection(build, path)?.choice).toBe('development')
  savePackagedChannel(path, 'production', (file, text) => writeFileSync(file, text))
  const selected = packagedChannelSelection(build, path)
  expect(selected?.choice).toBe('production')
  expect(selected?.build.channel).toBe('semantier-production')
  expect(selected?.build.identity).toEqual(build.identity)
  expect(selected?.build.buildId).toBe(build.buildId)
  expect(build.channel).toBe('semantier-acceptance')
  expect(JSON.parse(readFileSync(path, 'utf8'))).toMatchObject({ branch: 'custom', other: 123 })
})
test('unrelated packages stay governed by their own channel; invalid persisted choices fail clearly', () => {
  const path = join(root, 'updates.json')
  expect(packagedChannelSelection({ ...build, repository: 'NousResearch/hermes-agent' }, path)).toBeNull()
  writeFileSync(path, JSON.stringify({ releaseChannel: 'untrusted-channel' }))
  expect(() => packagedChannelSelection(build, path)).toThrow('channel')
  expect(() => savePackagedChannel(path, 'https://evil.test', () => {})).toThrow('channel')
})
test('a configured HTTPS deployment origin is validated without losing package identity', () => {
  const path = join(root, 'updates.json')
  expect(packagedChannelSelection(build, path, 'https://updates.example.test/new')?.build.publicBase).toBe('https://updates.example.test/new')
  expect(() => packagedChannelSelection(build, path, 'http://remote.example.test')).toThrow()
})

test('a stale window cannot install from a different saved channel', () => {
  expect(() => assertPackagedChannelSelection('production', 'development')).toThrow('Check for updates again')
  expect(() => assertPackagedChannelSelection('production', undefined)).toThrow()
  expect(() => assertPackagedChannelSelection('production', 'production')).not.toThrow()
})
