import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { markChannelArtifactHandoff, prepareChannelArtifactCache, retireInstalledChannelArtifacts } from './channel-artifact-cache'

const binding = { manifestSha256: 'a'.repeat(64), artifactSha256: 'b'.repeat(64), buildId: 'c'.repeat(32), version: '0.0.55.0', identity: 'Semantier.Agent.Acceptance', publisher: 'CN=Semantier Technology' }

test('superseded unfinished download is retired but handed-off bytes survive until installed confirmation', async (): Promise<void> => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'channel-cache-'))

  try {
    const first = await prepareChannelArtifactCache(root, binding)
    await writeFile(path.join(first, 'destination.msix'), 'verified')
    const secondBinding = { ...binding, manifestSha256: 'd'.repeat(64), artifactSha256: 'e'.repeat(64), buildId: 'f'.repeat(32), version: '0.0.56.0' }
    const second = await prepareChannelArtifactCache(root, secondBinding)
    await expect(readFile(path.join(first, 'destination.msix'))).rejects.toThrow()
    await writeFile(path.join(second, 'destination.msix'), 'verified-new')
    await markChannelArtifactHandoff(root, path.join(second, 'destination.msix'))
    const third = await prepareChannelArtifactCache(root, binding)
    expect(await readFile(path.join(second, 'destination.msix'), 'utf8')).toBe('verified-new')
    await retireInstalledChannelArtifacts(root, { buildId: secondBinding.buildId, version: secondBinding.version, identity: binding.identity, publisher: binding.publisher })
    expect(await readdir(root)).toEqual([])
    expect(third).toBe(first)
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('an unknown handoff cannot grow cache beyond three candidates or erase unrelated data', async (): Promise<void> => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'channel-cache-hold-'))

  try {
    await writeFile(path.join(root, 'user-file'), 'preserve')

    for (const digit of ['1', '2', '3']) {
      const directory = await prepareChannelArtifactCache(root, { ...binding, manifestSha256: digit.repeat(64) })
      await writeFile(path.join(directory, 'destination.msix'), 'bytes')
      await markChannelArtifactHandoff(root, path.join(directory, 'destination.msix'))
    }

    await expect(prepareChannelArtifactCache(root, binding)).rejects.toThrow('pending_update_installation')
    expect(await readFile(path.join(root, 'user-file'), 'utf8')).toBe('preserve')
    await retireInstalledChannelArtifacts(root, { ...binding, publisher: 'different' })
    expect((await readdir(root)).length).toBe(7)
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('a linked custody marker fails closed without touching its target', async (): Promise<void> => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'channel-cache-link-'))

  try {
    const outside = path.join(root, 'unrelated.json')
    await writeFile(outside, 'preserve')
    const { symlink } = await import('node:fs/promises')
    await symlink(outside, path.join(root, `.channel-artifact-${binding.manifestSha256}-${binding.artifactSha256}.json`))
    await expect(prepareChannelArtifactCache(root, binding)).rejects.toThrow('ownership')
    expect(await readFile(outside, 'utf8')).toBe('preserve')
  } finally { await rm(root, { recursive: true, force: true }) }
})
