import { createHash } from 'node:crypto'
import { mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { resolveRegisteredChannelApp, runChannelCommand, verifyChannelDownload } from './channel-native'

test('registered package identity survives a volume junction without admitting escaped or undeclared executables', async (): Promise<void> => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'channel-registration-'))
  const physical = path.join(directory, 'volume', 'package')
  const alias = path.join(directory, 'WindowsApps')
  const executable = path.join(physical, 'desktop.exe')
  const outside = path.join(directory, 'outside.exe')

  try {
    await mkdir(physical, { recursive: true })
    await writeFile(executable, 'desktop')
    await writeFile(outside, 'unrelated')
    await symlink(physical, alias, 'junction')

    const registered = {
      identity: 'Acceptance',
      signer: 'retained-publisher',
      appPath: alias,
      nativeVersion: '0.0.39.0',
      executable,
      packageFullName: 'Acceptance_0.0.39.0',
      packageFamilyName: 'Acceptance_family',
      removalRoots: [alias],
      applications: [{ id: 'Desktop', executable: path.join(alias, 'desktop.exe') }]
    }

    const resolved = await resolveRegisteredChannelApp(registered, executable)
    expect(resolved.appPath).toBe(physical)
    expect(resolved.applicationId).toBe('Desktop')
    await expect(resolveRegisteredChannelApp(registered, outside)).rejects.toThrow('outside registered package')
    await expect(resolveRegisteredChannelApp({ ...registered, applications: [] }, executable)).rejects.toThrow(
      'not the registered application'
    )
    await expect(
      resolveRegisteredChannelApp(
        { ...registered, applications: [...registered.applications, ...registered.applications] },
        executable
      )
    ).rejects.toThrow('not the registered application')
    await symlink(outside, path.join(physical, 'escaped.exe'))
    await expect(resolveRegisteredChannelApp(registered, path.join(physical, 'escaped.exe'))).rejects.toThrow(
      'outside registered package'
    )
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('download verification binds native bytes to the manifest size and digest before signature preparation', async (): Promise<void> => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'channel-artifact-'))
  const file = path.join(directory, 'update.zip')

  try {
    const content = Buffer.from('real local archive fixture bytes')
    await writeFile(file, content)

    const artifact = {
      key: 'releases/fixture.zip',
      size: content.length,
      sha256: createHash('sha256').update(content).digest('hex')
    }

    await verifyChannelDownload([file], artifact)
    await writeFile(file, Buffer.alloc(content.length))
    await expect(verifyChannelDownload([file], artifact)).rejects.toThrow('digest')
    await expect(verifyChannelDownload([], artifact)).rejects.toThrow('one')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('native failure exposes a bounded exit reason without command paths or stderr payload', async (): Promise<void> => {
  const failure = await runChannelCommand(process.execPath, [
    '-e',
    'process.stderr.write("secret-canary C:/Users/Private/token"); process.exit(7)'
  ]).catch((error: Error) => error)
  expect(failure).toBeInstanceOf(Error)
  expect((failure as Error).message).toBe('Native channel operation failed (exit code 7)')
  expect((failure as Error).message).not.toContain('secret-canary')
  expect((failure as Error).cause).toBeInstanceOf(Error)
})
