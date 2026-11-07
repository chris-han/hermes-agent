/** Local updater custody only: independent of optional reporting/telemetry. */
import { randomUUID } from 'node:crypto'
import { lstat, mkdir, readdir, readFile, rename, rm, unlink, writeFile } from 'node:fs/promises'
import path from 'node:path'

export interface ChannelArtifactBinding {
  manifestSha256: string
  artifactSha256: string
  buildId: string
  version: string
  identity: string
  publisher: string
}
interface CacheRecord extends ChannelArtifactBinding {
  schema: 1
  phase: 'downloading' | 'handoff'
}
const namePattern = /^\.channel-artifact-[a-f0-9]{64}-[a-f0-9]{64}$/
const name = (binding: ChannelArtifactBinding): string =>
  `.channel-artifact-${binding.manifestSha256}-${binding.artifactSha256}`

function validRecord(record: CacheRecord): boolean {
  return !(
    !record ||
    record.schema !== 1 ||
    !['downloading', 'handoff'].includes(record.phase) ||
    !/^[a-f0-9]{32}$/.test(record.buildId) ||
    !/^\d+\.\d+\.\d+\.\d+$/.test(record.version) ||
    !/^[a-f0-9]{64}$/.test(record.manifestSha256) ||
    !/^[a-f0-9]{64}$/.test(record.artifactSha256) ||
    typeof record.identity !== 'string' ||
    !record.identity ||
    record.identity.length > 160 ||
    typeof record.publisher !== 'string' ||
    !record.publisher ||
    record.publisher.length > 512
  )
}

async function records(root: string): Promise<CacheRecord[]> {
  const entries = await readdir(root).catch((error: NodeJS.ErrnoException): string[] => {
    if (error.code !== 'ENOENT') {
      throw error
    }

    return []
  })

  const result: CacheRecord[] = []

  for (const entry of entries) {
    if (!entry.endsWith('.json') || !namePattern.test(entry.slice(0, -5))) {
      continue
    }
    const file = path.join(root, entry)
    const info = await lstat(file)

    if (!info.isFile() || info.isSymbolicLink() || info.size > 4096) {
      throw new Error('Update cache ownership requires recovery')
    }
    const record: CacheRecord = JSON.parse(await readFile(file, 'utf8'))

    if (!validRecord(record) || entry !== name(record) + '.json') {
      throw new Error('Update cache ownership requires recovery')
    }
    result.push(record)
  }

  return result
}

async function save(root: string, record: CacheRecord): Promise<void> {
  const destination = path.join(root, name(record) + '.json')
  const temporary = destination + '.' + randomUUID() + '.tmp'

  try {
    await writeFile(temporary, JSON.stringify(record) + '\n', { flag: 'wx', mode: 0o600 })
    await rename(temporary, destination)
  } finally {
    await unlink(temporary).catch((error: NodeJS.ErrnoException): void => {
      if (error.code !== 'ENOENT') {
        throw error
      }
    })
  }
}

async function retire(root: string, record: CacheRecord): Promise<void> {
  const directory = path.join(root, name(record))

  try {
    const info = await lstat(directory)

    if (!info.isDirectory() || info.isSymbolicLink()) {
      throw new Error('Update cache custody requires recovery')
    }
    await rm(directory, { recursive: true })
  } catch (error) {
    if (!(error instanceof Error) || !('code' in error)) {
      throw error
    }

    if (error.code === 'EBUSY' || error.code === 'EPERM') {
      return
    } // Retain external journal for later recovery.

    if (error.code !== 'ENOENT') {
      throw error
    }
  }

  await unlink(path.join(root, name(record) + '.json'))
}

/** Called inside the existing exclusive update apply; handed-off packages are held. */
export async function prepareChannelArtifactCache(root: string, binding: ChannelArtifactBinding): Promise<string> {
  if (!validRecord({ ...binding, schema: 1, phase: 'downloading' })) {
    throw new Error('Invalid update cache binding')
  }
  await mkdir(root, { recursive: true })
  const existing = await records(root)

  for (const record of existing) {
    if (name(record) !== name(binding) && record.phase === 'downloading') {
      await retire(root, record)
    }
  }

  const current = (await records(root)).find(record => name(record) === name(binding))

  if (current && Object.keys(binding).some(key => current[key] !== binding[key])) {
    throw new Error('Update cache binding mismatch')
  }

  if (!current && (await records(root)).length >= 3) {
    throw new Error('pending_update_installation')
  }
  const directory = path.join(root, name(binding))

  try {
    const info = await lstat(directory)

    if (!info.isDirectory() || info.isSymbolicLink()) {
      throw new Error('Update cache custody requires recovery')
    }

    if (!current) {
      throw new Error('Update cache ownership requires recovery')
    }
  } catch (error) {
    if (!(error instanceof Error) || !('code' in error) || error.code !== 'ENOENT') {
      throw error
    }

    if (!current) {
      await save(root, { ...binding, schema: 1, phase: 'downloading' })
    }
    await mkdir(directory, { mode: 0o700 })
  }

  return directory
}

/** Persist handoff intent before native open; unknown acceptance retains custody. */
export async function markChannelArtifactHandoff(
  root: string,
  file: string
): Promise<(() => Promise<void>) | undefined> {
  const directory = path.dirname(file)

  if (path.dirname(directory) !== root || !namePattern.test(path.basename(directory))) {
    return
  }
  const record = (await records(root)).find(item => name(item) === path.basename(directory))

  if (!record) {
    throw new Error('Update cache ownership requires recovery')
  }
  await save(root, { ...record, phase: 'handoff' })

  return async (): Promise<void> => {
    await save(root, record)
  }
}

/** Actual packaged identity/version comes from the existing native inspector. */
export async function retireInstalledChannelArtifacts(
  root: string,
  actual: Pick<ChannelArtifactBinding, 'buildId' | 'version' | 'identity' | 'publisher'>
): Promise<void> {
  for (const record of await records(root)) {
    if (record.identity !== actual.identity || record.publisher !== actual.publisher) {
      continue
    }
    const version = record.version.split('.').map(Number)
    const installed = actual.version.split('.').map(Number)
    const difference = version.map((part, index) => part - installed[index]).find(part => part !== 0) ?? 0

    if (difference < 0 || (difference === 0 && record.buildId === actual.buildId)) {
      await retire(root, record)
    }
  }
}
