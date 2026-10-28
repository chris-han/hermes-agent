import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { lstat, mkdir, open, rename, unlink } from 'node:fs/promises'
import type { FileHandle } from 'node:fs/promises'
import path from 'node:path'

import { type ArtifactChunks, channelPublicBase, decodeArtifactChunks } from './channel-protocol'

export interface PinnedArtifact {
  url: string
  sha256: string
  size: number
  format: 'zip' | 'msix' | 'msixbundle'
  chunks?: ArtifactChunks
  chunkBase?: string
}

export async function verifyPinnedArtifact(file: string, artifact: PinnedArtifact): Promise<void> {
  const info = await lstat(file)

  if (!info.isFile() || info.isSymbolicLink() || info.size !== artifact.size) {
    throw new Error('Pinned artifact size or type mismatch')
  }

  const hash = createHash('sha256')

  for await (const chunk of createReadStream(file)) {
    hash.update(chunk)
  }

  if (hash.digest('hex') !== artifact.sha256) {
    throw new Error('Pinned artifact digest mismatch')
  }
}

/** Caller owns private storage and its lifetime. Redirects cannot change authority. */
export async function downloadPinnedArtifact(directory: string, artifact: PinnedArtifact, fetchArtifact: typeof fetch = fetch): Promise<string> {
  const file: string = path.join(directory, `destination.${artifact.format}`)

  try {
    await verifyPinnedArtifact(file, artifact)

    return file
  } catch (error) {
    if (!(error instanceof Error) || !('code' in error) || error.code !== 'ENOENT') {
      throw error
    }
  }

  if (artifact.chunks) {
    return await downloadChunkedArtifact(directory, artifact, fetchArtifact)
  }

  const temporary: string = `${file}.partial`
  await unlink(temporary).catch((error: NodeJS.ErrnoException): void => {
    if (error.code !== 'ENOENT') {
      throw error
    }
  })
  const signal = AbortSignal.timeout(600_000)
  let size = 0
  let lastError: unknown

  const retryable = (error: unknown): boolean => !signal.aborted &&
    (error instanceof TypeError || (error instanceof Error && error.message.startsWith('net::ERR_')))

  // Retry only interrupted transport, within the original overall time limit.
  // Partial bytes never become an artifact until the complete digest verifies.
  for (let attempt = 0; attempt < 4; attempt++) {
    let response: Response

    try {
      response = await fetchArtifact(artifact.url, {
        redirect: 'error', signal,
        headers: size ? { Range: `bytes=${size}-` } : undefined
      })
    } catch (error) {
      if (!retryable(error)) { throw error }
      lastError = error

      continue
    }

    if (!response.ok || !response.body) {
      throw new Error(`Pinned artifact download failed (${response.status})`)
    }

    const resumed = size > 0 && response.status === 206

    if (response.status !== 200 && (!resumed || response.headers.get('Content-Range') !== `bytes ${size}-${artifact.size - 1}/${artifact.size}`)) {
      await response.body.cancel()
      throw new Error('Pinned artifact resume range mismatch')
    }

    if (!resumed && size) {
      await unlink(temporary)
      size = 0
    }

    if (resumed) {
      const partial = await lstat(temporary)

      if (!partial.isFile() || partial.isSymbolicLink() || partial.size !== size) {
        throw new Error('Pinned artifact partial size or type mismatch')
      }
    }

    const handle: FileHandle = await open(temporary, resumed ? 'a' : 'wx', 0o600)
    const reader = response.body.getReader()
    let interrupted = false

    try {
      for (;;) {
        let chunk: ReadableStreamReadResult<Uint8Array>

        try { chunk = await reader.read() }
        catch (error) {
          if (!retryable(error)) { throw error }
          lastError = error
          interrupted = true

          break
        }

        if (chunk.done) { break }
        size += chunk.value.byteLength

        if (size > artifact.size) { throw new Error('Pinned artifact download exceeds pinned size') }
        await handle.writeFile(chunk.value)
      }

      await handle.sync()
    } finally {
      await handle.close()
      await reader.cancel().catch(() => undefined)
    }

    if (interrupted && size < artifact.size) {
      // A response may fail before delivering its first byte.
      if (!size) { await unlink(temporary) }

      continue
    }

    await verifyPinnedArtifact(temporary, artifact)
    await rename(temporary, file)

    return file
  }

  throw lastError ?? new Error('Pinned artifact transfer interrupted')
}

/** Chunk state is scoped to the complete digest; verified chunks survive retries. */
async function downloadChunkedArtifact(directory: string, artifact: PinnedArtifact, fetchArtifact: typeof fetch): Promise<string> {
  const descriptor = decodeArtifactChunks(artifact.chunks, artifact)
  const base = channelPublicBase(artifact.chunkBase ?? '')

  if (!artifact.url.startsWith(`${base}/releases/`) || new URL(artifact.url).origin !== new URL(base).origin) {
    throw new Error('Artifact chunk origin mismatch')
  }

  const parts: string[] = []

  for (const [index, chunk] of descriptor.chunks.entries()) {
    const storage = path.join(directory, `chunks-${artifact.sha256}`, String(index))

    for (const folder of [path.dirname(storage), storage]) {
      try {
        const info = await lstat(folder)

        if (!info.isDirectory() || info.isSymbolicLink()) { throw new Error('Artifact chunk storage type mismatch') }
      } catch (error) {
        if (!(error instanceof Error) || !('code' in error) || error.code !== 'ENOENT') { throw error }
        await mkdir(folder, { mode: 0o700 })
      }
    }

    parts.push(await downloadPinnedArtifact(storage, {
      url: `${base}/${chunk.key}`, sha256: chunk.sha256, size: chunk.size, format: artifact.format
    }, fetchArtifact))
  }

  const file = path.join(directory, `destination.${artifact.format}`)
  const temporary = `${file}.partial`
  await unlink(temporary).catch((error: NodeJS.ErrnoException): void => {
    if (error.code !== 'ENOENT') { throw error }
  })
  const output = await open(temporary, 'wx', 0o600)

  try {
    for (const part of parts) {
      for await (const body of createReadStream(part)) { await output.writeFile(body) }
    }

    await output.sync()
  } finally { await output.close() }

  await verifyPinnedArtifact(temporary, artifact)
  await rename(temporary, file)

  return file
}
