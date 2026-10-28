import { createHash } from 'node:crypto'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { downloadPinnedArtifact } from './artifact'
import { decodeArtifactChunks } from './channel-protocol'

const sha = (body: Buffer): string => createHash('sha256').update(body).digest('hex')
const buildId = 'a'.repeat(32)
const bytes = Buffer.from('signed-package-exact-bytes')
const facts = { size: bytes.length, sha256: sha(bytes) }

const descriptor = {
  schema: 1 as const, buildId, artifact: facts, chunkSize: 134217728,
  chunks: [{ key: `releases/channel-builds/${buildId}/chunks/${facts.sha256}/000000.part`, offset: 0, ...facts }]
}

test('chunk decoder rejects mixed, escaping, incomplete and unsupported descriptors', (): void => {
  expect(decodeArtifactChunks(descriptor, facts)).toEqual(descriptor)

  for (const bad of [
    { ...descriptor, schema: 2 }, { ...descriptor, buildId: '../escape' },
    { ...descriptor, chunkSize: 512 }, { ...descriptor, chunks: [] },
    { ...descriptor, artifact: { ...facts, sha256: 'b'.repeat(64) } },
    { ...descriptor, chunks: [{ ...descriptor.chunks[0], offset: 1 }] },
    { ...descriptor, chunks: [{ ...descriptor.chunks[0], key: 'https://other.example/app' }] },
    { ...descriptor, chunks: [descriptor.chunks[0], descriptor.chunks[0]] }
  ]) { expect(() => decodeArtifactChunks(bad, facts)).toThrow() }
})

test('pinned artifact download assembles verified public chunk bytes and reuses verified final file', async (): Promise<void> => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'artifact-chunks-'))
  let reads = 0

  const fetchArtifact: typeof fetch = async (input): Promise<Response> => {
    expect(input).toBe(`https://s3.semantier.com/${descriptor.chunks[0].key}`)
    reads++

    return new Response(bytes)
  }

  try {
    const artifact = { url: `https://s3.semantier.com/releases/channel-builds/${buildId}/app.msix`, ...facts, format: 'msix' as const, chunkBase: 'https://s3.semantier.com', chunks: descriptor }
    const result = await downloadPinnedArtifact(directory, artifact, fetchArtifact)
    expect(await readFile(result)).toEqual(bytes)
    expect(await downloadPinnedArtifact(directory, artifact, fetchArtifact)).toBe(result)
    expect(reads).toBe(1)
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('chunk corruption never produces a verified package or requests the whole object', async (): Promise<void> => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'artifact-corrupt-'))

  try {
    const artifact = { url: `https://s3.semantier.com/releases/channel-builds/${buildId}/app.msix`, ...facts, format: 'msix' as const, chunkBase: 'https://s3.semantier.com', chunks: descriptor }

    const corrupt: typeof fetch = async (input): Promise<Response> => {
      expect(input).toContain('/chunks/')

      return new Response(Buffer.alloc(bytes.length))
    }

    await expect(downloadPinnedArtifact(directory, artifact, corrupt)).rejects.toThrow('digest')
    await expect(readFile(path.join(directory, 'destination.msix'))).rejects.toThrow()
    await expect(downloadPinnedArtifact(directory, { ...artifact, chunkBase: 'https://other.example' }, corrupt)).rejects.toThrow('origin')
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('a new download invocation reuses a completed 128 MiB chunk after interruption', async (): Promise<void> => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'artifact-restart-'))
  const block = new Uint8Array(65536)
  const chunkSize = 134217728
  const first = createHash('sha256')
  const complete = createHash('sha256')

  for (let count = 0; count < chunkSize / block.length; count++) { first.update(block); complete.update(block) }
  const tail = new Uint8Array([1, 2, 3])
  const total = { size: chunkSize + tail.length, sha256: complete.update(tail).digest('hex') }
  const prefix = `releases/channel-builds/${buildId}/chunks/${total.sha256}/`

  const chunks = {
    schema: 1 as const, buildId, artifact: total, chunkSize,
    chunks: [
      { key: prefix + '000000.part', offset: 0, size: chunkSize, sha256: first.digest('hex') },
      { key: prefix + '000001.part', offset: chunkSize, size: tail.length, sha256: sha(Buffer.from(tail)) }
    ]
  }

  const reads: string[] = []
  let interrupted = true

  const transfer: typeof fetch = async (input): Promise<Response> => {
    const key = String(input)
    reads.push(key)

    if (key.endsWith('000001.part')) {
      if (interrupted) { return new Response(null, { status: 503 }) }

      return new Response(tail)
    }

    let sent = 0

    return new Response(new ReadableStream<Uint8Array>({
      pull(controller): void {
        if (sent === chunkSize) { controller.close();

 return }

        controller.enqueue(block)
        sent += block.length
      }
    }))
  }

  const artifact = { url: `https://s3.semantier.com/releases/channel-builds/${buildId}/app.msix`, ...total, format: 'msix' as const, chunkBase: 'https://s3.semantier.com', chunks }

  try {
    await expect(downloadPinnedArtifact(directory, artifact, transfer)).rejects.toThrow('503')
    interrupted = false
    const file = await downloadPinnedArtifact(directory, artifact, transfer)
    expect(file).toBe(path.join(directory, 'destination.msix'))
    expect(reads.map(key => key.slice(-11))).toEqual(['000000.part', '000001.part', '000001.part'])
  } finally { await rm(directory, { recursive: true, force: true }) }
})
