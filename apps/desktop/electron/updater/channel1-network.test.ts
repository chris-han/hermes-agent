import { expect, test, vi } from 'vitest'

import { channel1ArtifactFetch, createDesktopUpdateFetches } from './channel1-network'

const url = `https://assets.semantier.cn/releases/channel-builds/${'a'.repeat(32)}/SemantierAgent.msix`

test('production wiring preserves cloud metadata and permits standard OSS HTTPS', async () => {
  const system = vi.fn(async () => new Response('metadata'))
  const partition = {setProxy: vi.fn(async () => undefined), fetch: vi.fn(async () => new Response('package'))}
  const sessions = {fromPartition: vi.fn(() => partition)}
  const transport = createDesktopUpdateFetches('win32', system, sessions)
  await transport.metadata('https://cloud.semantier.cn/releases/channels/semantier-acceptance.json', {redirect:'error'})
  expect(sessions.fromPartition).not.toHaveBeenCalled()
  await transport.artifact(url, {redirect:'error'})
  expect(sessions.fromPartition).toHaveBeenCalledWith('semantier-channel1-artifacts', {cache:false})
  expect(system).toHaveBeenCalledTimes(1)
})

test.each([url, 'https://assets.semantier.cn/releases/channel-builds/1.2.131.4/SemantierAgent.msix'])('Channel 1 packages use an isolated direct session (%s)', async (url) => {
  const fallback = vi.fn(async () => new Response(null, { status: 421 }))
  const direct = { setProxy: vi.fn(async () => undefined), fetch: vi.fn(async () => new Response('package')) }
  const create = vi.fn(() => direct)
  const read = channel1ArtifactFetch(fallback, create)
  expect((await read(url, { redirect: 'error' })).status).toBe(200)
  expect(direct.setProxy).toHaveBeenCalledWith({ mode: 'direct' })
  expect(direct.fetch).toHaveBeenCalledWith(url, expect.objectContaining({ credentials: 'omit', redirect: 'error' }))
  expect(fallback).not.toHaveBeenCalled()
  await read(url, { redirect: 'error' })
  expect(create).toHaveBeenCalledTimes(1)
  await read('https://download.semantier.com/other.msix', { redirect: 'error' })
  expect(fallback).toHaveBeenCalledTimes(1)
})

test('direct failure never falls back to proxy or another protocol and redirects remain forbidden', async () => {
  const fallback = vi.fn()

  const read = channel1ArtifactFetch(fallback, () => ({setProxy: async () => undefined,
    fetch: async () => { throw new Error('OSS network unavailable') }}))

  await expect(read(url, { redirect: 'error' })).rejects.toThrow('OSS network unavailable')
  expect(fallback).not.toHaveBeenCalled()
})

test('canonical package transport rejects malformed paths, foreign redirects and cancellation without fallback', async () => {
  const fallback = vi.fn()

  const direct = vi.fn(async () => {
    const response = new Response('foreign')
    Object.defineProperty(response, 'url', { value: 'https://foreign.example/package.msix' })

    return response
  })

  const read = channel1ArtifactFetch(fallback, () => ({setProxy: async () => undefined, fetch: direct}))
  await expect(read(url+'?replacement=1', {redirect:'error'})).rejects.toThrow('URL')
  await expect(read(url, {redirect:'error'})).rejects.toThrow('origin')
  const controller = new AbortController()
  controller.abort()

  const cancelled = channel1ArtifactFetch(fallback, () => ({
    setProxy: async () => undefined,
    fetch: async (_input, init) => {
      init?.signal?.throwIfAborted()

      return new Response('unreachable')
    }
  }))

  await expect(cancelled(url, {redirect:'error', signal:controller.signal})).rejects.toThrow()
  expect(fallback).not.toHaveBeenCalled()
})
