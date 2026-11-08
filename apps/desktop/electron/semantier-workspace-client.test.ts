import { expect, it, vi } from 'vitest'

import { createSemantierWorkspaceClient } from './semantier-workspace-client'

it('refuses insecure remote origins and URL credentials', () => {
  for (const origin of ['http://cloud.example', 'https://user:secret@cloud.example', 'https://cloud.example/other']) {
    expect(() => createSemantierWorkspaceClient(origin, fetch)).toThrow()
  }
})

it('does not adopt a foreign history placeholder as the requested session', async () => {
  const transport = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({ sessionId: 'new', messages: [] })))
  const client = createSemantierWorkspaceClient('https://cloud.example', transport)
  await expect(client.history('session_original')).rejects.toMatchObject({ statusCode: 404 })
})

it('does not retry an uncertain send and retains its request identity', async () => {
  const transport = vi.fn<typeof fetch>().mockRejectedValue(new TypeError('Connection lost'))
  const client = createSemantierWorkspaceClient('https://cloud.example', transport)
  await expect(client.send('session_original', 'Hello', 'request-001')).rejects.toThrow('Connection lost')
  expect(transport).toHaveBeenCalledTimes(1)
  const [url, options] = transport.mock.calls[0]
  expect(url).toBe('https://cloud.example/api/send-stream')
  expect(options).toMatchObject({ credentials: 'include', redirect: 'error', method: 'POST' })
  expect(JSON.parse(String(options?.body))).toMatchObject({ sessionKey: 'session_original', requestIdentity: 'request-001', idempotencyKey: 'request-001' })
})

it('marks revoked authentication for recovery without retry', async () => {
  const transport = vi.fn<typeof fetch>().mockResolvedValue(new Response('', { status: 403 }))
  const client = createSemantierWorkspaceClient('https://cloud.example', transport)
  await expect(client.identity()).rejects.toMatchObject({ statusCode: 403, needsWorkspaceLogin: true })
  expect(transport).toHaveBeenCalledTimes(1)
})
