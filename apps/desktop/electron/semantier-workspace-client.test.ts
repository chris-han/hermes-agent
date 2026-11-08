import { expect, it, vi } from 'vitest'

import { createSemantierWorkspaceClient, workspaceTranscript, workspaceDesktopRead, requireWorkspaceCompletion } from './semantier-workspace-client'

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

it('projects server conversation and message IDs without replacement or precision loss', async () => {
  const history = { sessionId: 'session_original', messages: [
    { messageId: '204', role: 'assistant', text: 'Reminder', timestamp: 1794102384000, metadata: { author_bot_id: 'bot_a' } },
    { messageId: '9007199254740993', role: 'user', text: 'Hello' }
  ] }
  const rows = workspaceTranscript(history)
  expect(rows[0]).toMatchObject({ messageId: '204', row_id: 204, content: 'Reminder', timestamp: 1794102384, display_metadata: { author_bot_id: 'bot_a' } })
  expect(rows[1].messageId).toBe('9007199254740993')
  expect(rows[1]).not.toHaveProperty('row_id')
  const transport = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify(history)))
  const client = createSemantierWorkspaceClient('https://cloud.example', transport)
  const page = await workspaceDesktopRead(client, '/api/sessions/session_original/messages?order=latest')
  expect(page).toMatchObject({ session_id: 'session_original', total: 2 })
  await expect(workspaceDesktopRead(client, '/api/config')).rejects.toMatchObject({ statusCode: 404 })
})

it('requires the actual completed SSE terminal and rejects failed or uncertain outcomes', async () => {
  await expect(requireWorkspaceCompletion(new Response('event: done\r\ndata: {"state":"complete"}\r\n\r\n'))).resolves.toBeUndefined()
  await expect(requireWorkspaceCompletion(new Response('event: done\ndata: {"state":"error"}\n\n'))).rejects.toThrow('failed')
  await expect(requireWorkspaceCompletion(new Response('event: delta\ndata: {"text":"unfinished"}\n\n'))).rejects.toThrow('uncertain')
})
