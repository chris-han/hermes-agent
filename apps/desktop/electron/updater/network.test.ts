import { expect, test, vi } from 'vitest'

import { nativeUpdateFetch } from './network'

test('binds an omitted native response URL only with redirect rejection and no cookies', async () => {
  const native = vi.fn(async () => new Response('metadata'))
  const read = nativeUpdateFetch(native)
  const response = await read('https://updates.example.test/channel.json', { redirect: 'error' })
  expect(response.url).toBe('https://updates.example.test/channel.json')
  expect(native).toHaveBeenCalledWith(
    'https://updates.example.test/channel.json',
    expect.objectContaining({ redirect: 'error', credentials: 'omit' })
  )
  await expect(read('https://updates.example.test/channel.json', { redirect: 'follow' })).rejects.toThrow('redirect')
})

test('refuses mismatched or redirected native responses and preserves transport failures', async () => {
  for (const fields of [{ url: 'https://other.example.test/channel.json' }, { redirected: true }]) {
    const response = new Response('untrusted')

    for (const [key, value] of Object.entries(fields)) {
      Object.defineProperty(response, key, { value })
    }
    await expect(
      nativeUpdateFetch(async () => response)('https://updates.example.test/channel.json', { redirect: 'error' })
    ).rejects.toThrow('origin')
  }

  await expect(
    nativeUpdateFetch(async () => {
      throw new Error('redirect refused')
    })('https://updates.example.test/channel.json', { redirect: 'error' })
  ).rejects.toThrow('redirect refused')
})
