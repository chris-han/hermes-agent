import { webcrypto } from 'node:crypto'

import { describe, expect, it } from 'vitest'

import { modelDiscoveryOwnerKey, modelDiscoveryScope } from './model-discovery-scope'

describe('model discovery connection scope', () => {
  it('uses registry identity or the stable local scope', async () => {
    expect(await modelDiscoveryScope('gateway-a', null)).toBe('gateway-a')
    expect(await modelDiscoveryScope(null, { mode: 'local', baseUrl: 'http://localhost:1234' })).toBe('local')
  })

  it('partitions legacy remotes stably without exposing their address or credentials', async () => {
    Object.defineProperty(globalThis.crypto, 'subtle', { configurable: true, value: webcrypto.subtle })
    const remote = {
      mode: 'remote' as const,
      baseUrl: 'https://secret.example/path?token=private',
      remoteIdentity: 'actor'
    }
    const first = await modelDiscoveryScope(null, remote)
    const same = await modelDiscoveryScope(null, { ...remote })
    const second = await modelDiscoveryScope(null, { ...remote, baseUrl: 'https://other.example/path' })
    expect(first).toBe(same)
    expect(first).not.toBe(second)
    expect(first).not.toContain('secret.example')
    expect(first).not.toContain('private')
  })

  it('does not invent a source before a connection exists', async () => {
    await expect(modelDiscoveryScope(null, null)).rejects.toThrow()
  })

  it('uses distinct in-memory keys for unregistered connections without exposing their URLs', () => {
    const first = { mode: 'remote' as const, baseUrl: 'https://first.example' }
    const second = { mode: 'remote' as const, baseUrl: 'https://second.example' }
    expect(modelDiscoveryOwnerKey(null, first)).toBe(modelDiscoveryOwnerKey(null, first))
    expect(modelDiscoveryOwnerKey(null, first)).not.toBe(modelDiscoveryOwnerKey(null, second))
    expect(modelDiscoveryOwnerKey(null, first)).not.toContain('first.example')
  })
})
