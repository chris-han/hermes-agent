import { expect, it } from 'vitest'

import { connectionDialFieldsChanged, mergeConnectionInput, normalizeConnectionInput, normalizeRegistry } from './connection-registry'

it('retains explicit HTTP provenance without migrating legacy Nous entries', () => {
  const registry = normalizeRegistry(null)
  const http = normalizeConnectionInput({ kind: 'cloud', label: 'Semantier Cloud', url: 'https://cloud.semantier.cn', transport: 'semantier-workspace-http' }, registry)
  const legacy = normalizeConnectionInput({ kind: 'cloud', label: 'Existing Nous', url: 'https://agent.example' }, registry)
  const restored = normalizeRegistry({ ...registry, connections: [...registry.connections, http, legacy] })
  expect(restored.connections.find(x => x.id === http.id)?.transport).toBe('semantier-workspace-http')
  expect(restored.connections.find(x => x.id === legacy.id)?.transport).toBeUndefined()
  expect(mergeConnectionInput({ id: http.id, kind: 'cloud', label: 'My Cloud' }, http).transport).toBe('semantier-workspace-http')
  expect(connectionDialFieldsChanged(legacy, { ...legacy, transport: 'semantier-workspace-http' })).toBe(true)
})

it('rejects workspace HTTP with gateway secrets, insecure origins or a local kind', () => {
  const registry = normalizeRegistry(null)
  const input = { kind: 'cloud' as const, label: 'Cloud', url: 'https://cloud.example', transport: 'semantier-workspace-http' as const }
  expect(() => normalizeConnectionInput({ ...input, token: 'gateway-token' }, registry)).toThrow('browser session')
  expect(() => normalizeConnectionInput({ ...input, url: 'http://cloud.example' }, registry)).toThrow('secure')
  expect(() => normalizeConnectionInput({ ...input, kind: 'local' }, registry)).toThrow('Cloud')
})
