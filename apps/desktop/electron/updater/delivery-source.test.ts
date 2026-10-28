import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { DeliverySources } from './delivery-source'
import { UpdateOperation } from './operation'

test('three independent origins persist without changing the build stamp and reject invalid endpoints', () => {
  const root = mkdtempSync(path.join(tmpdir(), 'update-sources-'))

  try {
    const file = path.join(root, 'source.json')
    const config = path.join(root, 'config.yaml')
    writeFileSync(config, 'updates:\n  desktop_feed_base_url: https://ecs.example.com\n  desktop_local_feed_base_url: http://127.0.0.1:18780\n')
    const source = new DeliverySources(file, config, 'http://127.0.0.1:18780')
    expect(source.status().selected).toBe('channel3')
    expect(source.status().options.map(row => row.id)).toEqual(['channel1', 'channel2', 'channel3'])
    source.select('channel2')
    expect(new DeliverySources(file, config, 'http://127.0.0.1:18780').base()).toBe('https://s3.semantier.com')
    source.select('channel1')
    expect(source.base()).toBe('https://ecs.example.com')
    expect(() => source.select('https://evil.example')).toThrow()
    writeFileSync(config, 'updates:\n  desktop_local_feed_base_url: http://evil.example\n')
    source.select('channel3')
    expect(() => source.base()).toThrow(/loopback/)
  } finally { rmSync(root, { recursive: true, force: true }) }
})

test('reset discards old target and is refused during apply and after handoff', async () => {
  let initializations = 0

  const operation = new UpdateOperation(async () => { initializations += 1;

 return null })

  await operation.resolve()
  operation.reset()
  await operation.resolve()
  expect(initializations).toBe(2)
  await operation.apply(async () => {
    expect(() => operation.reset()).toThrow(/progress/)

    return { ok: true, handedOff: true }
  })
  expect(() => operation.reset()).toThrow(/progress/)
})


test('legacy Cloudflare stamp retains source selection while resolving the new public origin', () => {
  const root = mkdtempSync(path.join(tmpdir(), 'update-source-migration-'))

  try {
    const source = new DeliverySources(path.join(root, 'source.json'), path.join(root, 'config.yaml'), 'https://update2.semantier.com')
    expect(source.status().selected).toBe('channel2')
    expect(source.base()).toBe('https://s3.semantier.com')
    expect(source.status().options.find(row => row.id === 'channel1')?.publicBase).toBe('')
  } finally { rmSync(root, { recursive: true, force: true }) }
})
