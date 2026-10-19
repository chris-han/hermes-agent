import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test, vi } from 'vitest'
vi.mock('electron', () => ({
  nativeImage: {
    createFromBuffer: () => ({
      isEmpty: () => false,
      getSize: () => ({ width: 64, height: 64 }),
      toPNG: () => Buffer.from('canonical')
    })
  }
}))
import { createDesktopBrandingStore } from './desktop-branding'
const homes: string[] = []
afterEach(() => {
  for (const dir of homes.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

function fixture() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-brand-'))
  homes.push(dir)

  return path.join(dir, 'brand.json')
}

test('atomic commits publish successful writes, serialize peers and survive cold reads/reset', async () => {
  const file = fixture()
  const changed: unknown[] = []
  const store = createDesktopBrandingStore(file, record => changed.push(record))
  expect(store.get().displayName).toBe('')
  await Promise.all([
    store.set({ version: 1, displayName: 'First', logo: null }),
    store.set({ version: 1, displayName: '组织', logo: null })
  ])
  expect(changed).toHaveLength(2)
  expect(createDesktopBrandingStore(file).get().displayName).toBe('组织')
  fs.mkdirSync(file + '.tmp')
  await expect(store.set({ version: 1, displayName: 'Lost', logo: null })).rejects.toThrow()
  expect(store.get().displayName).toBe('组织')
  expect(changed).toHaveLength(2)
  fs.rmdirSync(file + '.tmp')
  await store.set({ version: 1, displayName: '', logo: null })
  expect(createDesktopBrandingStore(file).get()).toEqual({ version: 1, displayName: '', logo: null })
})
test('rejects invalid schemas and spoofed image bytes without changing prior state', async () => {
  const store = createDesktopBrandingStore(fixture())

  for (const input of [
    null,
    { version: 2 },
    { version: 1, displayName: 'x'.repeat(121), logo: null },
    { version: 1, displayName: 'x', logo: 'https://evil/logo.png' },
    { version: 1, displayName: 'x', logo: 'data:image/png;base64,PHN2Zz4=' }
  ]) {
    await expect(store.set(input)).rejects.toThrow()
    expect(store.get().displayName).toBe('')
  }
})
test('corrupt saved data reports load failure', () => {
  const file = fixture()
  fs.writeFileSync(file, '{')
  expect(() => createDesktopBrandingStore(file).get()).toThrow()
})
