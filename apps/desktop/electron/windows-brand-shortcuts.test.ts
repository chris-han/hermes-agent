import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createBrandShortcutService, pngsToIco, validateShortcutName, type ShortcutServiceOptions } from './windows-brand-shortcuts'
import type { DesktopBranding } from './desktop-branding-types'

const mocks = vi.hoisted(() => ({ failWrite: false, reveal: vi.fn(), png: Buffer.alloc(0) }))
vi.mock('./product-identity', () => ({ PRODUCT_IDENTITY: { appId: 'test', msixAppIdWithOrg: 'test' } }))
vi.mock('electron', () => {
  const image = (width = 32, height = 32): any => ({
    isEmpty: () => false,
    getSize: () => ({ width, height }),
    resize: ({ width, height }: any) => image(width, height),
    toBitmap: () => Buffer.alloc(width * height * 4),
    toPNG: () => mocks.png
  })
  return {
    app: { getAppPath: () => '/unused' },
    nativeImage: {
      createFromBuffer: () => image(),
      createFromDataURL: () => image(),
      createFromBitmap: (_b: Buffer, size: any) => image(size.width, size.height)
    },
    shell: {
      showItemInFolder: mocks.reveal,
      readShortcutLink: (file: string) => JSON.parse(fs.readFileSync(file, 'utf8')),
      writeShortcutLink: (file: string, _op: string, details: unknown) => {
        if (mocks.failWrite) {
          return false
        }
        fs.writeFileSync(file, JSON.stringify(details))
        return true
      }
    }
  }
})

let root: string
let branding: DesktopBranding
beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'brand-shortcuts-'))
  mocks.failWrite = false
  mocks.reveal.mockClear()
  mocks.png = fs.readFileSync(new URL('../src/assets/semantier-logo.png', import.meta.url))
  branding = {
    version: 1,
    displayName: 'Research',
    displayNameZh: '研究',
    logo: `data:image/png;base64,${mocks.png.toString('base64')}`
  }
})
afterEach(() => fs.rmSync(root, { recursive: true, force: true }))
const roots = () => ({ desktop: path.join(root, 'Desktop'), startMenu: path.join(root, 'Start') })
const service = (target: ShortcutServiceOptions['target'] = async () => ({ target: 'C:\\app.exe', args: '', appUserModelId: 'test' })) =>
  createBrandShortcutService({ directory: path.join(root, 'state'), roots: roots(), branding: () => branding, target })

describe('branding shortcuts', () => {
  it('accepts Chinese and English names and refuses paths and reserved Windows filenames', () => {
    expect(validateShortcutName('司序科技')).toBe('司序科技')
    expect(validateShortcutName('Semantier Agent')).toBe('Semantier Agent')
    for (const name of ['', 'CON', 'nul.txt', '../app', 'C:\\app', 'name.', ' name', 'a/b']) {
      expect(() => validateShortcutName(name)).toThrow('Invalid shortcut name')
    }
  })
  it('encodes independent PNG icon entries with correct sizes and offsets', () => {
    const ico = pngsToIco([
      { size: 16, png: Buffer.from('small') },
      { size: 256, png: Buffer.from('large') }
    ])
    expect(ico.readUInt16LE(2)).toBe(1)
    expect(ico.readUInt16LE(4)).toBe(2)
    expect(ico[6]).toBe(16)
    expect(ico[22]).toBe(0)
    expect(ico.readUInt32LE(18)).toBe(38)
    expect(ico.readUInt32LE(34)).toBe(43)
    expect(ico.subarray(38).toString()).toBe('smalllarge')
  })
  it('creates and updates owned links without overwriting a foreign shortcut', async () => {
    const instance = service()
    const request = { name: 'Research', destinations: ['desktop', 'startMenu'] as const, branding }
    expect(await instance.create({ ...request, destinations: [...request.destinations] })).toEqual([
      { destination: 'desktop', status: 'created' },
      { destination: 'startMenu', status: 'created' }
    ])
    const file = path.join(roots().desktop, 'Research.lnk')
    const actual = JSON.parse(fs.readFileSync(file, 'utf8'))
    expect(fs.readFileSync(actual.icon).readUInt16LE(4)).toBe(7)
    expect(await instance.create({ ...request, destinations: ['desktop'] })).toEqual([
      { destination: 'desktop', status: 'updated' }
    ])
    instance.reveal('Research', 'desktop')
    expect(mocks.reveal).toHaveBeenCalledWith(file)
    fs.writeFileSync(file, 'unrelated')
    expect(await instance.create({ ...request, destinations: ['desktop'] })).toEqual([
      { destination: 'desktop', status: 'conflict' }
    ])
    expect(fs.readFileSync(file, 'utf8')).toBe('unrelated')
    expect(() => instance.reveal('Research', 'desktop')).toThrow()
  })
  it('keeps the successful location when the other fails and retries without duplicating it', async () => {
    fs.writeFileSync(roots().startMenu, 'not a directory')
    const instance = service()
    expect(await instance.create({ name: 'Research', destinations: ['desktop', 'startMenu'], branding })).toEqual([
      { destination: 'desktop', status: 'created' },
      { destination: 'startMenu', status: 'failed' }
    ])
    fs.unlinkSync(roots().startMenu)
    expect(await instance.create({ name: 'Research', destinations: ['startMenu'], branding })).toEqual([
      { destination: 'startMenu', status: 'created' }
    ])
    expect(fs.readdirSync(roots().desktop)).toEqual(['Research.lnk'])
  })
  it('rejects stale branding and unknown destinations before writes', async () => {
    const instance = service(async () => {
      branding = { ...branding, displayName: 'New' }
      return { target: 'C:\\app.exe' }
    })
    await expect(instance.create({ name: 'Research', destinations: ['desktop'], branding })).rejects.toThrow(
      'Branding changed'
    )
    expect(fs.existsSync(roots().desktop)).toBe(false)
    await expect(service().create({ name: 'Research', destinations: ['other'] as any, branding })).rejects.toThrow(
      'Invalid shortcut destination'
    )
  })
  it('retains the existing link after a failed native write and refuses revealing arbitrary names', async () => {
    const instance = service()
    await instance.create({ name: 'Research', destinations: ['desktop'], branding })
    const file = path.join(roots().desktop, 'Research.lnk')
    const before = fs.readFileSync(file)
    mocks.failWrite = true
    expect(await instance.create({ name: 'Research', destinations: ['desktop'], branding })).toEqual([
      { destination: 'desktop', status: 'failed' }
    ])
    expect(fs.readFileSync(file)).toEqual(before)
    expect(() => instance.reveal('../other', 'desktop')).toThrow()
  })
})
