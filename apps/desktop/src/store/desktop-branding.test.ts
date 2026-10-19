import { afterEach, expect, test, vi } from 'vitest'

import type { DesktopBranding } from '../../electron/desktop-branding-types'
const original = window.hermesDesktop
afterEach(() => {
  window.hermesDesktop = original
  vi.resetModules()
})
test('a delayed startup read and save response cannot overwrite a newer peer publication', async () => {
  vi.resetModules()
  let read!: (value: DesktopBranding) => void
  let write!: (value: DesktopBranding) => void
  let publish!: (value: DesktopBranding) => void
  const initial: DesktopBranding = { version: 1, displayName: 'Initial', logo: null }
  window.hermesDesktop = {
    ...original,
    branding: {
      initial,
      get: () =>
        new Promise(resolve => {
          read = resolve
        }),
      set: () =>
        new Promise(resolve => {
          write = resolve
        }),
      onChanged: callback => {
        publish = callback

        return () => {}
      }
    }
  }
  const store = await import('./desktop-branding')
  const peer = { ...initial, displayName: 'Peer' }
  publish(peer)
  read(initial)
  await Promise.resolve()
  expect(store.$desktopBranding.get()).toEqual(peer)
  const saving = store.saveDesktopBranding({ ...initial, displayName: 'Mine' })
  const newer = { ...initial, displayName: 'Newer peer' }
  publish(newer)
  write({ ...initial, displayName: 'Mine' })
  expect(await saving).toEqual(newer)
  expect(store.$desktopBranding.get()).toEqual(newer)
})
