import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import { $updateApply, $updateChecking, $updateStatus } from '@/store/updates'

import { UpdateDeliverySourceRow } from './update-delivery-source-row'

vi.mock('@/i18n', () => ({ useI18n: () => ({ locale: 'en' }) }))

const selection = {
  selected: 'channel1' as const,
  options: [
    { id: 'channel1' as const, publicBase: 'https://ecs.example.com' },
    { id: 'channel2' as const, publicBase: 'https://update2.semantier.com' },
    { id: 'channel3' as const, publicBase: 'http://127.0.0.1:18780' }
  ]
}

beforeEach(() => { Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() }) })
afterEach(() => { cleanup(); $updateChecking.set(false); vi.restoreAllMocks() })
test('three choices use the native saved source and discard the previous actionable target', async () => {
  let complete!: (result: unknown) => void
  const save = vi.fn(() => new Promise(resolve => { complete = resolve }))
  window.hermesDesktop = { updates: {
    getDeliverySources: async () => selection,
    setDeliverySource: save
  } } as never
  $updateStatus.set({ supported: true, updateAvailable: true, releaseId: 'old' })
  $updateChecking.set(false)
  $updateApply.set({ ...$updateApply.get(), applying: false })
  render(<UpdateDeliverySourceRow />)
  const trigger = await screen.findByRole('combobox')
  expect(trigger.textContent).toContain('Channel 1')
  fireEvent.keyDown(trigger, { key: 'ArrowDown' })
  expect(await screen.findByRole('option', { name: 'Channel 1 · Alibaba Cloud ECS' })).toBeTruthy()
  expect(screen.getByRole('option', { name: 'Channel 3 · Local development' })).toBeTruthy()
  fireEvent.click(screen.getByRole('option', { name: 'Channel 2 · Cloudflare' }))
  await waitFor(() => expect(save).toHaveBeenCalledWith('channel2'))
  expect($updateStatus.get()).toBeNull()
  expect(trigger.hasAttribute('disabled')).toBe(true)
  complete({ selection: { ...selection, selected: 'channel2' }, status: { supported: true, updateAvailable: false } })
  await waitFor(() => expect(trigger.textContent).toContain('Channel 2'))
  expect(screen.getByText('https://update2.semantier.com')).toBeTruthy()
  expect($updateStatus.get()?.updateAvailable).toBe(false)
})
test('failed save exposes retry and restores persisted selection without an old install action', async () => {
  window.hermesDesktop = { updates: {
    getDeliverySources: async () => selection,
    setDeliverySource: async () => { throw new Error('disk') }
  } } as never
  $updateStatus.set({ supported: true, updateAvailable: true })
  $updateChecking.set(false)
  render(<UpdateDeliverySourceRow />)
  fireEvent.keyDown(await screen.findByRole('combobox'), { key: 'ArrowDown' })
  fireEvent.click(await screen.findByRole('option', { name: 'Channel 2 · Cloudflare' }))
  expect((await screen.findByRole('alert')).textContent).toContain('Could not change update source')
  expect(screen.getByRole('combobox').textContent).toContain('Channel 1')
  expect($updateStatus.get()).toBeNull()
})
