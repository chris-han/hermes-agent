import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { expect, it, vi } from 'vitest'

import { SemantierWinmanCapability } from './semantier-winman-capability'

vi.mock('@/i18n', () => ({ useI18n: () => ({ locale: 'en' }) }))

it('requires an explicit request and never labels unavailable execution as on', async () => {
  const origin = 'https://cloud.example'
  const scope = { userId: 'a', workspaceId: 'wa', organizationId: 'oa' }
  const read = vi.fn().mockResolvedValue({ desired: { version: 1, enabled: false, revision: 0 }, execution: 'unavailable' })
  const write = vi.fn().mockResolvedValue({ desired: { version: 1, enabled: true, revision: 1 }, execution: 'unavailable' })
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: { winmanCapability: read, setWinmanCapability: write } } })
  render(<SemantierWinmanCapability origin={origin} scope={scope} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Request enablement' }))
  expect(write).not.toHaveBeenCalled()
  expect(screen.getByText(/incur model costs/)).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Confirm request' }))
  await waitFor(() => expect(write).toHaveBeenCalledWith(origin, scope, true))
  expect(await screen.findByText(/Enablement requested. Cloud execution is unavailable/)).toBeTruthy()
})

it('refreshes the same account preference after another client changes it', async () => {
  const origin = 'https://cloud.example'
  const scope = { userId: 'a', workspaceId: 'wa', organizationId: 'oa' }
  const read = vi.fn().mockResolvedValueOnce({ desired: { version: 1, enabled: false, revision: 0 }, execution: 'unavailable' })
    .mockResolvedValueOnce({ desired: { version: 1, enabled: true, revision: 1 }, execution: 'unavailable' })
  const write = vi.fn()
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: { winmanCapability: read, setWinmanCapability: write } } })
  render(<SemantierWinmanCapability origin={origin} scope={scope} />)
  expect(await screen.findByText(/Preference off/)).toBeTruthy()
  fireEvent.focus(window)
  expect(await screen.findByText(/Enablement requested/)).toBeTruthy()
  expect(write).not.toHaveBeenCalled()
})

it('does not refresh over an unresolved preference write', async () => {
  const scope = { userId: 'a', workspaceId: 'wa', organizationId: 'oa' }
  const read = vi.fn().mockResolvedValue({ desired: { version: 1, enabled: false, revision: 0 }, execution: 'unavailable' })
  let complete!: (value: unknown) => void
  const write = vi.fn().mockImplementation(() => new Promise(resolve => {complete = resolve}))
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: { winmanCapability: read, setWinmanCapability: write } } })
  render(<SemantierWinmanCapability origin="https://cloud.example" scope={scope} />)
  fireEvent.click(await screen.findByRole('button', { name: 'Request enablement' }))
  fireEvent.click(screen.getByRole('button', { name: 'Confirm request' }))
  fireEvent.focus(window)
  expect(read).toHaveBeenCalledTimes(1)
  complete({ desired: { version: 1, enabled: true, revision: 1 }, execution: 'unavailable' })
  expect(await screen.findByText(/Enablement requested/)).toBeTruthy()
})
