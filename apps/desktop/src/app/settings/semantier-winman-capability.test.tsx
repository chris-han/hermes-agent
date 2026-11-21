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
