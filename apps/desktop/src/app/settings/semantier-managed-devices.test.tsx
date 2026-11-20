import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { expect, it, vi } from 'vitest'

import { SemantierManagedDevices } from './semantier-managed-devices'

vi.mock('@/i18n', () => ({ useI18n: () => ({ locale: 'en' }) }))
const scope = { userId: 'user-a', workspaceId: 'workspace-a', organizationId: 'org-a' }

const device = {
  device_id: 'device-a',
  status: 'active',
  assurance: 'UNKNOWN',
  created_at: '2026-11-20T00:00:00Z',
  last_seen_at: '2026-11-20T00:00:00Z',
  revoked_at: null
}

it('reads server devices on demand and requires deliberate removal confirmation', async () => {
  const managedDevices = vi
    .fn()
    .mockResolvedValue({ included_windows_seats: 2, devices: [device], additional_seat_checkout: 'unavailable' })

  const managedDeviceRevoke = vi.fn().mockResolvedValue({ device_id: 'device-a', status: 'revoked' })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { workspace: { managedDevices, managedDeviceRevoke } }
  })
  render(<SemantierManagedDevices origin="https://cloud.example" scope={scope} />)
  expect(managedDevices).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'View Windows PCs' }))
  expect(await screen.findByText('device-a')).toBeTruthy()
  expect(screen.getByText('Hardware not verified')).toBeTruthy()
  expect(screen.queryByText('TPM verified by server')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Remove PC' }))
  expect(managedDeviceRevoke).not.toHaveBeenCalled()
  managedDevices.mockResolvedValue({
    included_windows_seats: 2,
    devices: [{ ...device, status: 'revoked' }],
    additional_seat_checkout: 'unavailable'
  })
  fireEvent.click(screen.getByRole('button', { name: 'Confirm removal' }))
  await waitFor(() => expect(managedDeviceRevoke).toHaveBeenCalledWith('https://cloud.example', scope, 'device-a'))
  expect(await screen.findByText('Revoked')).toBeTruthy()
})

it('clears the old account inventory and ignores its delayed response', async () => {
  let resolve!: (value: unknown) => void

  const managedDevices = vi.fn().mockReturnValue(
    new Promise(r => {
      resolve = r
    })
  )

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: { managedDevices } } })
  const view = render(<SemantierManagedDevices origin="https://cloud.example" scope={scope} />)
  fireEvent.click(screen.getByRole('button', { name: 'View Windows PCs' }))
  view.rerender(<SemantierManagedDevices origin="https://cloud.example" scope={{ ...scope, userId: 'user-b' }} />)
  await act(async () => {
    resolve({ included_windows_seats: 2, devices: [device], additional_seat_checkout: 'unavailable' })
  })
  expect(screen.queryByText('device-a')).toBeNull()
})
