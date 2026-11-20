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
    .mockResolvedValue({
      included_windows_seats: 2,
      devices: [device],
      additional_seat_checkout: 'unavailable',
      seat_summary: {
        included_windows_seats: 2,
        purchased_windows_seats: 0,
        development_windows_seats: 1,
        licensed_windows_seats: 3,
        active_windows_devices: 1,
        new_enrollment_limit: 2,
        retained_above_capacity: 0,
        capacity_revision: 'synthetic.v1'
      }
    })

  const managedDeviceRevoke = vi.fn().mockResolvedValue({ device_id: 'device-a', status: 'revoked' })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { workspace: { managedDevices, managedDeviceRevoke } }
  })
  render(<SemantierManagedDevices origin="https://cloud.example" scope={scope} />)
  expect(managedDevices).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'View Windows PCs' }))
  expect(await screen.findByText('device-a')).toBeTruthy()
  expect(screen.getByText('2 included · 0 purchased additional seats')).toBeTruthy()
  expect(screen.getByText('DEVELOPMENT test seats: 1 (no purchase)')).toBeTruthy()
  expect(screen.getByText('Current enrollment ceiling: 2 PCs')).toBeTruthy()
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
