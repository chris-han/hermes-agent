import { expect, it, vi } from 'vitest'

import { ManagedDeviceEnrollment } from './managed-device-enrollment'
import type { ManagedDeviceEnrollmentProof, ManagedDeviceReceipt } from './semantier-workspace-client'

const scope = { userId: 'u', workspaceId: 'w', organizationId: 'o' }

const identity = {
  authenticated: true,
  user: { user_id: 'u', workspace_id: 'w' },
  workspace: { id: 'w' },
  organization_id: 'o'
}

function fixture() {
  let state: unknown
  const writes: unknown[] = []

  const client = {
    identity: vi.fn(async () => identity),
    managedDevices: vi.fn(async () => ({
      included_windows_seats: 2,
      devices: [],
      additional_seat_checkout: 'unavailable',
      enrollment_policy: 'hardware_attested' as 'hardware_attested' | 'development_key_proof',
      enrollment_available: true
    })),
    managedDeviceChallenge: vi.fn(async (command: any) => ({
      context: { user_id: 'u', organization_id: 'o', workspace_id: 'w', challenge_id: 'ch', ...command },
      assurance_required: 'TPM_ATTESTED'
    })),
    managedDeviceEnroll: vi.fn(async (_proof: ManagedDeviceEnrollmentProof): Promise<ManagedDeviceReceipt> => ({
      device_id: 'md_server',
      status: 'active' as const,
      assurance: 'TPM_ATTESTED',
      capacity_revision: 'fixture'
    }))
  }

  const key = vi.fn(async (request: any) => ({
    key_id: request.key_id,
    public_key: 'PUBLIC',
    signature: 'SIGNATURE',
    assurance: 'NOT_VERIFIED'
  }))

  const options = {
    client: () => client,
    key,
    read: () => state,
    write: (value: unknown) => {
      state = value
      writes.push(value)
    },
    attest: vi.fn(async () => ({ fixture: 'synthetic verifier evidence; not actual attestation' }))
  }

  return { options, client, key, writes, getState: () => state }
}

it('requires current scope and an attestation provider before creating a key', async () => {
  const f = fixture()
  const controller = new ManagedDeviceEnrollment({ ...f.options, attest: undefined })
  await expect(controller.enroll('https://cloud.example', scope)).rejects.toThrow('attestation')
  expect(f.key).not.toHaveBeenCalled()
  expect(f.writes).toHaveLength(0)
})
it('persists a pending key before enrollment and reuses it after a lost response', async () => {
  const f = fixture()
  const controller = new ManagedDeviceEnrollment(f.options)
  f.client.managedDeviceEnroll.mockRejectedValueOnce(new Error('response lost'))
  await expect(controller.enroll('https://cloud.example', scope)).rejects.toThrow('response lost')
  const first = JSON.stringify(f.getState())
  expect(first).not.toContain('device_id')
  const id = f.key.mock.calls[0][0].key_id
  const result = await controller.enroll('https://cloud.example', scope)
  expect(result.device_id).toBe('md_server')
  expect(f.key.mock.calls.filter(([r]) => r.operation === 'create')).toHaveLength(1)
  expect(f.key.mock.calls.every(([r]) => r.key_id === id)).toBe(true)
  expect(f.client.managedDeviceChallenge.mock.calls[1][0].intent).toBe('reconnect')
})
it('does not recreate a missing key or adopt a foreign account response', async () => {
  const f = fixture()
  const controller = new ManagedDeviceEnrollment(f.options)
  await controller.enroll('https://cloud.example', scope)
  f.key.mockRejectedValueOnce(new Error('native key unavailable'))
  await expect(controller.enroll('https://cloud.example', scope)).rejects.toThrow('native key unavailable')
  expect(f.key.mock.calls.filter(([r]) => r.operation === 'create')).toHaveLength(1)
  f.client.identity.mockResolvedValueOnce({ ...identity, user: { user_id: 'foreign', workspace_id: 'w' } })
  await expect(controller.enroll('https://cloud.example', scope)).rejects.toThrow('changed')
})

it('serializes concurrent enrollment and preserves bindings for another origin', async () => {
  const f = fixture()
  const controller = new ManagedDeviceEnrollment(f.options)
  await Promise.all([
    controller.enroll('https://cloud.example', scope),
    controller.enroll('https://cloud.example', scope)
  ])
  expect(f.key.mock.calls.filter(([r]) => r.operation === 'create')).toHaveLength(1)
  await controller.enroll('https://other.example', scope)
  expect(Object.keys((f.getState() as { bindings: object }).bindings)).toHaveLength(2)
})
it('recovers the exact pending replacement after losing its response', async () => {
  const f = fixture()
  const controller = new ManagedDeviceEnrollment(f.options)
  await controller.enroll('https://cloud.example', scope)
  f.client.managedDeviceEnroll.mockRejectedValueOnce(new Error('replacement response lost'))
  await expect(controller.enroll('https://cloud.example', scope, 'md_server')).rejects.toThrow(
    'replacement response lost'
  )
  f.client.managedDeviceEnroll.mockResolvedValueOnce({
    device_id: 'md_replacement',
    status: 'active',
    assurance: 'TPM_ATTESTED',
    capacity_revision: 'fixture'
  })
  expect((await controller.enroll('https://cloud.example', scope)).device_id).toBe('md_replacement')
  expect(f.key.mock.calls.filter(([r]) => r.operation === 'create')).toHaveLength(2)
  expect(f.client.managedDeviceChallenge.mock.calls.at(-1)![0].intent).toBe('reconnect')
})
it('refuses corrupt recovery data instead of creating a new key', async () => {
  const f = fixture()
  const controller = new ManagedDeviceEnrollment({ ...f.options, read: () => null })
  await expect(controller.enroll('https://cloud.example', scope)).rejects.toThrow('recovery data')
  expect(f.key).not.toHaveBeenCalled()
})

it('uses server-selected DEVELOPMENT key proof without inventing attestation', async () => {
  const f = fixture()
  f.client.managedDevices.mockResolvedValue({
    ...(await f.client.managedDevices()),
    enrollment_policy: 'development_key_proof'
  })
  f.client.managedDeviceEnroll.mockResolvedValue({
    device_id: 'md_dev',
    status: 'active',
    assurance: 'NOT_VERIFIED',
    hardware_attestation: 'NOT_VERIFIED',
    hardware_verified: false,
    capacity_revision: 'fixture'
  })
  const controller = new ManagedDeviceEnrollment({ ...f.options, attest: undefined })
  const receipt = await controller.enroll('https://cloud.example', scope)
  expect(receipt.hardware_verified).toBe(false)
  expect(f.options.attest).not.toHaveBeenCalled()
  expect(f.client.managedDeviceEnroll.mock.calls[0][0]).not.toHaveProperty('attestation')
})

it('recovers a quota-refused pending key into an explicit replacement only after a server miss', async () => {
  const f = fixture()
  const controller = new ManagedDeviceEnrollment(f.options)
  f.client.managedDeviceEnroll.mockRejectedValueOnce(
    Object.assign(new Error('DEVICE_SEAT_LIMIT'), { statusCode: 409, managedDeviceCode: 'DEVICE_SEAT_LIMIT' })
  )
  await expect(controller.enroll('https://cloud.example', scope)).rejects.toThrow('DEVICE_SEAT_LIMIT')
  f.client.managedDeviceEnroll.mockRejectedValueOnce(
    Object.assign(new Error('device_unavailable'), { statusCode: 404, managedDeviceCode: 'device_unavailable' })
  )
  await controller.enroll('https://cloud.example', scope, 'md_old')
  expect(f.client.managedDeviceChallenge.mock.calls.at(-1)![0]).toMatchObject({
    intent: 'replace',
    replaces_device_id: 'md_old'
  })
  expect(f.key.mock.calls.filter(([r]) => r.operation === 'create')).toHaveLength(1)
})
