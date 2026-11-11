import { useStore } from '@nanostores/react'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ComponentProps } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

const discover = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', () => ({
  atom, useValue: useStore,
  Button: ({ variant: _variant, ...props }: ComponentProps<'button'> & { variant?: string }) => <button {...props} />,
  host: { state: { connectionId: atom('cloud'), profile: atom('default') } }
}))
vi.mock('./canonical-group-labels', () => ({ useCanonicalGroupLabels: () => ({ refreshGroups: 'Refresh gateway groups' }) }))
vi.mock('./canonical-groups', () => ({
  captureCanonicalGroupRoute: () => ({ connectionId: 'cloud', profile: 'default' }),
  discoverCanonicalGroups: discover
}))
import { CanonicalGroupList } from './canonical-group-registry'
afterEach(() => { cleanup(); discover.mockReset() })

it('omits unavailable group controls when the host explicitly has no group method', async () => {
  discover.mockRejectedValue(Object.assign(new Error('groups.capabilities unavailable'), { code: -32601 }))
  const { container } = render(<CanonicalGroupList onOpen={vi.fn()} />)
  await waitFor(() => expect(discover).toHaveBeenCalledOnce())
  await waitFor(() => expect(container.innerHTML).toBe(''))
})

it.each(['connection lost', 'permission denied'])('retains retry and error for %s', async message => {
  discover.mockRejectedValue(new Error(message))
  render(<CanonicalGroupList onOpen={vi.fn()} />)
  expect((await screen.findByRole('alert')).textContent).toBe(message)
  expect((screen.getByRole('button', { name: 'Refresh gateway groups' }) as HTMLButtonElement).disabled).toBe(false)
})

it('keeps native groups visible when the authority supports them', async () => {
  discover.mockResolvedValue({ driver: false, rooms: [{ room_id: 'room', name: 'Team', members: [] }] })
  render(<CanonicalGroupList onOpen={vi.fn()} />)
  expect((await screen.findByRole('button', { name: 'Team' }) as HTMLButtonElement).disabled).toBe(false)
})
