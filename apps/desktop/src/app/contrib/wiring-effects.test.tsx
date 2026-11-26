import { renderHook } from '@testing-library/react'
import { expect, it, vi } from 'vitest'

import { useGatewayScopeRefresh } from './wiring-effects'

vi.mock('@/store/profile', () => ({
  refreshActiveProfile: vi.fn()
}))
vi.mock('../right-sidebar/files/use-project-tree', () => ({
  resetProjectTreeState: vi.fn()
}))
vi.mock('./panes', () => ({
  $restartPreviewServer: { get: vi.fn() }
}))

it('preserves the draft override on owner discovery and reconnect, but reseeds on an owner change', () => {
  const refreshModel = vi.fn(async (_force: boolean) => {})
  const refreshConfig = vi.fn(async (_force: boolean) => {})

  const { rerender } = renderHook<void, { connection: string | null; profile: string }>(
    ({ connection, profile }) => useGatewayScopeRefresh(connection, profile, refreshModel, refreshConfig),
    { initialProps: { connection: null, profile: 'default' } }
  )

  rerender({ connection: 'local', profile: 'default' })
  rerender({ connection: null, profile: 'default' })
  rerender({ connection: 'local', profile: 'default' })
  expect(refreshModel.mock.calls).toEqual([[false]])

  rerender({ connection: null, profile: 'default' })
  rerender({ connection: 'cloud', profile: 'default' })
  rerender({ connection: 'cloud', profile: 'research' })
  expect(refreshModel.mock.calls).toEqual([[false], [true], [true]])
})
