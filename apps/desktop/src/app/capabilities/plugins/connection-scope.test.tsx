import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $pluginRecords } from '@/contrib/plugins-store'
import { queryClient } from '@/lib/query-client'
import { $agentPlugins, $agentPluginsOwner } from '@/store/agent-plugins'

import { PluginsTab } from './plugins-tab'

const { ambient, routed } = vi.hoisted(() => ({ ambient: vi.fn(), routed: vi.fn() }))
vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({ useGatewayRequest: () => ({ requestGateway: ambient }) }))
vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  requestGatewayForAgent: routed
}))

const row = {
  name: 'Remote research',
  key: 'research',
  source: 'git',
  status: 'enabled',
  version: '1',
  description: 'Remote profile configuration',
  settings_schema: [
    { key: 'topic', type: 'string', label: 'Research topic', description: '', required: false, value: 'before' }
  ]
}

afterEach(() => {
  cleanup()
  queryClient.clear()
  vi.clearAllMocks()
})
it('routes native list and schema settings to the pinned backend instead of the active gateway', async () => {
  $pluginRecords.set({})
  routed.mockImplementation(async (_connection, _profile, method, params) => {
    if (method === 'plugins.catalog.list') {
      return { available: false, entries: [] }
    }

    return params.action === 'list' ? { plugins: [row] } : { ok: true, plugin: row }
  })
  render(<PluginsTab profile={{ connectionId: 'remote-a', profile: 'worker' }} />)
  fireEvent.click(await screen.findByRole('button', { name: /Settings.*Remote research/i }))
  fireEvent.change(screen.getByRole('textbox', { name: /Research topic/ }), { target: { value: 'after' } })
  fireEvent.click(screen.getByRole('button', { name: /Save/ }))
  await waitFor(() =>
    expect(routed).toHaveBeenCalledWith(
      'remote-a',
      'worker',
      'plugins.manage',
      expect.objectContaining({ action: 'settings', profile: 'worker', values: { topic: 'after' } }),
      undefined
    )
  )
  expect(ambient).not.toHaveBeenCalled()
  expect($agentPluginsOwner.get()).toBe('remote-a::worker')
})

it('does not show cached agent rows from a different connection with the same profile name', async () => {
  $pluginRecords.set({})
  $agentPlugins.set([row as never])
  $agentPluginsOwner.set('remote-a::worker')
  routed.mockImplementation(() => new Promise(() => {}))
  render(<PluginsTab profile={{ connectionId: 'remote-b', profile: 'worker' }} />)
  expect(screen.queryByText('Remote research', { exact: true })).toBeNull()
})
