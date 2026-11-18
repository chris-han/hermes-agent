import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { CapabilitySearchResults } from './search-results'

const { rpc, hub } = vi.hoisted(() => ({ rpc: vi.fn(), hub: vi.fn() }))
vi.mock('@/store/gateway', async original => ({ ...(await original<object>()), requestGatewayForAgent: rpc }))
vi.mock('@/hermes', async original => ({ ...(await original<object>()), searchSkillsHub: hub }))
vi.mock('./mcp/use-mcp-servers', () => ({
  useMcpServers: () => ({
    availableCatalog: [
      {
        name: 'match-connector',
        description: 'Matching connector',
        command: 'example',
        args: [],
        env: {},
        required_env: []
      }
    ],
    catalog: [],
    servers: {},
    statuses: {},
    toolCounts: {},
    usageByServer: {},
    configLoading: false,
    catalogLoading: false,
    configFailed: false,
    refetchConfig: vi.fn()
  })
}))
vi.mock('./connectors/data/queries', () => ({
  useHostedConnectors: () => ({ rows: [], titles: {}, phase: 'ready', refetch: vi.fn() }),
  usePluginServers: () => []
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})
it('returns all five kinds together and opens the exact result without a mutation', async () => {
  rpc.mockImplementation(async (_connection, _profile, method) =>
    method === 'plugins.manage'
      ? {
          plugins: [
            {
              key: 'match-plugin',
              name: 'match-plugin',
              description: 'Matching plugin',
              source: 'user',
              status: 'enabled'
            }
          ]
        }
      : {
          available: true,
          entries: [
            { name: 'match-catalog-plugin', description: 'Matching catalog', repo: 'https://example.test/repo' }
          ]
        }
  )
  hub.mockResolvedValue({
    results: [{ identifier: 'official/match-skill', name: 'match-marketplace', description: 'Matching marketplace' }]
  })
  const onOpen = vi.fn()
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <CapabilitySearchResults
        onOpen={onOpen}
        profile={{ connectionId: 'remote-fixture', profile: 'worker' }}
        query="match"
        skills={[{ name: 'match-skill', description: 'Matching skill' } as never]}
        skillsFailed={false}
        toolsets={[
          { name: 'web', label: 'match-toolset', description: 'Matching tools', tools: ['match-tool'] } as never
        ]}
        toolsetsFailed={false}
      />
    </QueryClientProvider>
  )
  expect(await screen.findByRole('button', { name: /match-marketplace/ })).toBeTruthy()

  for (const name of ['Skills', 'Tools', 'Plugins', 'Connectors', 'Skill Marketplace']) {
    expect(within(screen.getByRole('region', { name })).getAllByRole('button').length).toBeGreaterThan(0)
  }

  fireEvent.click(screen.getByRole('button', { name: 'match-plugin' }))
  expect(onOpen).toHaveBeenCalledWith(expect.objectContaining({ tab: 'plugins', id: 'match-plugin' }))
  expect(rpc).toHaveBeenCalledWith('remote-fixture', 'worker', 'plugins.manage', { action: 'list', profile: 'worker' })
  expect(hub).toHaveBeenCalledWith('match', 'all', 100, { connectionId: 'remote-fixture', profile: 'worker' })
})
it('keeps installed matches available when discovery fails and discards late results from the old scope', async () => {
  let resolveOld!: (value: unknown) => void
  rpc.mockImplementation(async (_connection, profile, method) =>
    method === 'plugins.manage'
      ? { plugins: [] }
      : profile === 'old'
        ? await new Promise(resolve => {
            resolveOld = resolve
          })
        : { available: true, entries: [] }
  )
  hub.mockRejectedValue(new Error('offline'))
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const view = (profile: string) => (
    <QueryClientProvider client={client}>
      <CapabilitySearchResults
        key={profile}
        onOpen={vi.fn()}
        profile={{ connectionId: 'remote-fixture', profile }}
        query="match"
        skills={[{ name: 'match-skill', description: 'Installed' } as never]}
        skillsFailed={false}
        toolsets={[]}
        toolsetsFailed={false}
      />
    </QueryClientProvider>
  )

  const rendered = render(view('old'))
  await waitFor(() => expect(resolveOld).toBeTruthy())
  rendered.rerender(view('new'))
  resolveOld({
    available: true,
    entries: [{ name: 'match-old-secret', description: 'Old scope', repo: 'https://example.test/old' }]
  })
  expect(await screen.findByRole('button', { name: 'match-skill' })).toBeTruthy()
  expect(await screen.findByRole('alert')).toBeTruthy()
  expect(screen.queryByText('match-old-secret')).toBeNull()
})
