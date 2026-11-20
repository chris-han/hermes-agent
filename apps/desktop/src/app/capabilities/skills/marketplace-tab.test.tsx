import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { readMarketplaceSource, saveMarketplaceSource } from './marketplace-source'
import { MarketplaceTab } from './marketplace-tab'
const { search, sources, preview } = vi.hoisted(() => ({ search: vi.fn(), sources: vi.fn(), preview: vi.fn() }))
vi.mock('@/hermes', async original => ({
  ...(await original<object>()),
  searchSkillsHub: search,
  getSkillHubSources: sources,
  previewSkillHub: preview
}))
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  localStorage.clear()
})

function show(profile = 'fixture') {
  sources.mockResolvedValue({ sources: [], installed: {}, featured: [] })
  search.mockResolvedValue({ results: [], installed: {}, timed_out: [] })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <MarketplaceTab profile={profile} query="" />
    </QueryClientProvider>
  )
}

it('defaults to exact Semantier empty listing without persisting the default', async () => {
  show()
  await waitFor(() => expect(search).toHaveBeenCalledWith('', 'semantier', 100, 'fixture'))
  expect(localStorage.getItem('hermes.desktop.marketplaceSource.v1:fixture')).toBeNull()
  expect(screen.getByText('Semantier Official')).toBeTruthy()
})
it.each(['all', 'official', 'hermes-index', 'browse-sh', 'url', 'semantier', 'missing-provider'])(
  'preserves saved %s across remount and isolates profiles',
  async source => {
    saveMarketplaceSource('fixture', source)
    const first = show()

    if (source !== 'url') {
      await waitFor(() => expect(search).toHaveBeenCalledWith('', source, 100, 'fixture'))
    } else {
      expect(screen.getByRole('textbox', { name: 'SKILL.md URL' })).toBeTruthy()
    }

    first.unmount()
    show()
    expect(readMarketplaceSource('fixture')).toBe(source)
    expect(readMarketplaceSource('other')).toBe('semantier')
  }
)
it('previews Direct URL only after explicit submission', async () => {
  saveMarketplaceSource('fixture', 'url')
  preview.mockResolvedValue({ skill_md: 'Fixture content', identifier: 'https://example.org/SKILL.md' })
  show()
  fireEvent.change(screen.getByRole('textbox', { name: 'SKILL.md URL' }), {
    target: { value: 'https://example.org/SKILL.md' }
  })
  expect(preview).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Preview' }))
  await waitFor(() => expect(preview).toHaveBeenCalledWith('https://example.org/SKILL.md', 'fixture'))
  expect(await screen.findByText('Fixture content')).toBeTruthy()
})

it('keeps a saved source through registry failure and does not promote spoofed third-party authorship', async () => {
  saveMarketplaceSource('fixture', 'all')
  search.mockResolvedValueOnce({
    results: [
      {
        identifier: 'url/demo',
        name: 'Semantier looking author',
        description: 'Community',
        source: 'url',
        publisher_verified: true
      }
    ],
    installed: {},
    timed_out: [],
    errors: [{ source: 'semantier', message: 'Catalog rejected', code: 'invalid_catalog' }]
  })
  const view = show()
  await screen.findByText('Catalog rejected', { exact: false })
  expect(readMarketplaceSource('fixture')).toBe('all')
  expect(screen.queryByText('Publisher verified: Semantier')).toBeNull()
  view.unmount()
})

it('retains explicit all when source loading and browser storage fail', async () => {
  saveMarketplaceSource('fixture', 'all')
  sources.mockRejectedValueOnce(new Error('source list offline'))
  const original = Storage.prototype.setItem

  const failing = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
    throw new Error('quota')
  })

  try {
    show()
    await waitFor(() => expect(search).toHaveBeenCalledWith('', 'all', 100, 'fixture'))
    expect(readMarketplaceSource('fixture')).toBe('all')
    expect(() => saveMarketplaceSource('fixture', 'official')).not.toThrow()
  } finally {
    failing.mockRestore()
    Storage.prototype.setItem = original
  }
})

it('discards late old-profile query results after a scope change', async () => {
  let resolveOld!: (value: unknown) => void
  search.mockImplementationOnce(
    () =>
      new Promise(resolve => {
        resolveOld = resolve
      })
  )
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  sources.mockResolvedValue({ sources: [], installed: {}, featured: [] })
  search.mockResolvedValue({ results: [], installed: {}, timed_out: [] })

  const tree = (profile: string) => (
    <QueryClientProvider client={client}>
      <MarketplaceTab profile={profile} query="" />
    </QueryClientProvider>
  )

  const view = render(tree('old-profile'))
  await waitFor(() => expect(resolveOld).toBeTypeOf('function'))
  view.rerender(tree('new-profile'))
  resolveOld({
    results: [{ identifier: 'official/old', source: 'official', name: 'Old profile secret', description: '' }],
    installed: {},
    timed_out: []
  })
  await waitFor(() => expect(search).toHaveBeenCalledWith('', 'semantier', 100, 'new-profile'))
  expect(screen.queryByText('Old profile secret')).toBeNull()
})

it('never displays publisher authentication for Semantier, including historical verification metadata', async () => {
  search.mockResolvedValueOnce({
    results: [
      { identifier: 'semantier/semantier/demo', name: 'Unsigned demo', source: 'semantier', publisher_verified: true }
    ],
    installed: {},
    timed_out: []
  })
  show()
  await screen.findByRole('heading', { name: 'Unsigned demo' })
  expect(screen.queryByText('Publisher verified: Semantier')).toBeNull()
  expect(screen.getByText('Community source · publisher unverified')).toBeTruthy()
})
