import type { ModelOptionsResult } from '@hermes/shared'
import { fuzzyRank, modelSearchText } from '@hermes/shared'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { ModelPickerDialog } from './model-picker'

vi.mock('@/lib/model-options', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestModelOptions: vi.fn()
}))

import { requestModelOptions } from '@/lib/model-options'

stubResizeObserver()
stubMenuDomApis()

const OPTIONS: ModelOptionsResult = {
  model: 'Qwen3.6-27B-UD-Q4_K_XL',
  provider: 'llamacpp',
  providers: [
    {
      slug: 'llamacpp',
      name: 'Local',
      models: ['Qwen3.6-27B-UD-Q4_K_XL'],
      is_current: true,
      authenticated: true
    },
    {
      slug: 'nous',
      name: 'Nous',
      models: ['Hermes-4.5'],
      authenticated: true
    }
  ]
}

// One client per test, created in beforeEach: the tests seed jobs BEFORE
// rendering, so the picker must mount against the client that was seeded.
let client: QueryClient = new QueryClient()

function renderPicker(ui?: Partial<Parameters<typeof ModelPickerDialog>[0]>) {
  const element: ReactElement = (
    <QueryClientProvider client={client}>
      <I18nProvider>
        <ModelPickerDialog
          currentModel="Qwen3.6-27B-UD-Q4_K_XL"
          currentProvider="llamacpp"
          onOpenChange={() => undefined}
          onSelect={() => undefined}
          open
          {...ui}
        />
      </I18nProvider>
    </QueryClientProvider>
  )

  return render(element)
}

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  vi.mocked(requestModelOptions).mockResolvedValue(OPTIONS)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('ModelPickerDialog search ranking', () => {
  // Rows must come out in the order the shared fuzzyRank produces — the same
  // helper the web and TUI pickers use — so a query ranks identically on
  // every surface. Curated order puts the scattered match first; the ranked
  // order does not, which is what proves the picker is not substring-filtering.
  const MODELS = ['glm-4.6-omni', 'claude-sonnet-4', 'gpt-4o']

  it('orders model rows exactly as the shared fuzzyRank does', async () => {
    vi.mocked(requestModelOptions).mockResolvedValue({
      providers: [{ slug: 'nous', name: 'Nous', models: MODELS, authenticated: true }]
    })
    renderPicker({ currentModel: 'gpt-4o', currentProvider: 'nous' })
    await screen.findByText('gpt-4o')

    const query = 'g4o'
    fireEvent.change(screen.getByRole('combobox'), { target: { value: query } })

    const expected = fuzzyRank(MODELS, query, modelSearchText).map(r => r.item)

    expect(expected).not.toEqual(MODELS.filter(m => expected.includes(m)))
    await waitFor(() => {
      const rows = screen.getAllByRole('option').map(el => el.textContent?.trim())

      expect(rows).toEqual(expected)
    })
  })

  // Regression guard: main folded `[-_.]` on both sides (foldIncludes); the
  // shared ranker must too, or a query typed with the "wrong" separator
  // drops every row while the highlighter (which still folds) disagrees.
  it.each([
    ['gpt.4o', 'gpt-4o'],
    ['claude_3', 'claude-3-opus'],
    ['qwen3-8', 'qwen3.8-flash']
  ])('separator variant %s still lists %s', async (query, expected) => {
    const catalog = ['gpt-4o', 'claude-3-opus', 'qwen3.8-flash']

    vi.mocked(requestModelOptions).mockResolvedValue({
      providers: [{ slug: 'nous', name: 'Nous', models: catalog, authenticated: true }]
    })
    renderPicker({ currentModel: 'gpt-4o', currentProvider: 'nous' })
    await screen.findByText('gpt-4o')

    fireEvent.change(screen.getByRole('combobox'), { target: { value: query } })

    await waitFor(() => {
      const rows = screen.getAllByRole('option').map(el => el.textContent?.trim())

      expect(rows).toContain(expected)
    })
  })
})
