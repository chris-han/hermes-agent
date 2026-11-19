import type { ModelOptionsResult } from '@hermes/shared'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ConfirmHost } from '@/components/confirm-host'
import { I18nProvider } from '@/i18n'
import { $confirmRequest } from '@/store/confirm'
import { $customModels } from '@/store/custom-models'
import { $refreshingModelSources, modelDiscoveryRefreshKey } from '@/store/model-discovery'
import { $knownModels, $visibleModels, modelVisibilityKey, setVisibleModels } from '@/store/model-visibility'
import { stubResizeObserver } from '@/test/jsdom'

import { ModelVisibilityDialog } from './model-visibility-dialog'

vi.mock('@/lib/model-options', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestDiscoveredModelOptions: vi.fn()
}))

import { requestDiscoveredModelOptions as requestModelOptions } from '@/lib/model-options'

stubResizeObserver()

const OPTIONS: ModelOptionsResult = {
  providers: [
    { authenticated: true, models: ['gpt-5.5', 'gpt-6'], name: 'OpenAI Codex', slug: 'openai-codex' },
    { authenticated: true, models: ['qwen3-coder'], name: 'Qwen', slug: 'qwen' }
  ]
}

function renderDialog() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <I18nProvider>
        <ModelVisibilityDialog onOpenChange={() => {}} onOpenProviders={() => {}} open ownerConnectionId="test" />
        <ConfirmHost />
      </I18nProvider>
    </QueryClientProvider>
  )
}

const isShown = async (label: RegExp) =>
  (await screen.findByRole('switch', { name: label })).getAttribute('aria-checked') === 'true'

beforeEach(() => {
  window.localStorage.clear()
  $customModels.set([])
  $visibleModels.set(null)
  $knownModels.set(null)
  $refreshingModelSources.set(new Set())
  vi.mocked(requestModelOptions).mockResolvedValue(OPTIONS)
})

afterEach(() => {
  cleanup()
  $confirmRequest.set(null)
  vi.clearAllMocks()
})

describe('Edit Models reset', () => {
  it('shows nonblocking startup refresh progress for this source', async () => {
    $refreshingModelSources.set(new Set([modelDiscoveryRefreshKey('test', 'default')]))
    renderDialog()
    expect(await screen.findByText(/refreshing provider models/i)).toBeTruthy()
    expect(await isShown(/gpt-6/i)).toBe(true)
  })

  it('labels a retained verified list when refresh failed', async () => {
    vi.mocked(requestModelOptions).mockResolvedValue({
      providers: [
        { name: 'Lab', slug: 'lab', models: ['live-model'], discovery_state: 'stale', discovery_source_id: 'opaque' }
      ],
      discovery_contract: 'provider-query-v1'
    })
    renderDialog()
    expect(await screen.findByText(/last verified/i)).toBeTruthy()
    expect(await isShown(/live model/i)).toBe(true)
  })

  it('restores the default shortlist when a model is stuck hidden by the known snapshot', async () => {
    // A curation saved while the snapshot machinery was live: gpt-6 was listed
    // when the user kept gpt-5.5 and switched it off, so the snapshot counts it
    // as judged and the default rule never re-admits it — the dialog shows it
    // switched off with no per-model way back that survives a catalog change.
    setVisibleModels(new Set([modelVisibilityKey('openai-codex', 'gpt-5.5')]), OPTIONS.providers!)

    renderDialog()

    expect(await isShown(/gpt-6/i)).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: /reset to defaults/i }))
    fireEvent.click(await screen.findByRole('button', { name: /^reset$/i }))

    await waitFor(async () => expect(await isShown(/gpt-6/i)).toBe(true))
    expect(await isShown(/gpt-5\.5/i)).toBe(true)
    expect(window.localStorage.getItem('hermes.desktop.visible-models')).toBeNull()
    expect(window.localStorage.getItem('hermes.desktop.known-models')).toBeNull()
  })

  it('keeps the user’s choices when the reset is cancelled', async () => {
    setVisibleModels(new Set([modelVisibilityKey('openai-codex', 'gpt-5.5')]), OPTIONS.providers!)

    renderDialog()
    await isShown(/gpt-6/i)

    fireEvent.click(screen.getByRole('button', { name: /reset to defaults/i }))
    fireEvent.click(await screen.findByRole('button', { name: /cancel/i }))

    await waitFor(() => expect($confirmRequest.get()).toBeNull())
    expect(await isShown(/gpt-6/i)).toBe(false)
  })

  it('retains a saved custom model without inserting it into provider discovery', async () => {
    // An aggregator row defaults to its featured shortlist; the typed id is
    // appended after the catalog, so the bare default rule would hide it.
    vi.mocked(requestModelOptions).mockResolvedValue({
      providers: [
        {
          authenticated: true,
          featured_models: ['openai/gpt-6'],
          models: ['openai/gpt-5', 'openai/gpt-6', 'anthropic/claude-x'],
          name: 'OpenRouter',
          slug: 'openrouter'
        }
      ]
    })
    $customModels.set([{ model: 'acme/model-x', provider: 'openrouter' }])
    $visibleModels.set(new Set([modelVisibilityKey('openrouter', 'acme/model-x')]))

    renderDialog()

    expect(await isShown(/gpt-6/i)).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: /reset to defaults/i }))
    fireEvent.click(await screen.findByRole('button', { name: /^reset$/i }))

    await waitFor(async () => expect(await isShown(/gpt-6/i)).toBe(true))
    expect(screen.queryByText('Model X')).toBeNull()
    expect($customModels.get()).toEqual([{ model: 'acme/model-x', provider: 'openrouter' }])
  })

  it('offers no reset while the list is still the default', async () => {
    renderDialog()

    expect(await isShown(/gpt-6/i)).toBe(true)
    expect(screen.queryByRole('button', { name: /reset to defaults/i })).toBeNull()
  })

  it('does not offer reset before the provider catalog has loaded', () => {
    $visibleModels.set(new Set([modelVisibilityKey('openai-codex', 'gpt-5.5')]))
    vi.mocked(requestModelOptions).mockReturnValue(new Promise(() => {}))

    renderDialog()

    expect(screen.queryByRole('button', { name: /reset to defaults/i })).toBeNull()
  })
})
