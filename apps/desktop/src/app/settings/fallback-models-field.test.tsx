import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

// Radix Select calls scrollIntoView / pointer-capture APIs jsdom lacks.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const getGlobalModelOptions = vi.fn()
vi.mock('@/store/session', async importOriginal => ({ ...(await importOriginal<Record<string, unknown>>()), $connection: (await import('nanostores')).atom({ mode: 'local', baseUrl: 'http://fixture', connectionId: 'test-local' }) }))
vi.mock('@/lib/model-options', async importOriginal => ({ ...(await importOriginal<Record<string, unknown>>()), requestDiscoveredModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args) }))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getGlobalModelOptions: () => getGlobalModelOptions()
}))

// Load once at module scope so no test's 15s budget pays the heavy transform
// + import (the first-test timeout flake under CI load).
const { FallbackModelsField } = await import('./fallback-models-field')

beforeEach(() => {
  getGlobalModelOptions.mockResolvedValue({
    providers: [
      { name: 'GitHub Copilot', slug: 'copilot', models: ['gpt-5-mini', 'gpt-5.4-mini'], eligible_models: ['gpt-5-mini', 'gpt-5.4-mini'] },
      { name: 'OpenAI Codex', slug: 'openai-codex', models: ['gpt-5.4-mini'], eligible_models: ['gpt-5.4-mini'] },
      { name: 'Nous', slug: 'nous', models: ['hermes-4'], eligible_models: ['hermes-4'] }
    ]
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderField(value: unknown, onChange = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <FallbackModelsField onChange={onChange} value={value} />
    </QueryClientProvider>
  )

  return onChange
}

function renderFieldWithRerender(value: unknown, onChange = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const view = render(
    <QueryClientProvider client={client}>
      <FallbackModelsField onChange={onChange} value={value} />
    </QueryClientProvider>
  )

  return (next: unknown) =>
    view.rerender(
      <QueryClientProvider client={client}>
        <FallbackModelsField onChange={onChange} value={next} />
      </QueryClientProvider>
    )
}

const CHAIN = [
  { provider: 'copilot', model: 'gpt-5-mini' },
  { provider: 'openai-codex', model: 'gpt-5.4-mini' }
]

describe('FallbackModelsField', () => {
  it('keeps reasoning and removal as drafts until one chain Apply, and Cancel restores the saved chain', async () => {
    const onChange = renderField(CHAIN)
    fireEvent.click(screen.getAllByLabelText('Remove')[1])
    expect(onChange).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.getAllByLabelText('Remove')).toHaveLength(2)
    fireEvent.click(screen.getAllByRole('combobox', { name: /reasoning/i })[0])
    fireEvent.click(await screen.findByRole('option', { name: 'High' }))
    expect(onChange).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    await waitFor(() => expect(onChange).toHaveBeenCalledWith([{ ...CHAIN[0], reasoning_effort: 'high' }, CHAIN[1]]))
  })

  it('does not delete a saved fallback during an incomplete provider replacement', async () => {
    const routed = { ...CHAIN[0], base_url: 'http://fixture/v1', key_env: 'FIXTURE_KEY', api_key: '${FIXTURE_KEY}', api_mode: 'responses' }
    const onChange = renderField([routed, CHAIN[1]])
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())
    fireEvent.click(screen.getAllByRole('combobox')[0])
    fireEvent.click(await screen.findByRole('option', { name: 'Nous' }))
    expect(onChange.mock.calls.every(([chain]) => JSON.stringify(chain) === JSON.stringify([routed, CHAIN[1]]))).toBe(true)
  })

  it('renders each {provider, model} entry as its own row (never "[object Object]")', async () => {
    await renderField(CHAIN)

    // One Remove control per entry proves the object list became rows — the old
    // generic `list` input stringified the array to "[object Object]".
    expect(screen.getAllByLabelText('Remove')).toHaveLength(2)
    expect(screen.getByText('Add fallback')).toBeTruthy()
    expect(screen.queryByText(/\[object Object\]/)).toBeNull()
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())
  })

  it('removing a row emits the remaining entries with their routing keys intact', async () => {
    // #89184: a hand-written local-gateway chain carries base_url/api_key per
    // entry; the editor only owns provider/model and must not strip the rest.
    const routed = { provider: 'custom', model: 'glm-5.08', base_url: 'http://gw:8080/v1', api_key: '${GW_KEY}' }
    const onChange = await renderField([...CHAIN, routed])

    fireEvent.click(screen.getAllByLabelText('Remove')[0])

    expect(onChange).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    expect(onChange.mock.calls.at(-1)?.[0]).toEqual([{ provider: 'openai-codex', model: 'gpt-5.4-mini' }, routed])
  })

  it('adding a blank row does not persist a partial entry', async () => {
    const onChange = await renderField(CHAIN)

    fireEvent.click(screen.getByText('Add fallback'))

    // The new empty row stays in the UI but only complete pairs are emitted.
    expect(onChange).not.toHaveBeenCalled()
    expect(screen.getAllByLabelText('Remove')).toHaveLength(3)
  })

  it('resyncs rows when persisted config changes', async () => {
    const rerender = await renderFieldWithRerender(CHAIN)
    expect(screen.getAllByLabelText('Remove')).toHaveLength(2)

    rerender([{ provider: 'nous', model: 'hermes-4' }])

    await waitFor(() => expect(screen.getAllByLabelText('Remove')).toHaveLength(1))
  })

  it('keeps a draft row visible after autosave re-renders the same persisted chain', async () => {
    const onChange = vi.fn()
    const rerender = await renderFieldWithRerender([], onChange)

    fireEvent.click(screen.getByText('Add fallback'))

    expect(onChange).not.toHaveBeenCalled()
    expect(screen.getAllByLabelText('Remove')).toHaveLength(1)

    // Parent autosave echo — same complete chain, new array identity.
    rerender([])

    await waitFor(() => expect(screen.getAllByLabelText('Remove')).toHaveLength(1))
  })
})

describe('fallback atomic searchable edits', () => {
  const routed = { ...CHAIN[0], base_url: 'http://fixture/v1', key_env: 'FIXTURE_KEY', api_key: '${FIXTURE_KEY}', api_mode: 'responses' }

  it('searches names locally, cancels replacement, and retains same-provider route metadata', async () => {
    const onChange = renderField([routed, CHAIN[1]])
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())
    const calls = getGlobalModelOptions.mock.calls.length
    fireEvent.click(screen.getAllByRole('combobox')[0])
    await screen.findByRole('option', { name: 'Nous' })
    fireEvent.change(screen.getByPlaceholderText('Provider'), { target: { value: 'nOuS' } })
    expect(screen.queryByRole('option', { name: 'GitHub Copilot' })).toBeNull()
    fireEvent.click(await screen.findByRole('option', { name: 'Nous' }))
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onChange).not.toHaveBeenCalled()
    fireEvent.click(screen.getAllByRole('combobox')[1])
    fireEvent.change(screen.getByPlaceholderText('Model'), { target: { value: '5.4' } })
    fireEvent.click(screen.getByRole('option', { name: 'gpt-5.4-mini' }))
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    expect(onChange).toHaveBeenLastCalledWith([{ ...routed, model: 'gpt-5.4-mini' }, CHAIN[1]])
    expect(getGlobalModelOptions).toHaveBeenCalledTimes(calls)
  })

  it('commits a replacement in order without previous provider routing metadata', async () => {
    const onChange = renderField([routed, CHAIN[1]])
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())
    fireEvent.click(screen.getAllByRole('combobox')[0])
    fireEvent.click(await screen.findByRole('option', { name: 'Nous' }))
    fireEvent.click(screen.getAllByRole('combobox')[1])
    fireEvent.click(screen.getByRole('option', { name: 'hermes-4' }))
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    expect(onChange).toHaveBeenLastCalledWith([{ provider: 'nous', model: 'hermes-4' }, CHAIN[1]])
  })

  it('removes another row without losing an incomplete replacement saved route', async () => {
    const onChange = renderField([routed, CHAIN[1]])
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())
    fireEvent.click(screen.getAllByRole('combobox')[0])
    fireEvent.click(await screen.findByRole('option', { name: 'Nous' }))
    fireEvent.click(screen.getAllByLabelText('Remove')[1])
    expect(onChange).not.toHaveBeenCalled()
    expect((screen.getByRole('button', { name: 'Apply' }) as HTMLButtonElement).disabled).toBe(true)
  })
})


describe('fallback discovery ownership', () => {
  it('drops a delayed old-source catalog and scopes discovery to the active settings profile', async () => {
    const { $connection } = await import('@/store/session')
    const { $settingsScopeOverride } = await import('@/store/settings-scope')
    let finish!: (value: unknown) => void
    getGlobalModelOptions.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    const onChange = renderField(CHAIN)
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())
    await act(async () => {
      $settingsScopeOverride.set('research')
      $connection.set({ mode: 'remote', baseUrl: 'http://source-b', connectionId: 'source-b' } as never)
    })
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenLastCalledWith(expect.objectContaining({ profile: 'research', ownerConnectionId: 'source-b' })))
    await act(async () => { finish({ providers: [{ name: 'Old Private Provider', slug: 'old', models: ['private'] }] }) })
    fireEvent.click(screen.getAllByRole('combobox')[0])
    expect(screen.queryByRole('option', { name: 'Old Private Provider' })).toBeNull()
    expect(onChange).not.toHaveBeenCalled()
    await act(async () => {
      $settingsScopeOverride.set(null)
      $connection.set({ mode: 'local', baseUrl: 'http://fixture', connectionId: 'test-local' } as never)
    })
  })
})

describe('fallback manual models', () => {
  it('allows an explicit unverified ID with IME-safe Enter and never renders route credentials', async () => {
    getGlobalModelOptions.mockResolvedValue({ providers: [{ name: 'Nous', slug: 'nous', authenticated: true, models: [], discovery_state: 'missing' }] })
    const onChange = renderField([{ provider: 'nous', model: 'saved-id', api_key: 'synthetic-private-key' }])
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())
    await waitFor(() => expect((screen.getByRole('combobox', { name: 'Model' }) as HTMLButtonElement).disabled).toBe(false))
    fireEvent.click(screen.getByRole('combobox', { name: 'Model' }))
    fireEvent.click(await screen.findByRole('option', { name: /Custom model/ }))
    const input = screen.getByPlaceholderText('Model id')
    fireEvent.change(input, { target: { value: 'manual/new-id' } })
    fireEvent.compositionStart(input)
    fireEvent.keyDown(input, { key: 'Enter', isComposing: true, keyCode: 229 })
    expect(screen.getByPlaceholderText('Model id')).toBeTruthy()
    fireEvent.compositionEnd(input)
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(onChange).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    expect(onChange).toHaveBeenLastCalledWith([{ provider: 'nous', model: 'manual/new-id', api_key: 'synthetic-private-key' }])
    expect(screen.queryByText('synthetic-private-key')).toBeNull()
  })
})


describe('chain Apply recovery', () => {
  it('retains a failed draft for retry and omits an inherited reasoning override', async () => {
    const apply = vi.fn().mockRejectedValueOnce(new Error('synthetic persistence failure')).mockResolvedValue(undefined)
    renderField([{ ...CHAIN[0], reasoning_effort: 'none', api_mode: 'responses' }], apply)
    fireEvent.click(screen.getByRole('combobox', { name: /Reasoning/ }))
    fireEvent.click(await screen.findByRole('option', { name: /inherit/i }))
    expect(apply).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    await screen.findByRole('alert')
    expect(apply).toHaveBeenCalledWith([{ ...CHAIN[0], api_mode: 'responses' }])
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
    expect(apply).toHaveBeenCalledTimes(2)
  })
})
