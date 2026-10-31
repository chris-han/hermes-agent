// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type I18nContextValue, I18nProvider, useI18n } from '@/i18n'
import type { CustomEndpointsResponse } from '@/types/hermes'

const getCustomEndpoints = vi.fn()
const saveCustomEndpoint = vi.fn()
const validateCustomEndpoint = vi.fn()
const notify = vi.fn()
const notifyError = vi.fn()
const triggerHaptic = vi.fn()

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: atom('default'),
  $profiles: atom([]),
  refreshProfiles: async () => {},
  normalizeProfileKey: (p: string | null) => p || 'default',
  profileLabel: (p: { display_name?: string; name: string }) => p.display_name || p.name
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  activateCustomEndpoint: vi.fn(),
  deleteCustomEndpoint: vi.fn(),
  getCustomEndpoints: (...args: unknown[]) => getCustomEndpoints(...args),
  getProfiles: async () => ({ profiles: (await import('@/store/profile')).$profiles.get() }),
  saveCustomEndpoint: (...args: unknown[]) => saveCustomEndpoint(...args),
  setApiRequestProfile: vi.fn(),
  validateCustomEndpoint: (...args: unknown[]) => validateCustomEndpoint(...args)
}))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: (...args: unknown[]) => triggerHaptic(...args) }))
vi.mock('@/store/notifications', () => ({
  notify: (...args: unknown[]) => notify(...args),
  notifyError: (...args: unknown[]) => notifyError(...args)
}))

// Load once at module scope so no test's 15s budget pays the heavy transform
// + import (the first-test timeout flake under CI load).
const { CustomEndpointsSettings } = await import('./custom-endpoints-settings')
const { $activeGatewayProfile, $profiles } = await import('@/store/profile')
const { $settingsScopeOverride } = await import('@/store/settings-scope')

const emptyResponse: CustomEndpointsResponse = {
  current: { base_url: '', model: '', provider: '' },
  endpoints: []
}

const savedResponse: CustomEndpointsResponse = {
  current: { base_url: 'http://profile-a.test/v1', model: 'model-a', provider: 'profile-a-endpoint' },
  endpoints: [
    {
      base_url: 'http://profile-a.test/v1',
      discover_models: true,
      has_api_key: false,
      id: 'profile-a-endpoint',
      is_current: true,
      model: 'model-a',
      models: ['model-a'],
      name: 'Profile A'
    }
  ],
  id: 'profile-a-endpoint',
  ok: true
}

beforeEach(() => {
  $activeGatewayProfile.set('default')
  $settingsScopeOverride.set(null)
  $profiles.set([])
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  $settingsScopeOverride.set(null)
})

describe('CustomEndpointsSettings', () => {
  it('shows explicit success beside Test and announces it again on a repeated test', async () => {
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    validateCustomEndpoint.mockResolvedValueOnce({ ok: true, reachable: true, models: ['example-model'], message: '' })
    let finish!: (value: unknown) => void
    validateCustomEndpoint.mockImplementationOnce(
      () =>
        new Promise(resolve => {
          finish = resolve
        })
    )
    render(<CustomEndpointsSettings />)
    const url = await screen.findByPlaceholderText('http://localhost:4000/v1')
    fireEvent.change(url, { target: { value: 'http://localhost:4000/v1' } })
    const test = screen.getByRole('button', { name: 'Test' })
    fireEvent.click(test)
    const result = await screen.findByRole('status')
    expect(result.textContent).toContain('Endpoint is reachable.')
    expect(result.textContent).toContain('Found 1 models.')
    expect(result.textContent).toContain('No inference request is sent')
    expect(test.parentElement?.nextElementSibling).toBe(result)
    expect(result.getAttribute('aria-live')).toBe('polite')
    fireEvent.click(test)
    expect(screen.queryByRole('status')).toBeNull()
    expect(test.hasAttribute('disabled')).toBe(true)
    await act(async () => finish({ ok: true, reachable: true, models: [], message: '' }))
    expect((await screen.findByRole('status')).textContent).toContain('Endpoint is reachable.')
    expect(validateCustomEndpoint).toHaveBeenCalledTimes(2)
    expect(saveCustomEndpoint).not.toHaveBeenCalled()
  })

  it('orders URL, key and Test before model setup without sending an automatic request', async () => {
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    validateCustomEndpoint.mockResolvedValue({ ok: true, reachable: true, models: [], message: '' })
    render(<CustomEndpointsSettings />)
    const url = await screen.findByPlaceholderText<HTMLInputElement>('http://localhost:4000/v1')
    const key = screen.getByLabelText<HTMLInputElement>('API Key')
    const test = screen.getByRole('button', { name: 'Test' })
    const mode = screen.getByRole('group', { name: 'API Mode' })
    expect(url.compareDocumentPosition(key) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(key.compareDocumentPosition(mode) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(key.closest('label')?.parentElement?.contains(test)).toBe(true)
    expect(url.value).toBe('')
    expect(validateCustomEndpoint).not.toHaveBeenCalled()
    fireEvent.change(url, { target: { value: 'http://localhost:4000/v1' } })
    fireEvent.change(key, { target: { value: 'test-key-only' } })
    fireEvent.click(test)
    await waitFor(() =>
      expect(validateCustomEndpoint).toHaveBeenCalledWith(
        expect.objectContaining({ base_url: 'http://localhost:4000/v1', api_key: 'test-key-only' }),
        'default'
      )
    )
  })

  it('preserves an unchanged masked key and manual model after catalog failure; Cancel drops a new secret', async () => {
    const retained = { ...savedResponse, endpoints: [{ ...savedResponse.endpoints[0], has_api_key: true }] }
    getCustomEndpoints.mockResolvedValue(retained)
    saveCustomEndpoint.mockResolvedValue(retained)
    validateCustomEndpoint.mockResolvedValue({ ok: false, reachable: true, models: [], message: 'Catalog unsupported' })
    render(<CustomEndpointsSettings />)
    await screen.findByText('Profile A')
    const key = screen.getByLabelText<HTMLInputElement>('API Key')
    expect(key.value).toBe('')
    fireEvent.click(screen.getByRole('button', { name: 'Test' }))
    await screen.findByText('Catalog unsupported')
    expect(screen.getByPlaceholderText<HTMLInputElement>('gpt-5.4').value).toBe('model-a')
    fireEvent.click(screen.getByRole('button', { name: 'Save and use' }))
    await waitFor(() =>
      expect(saveCustomEndpoint).toHaveBeenCalledWith(
        expect.objectContaining({
          api_key: undefined,
          model: 'model-a',
          base_url: 'http://profile-a.test/v1'
        }),
        'default'
      )
    )
    fireEvent.change(key, { target: { value: 'unsaved-private-key' } })
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(key.value).toBe('')
    expect(saveCustomEndpoint).toHaveBeenCalledTimes(1)
    expect(screen.queryByText('unsaved-private-key')).toBeNull()
  })

  it('localizes endpoint editing on language changes without changing transport or draft identifiers', async () => {
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    saveCustomEndpoint.mockResolvedValue(savedResponse)
    let language!: I18nContextValue

    function Surface() {
      language = useI18n()

      return <CustomEndpointsSettings />
    }

    render(
      <I18nProvider configClient={null} initialLocale="zh">
        <Surface />
      </I18nProvider>
    )
    await screen.findByText('暂无自定义端点')
    fireEvent.change(screen.getByRole('textbox', { name: '名称' }), { target: { value: 'Fixture Ω' } })
    fireEvent.change(screen.getByRole('textbox', { name: '端点 URL' }), { target: { value: 'http://fixture.test/v1' } })
    fireEvent.change(screen.getByRole('combobox', { name: '默认模型' }), { target: { value: 'fixture-model' } })
    fireEvent.click(screen.getByRole('button', { name: 'Responses API' }))
    await act(() => language.setLocale('zh-hant'))
    expect((screen.getByRole('textbox', { name: '名稱' }) as HTMLInputElement).value).toBe('Fixture Ω')
    expect(screen.getByText('API 模式')).toBeTruthy()
    expect(screen.getByRole('button', { name: '自動偵測' })).toBeTruthy()
    expect(saveCustomEndpoint).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('checkbox', { name: '此端點不需要 API 金鑰' }))
    fireEvent.click(screen.getByRole('button', { name: '儲存並使用' }))
    expect(saveCustomEndpoint).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'Fixture Ω',
        api_mode: 'codex_responses',
        base_url: 'http://fixture.test/v1',
        model: 'fixture-model'
      }),
      'default'
    )
  })

  it('sends the chosen API mode and discovered alias metadata on Save (#93622)', async () => {
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    validateCustomEndpoint.mockResolvedValue({
      message: '',
      model_details: [
        { id: 'gpt-5.6-sol' },
        { canonical_model: 'gpt-5.6-sol', id: 'gpt-5.6-sol-high', reasoning_effort: 'high' }
      ],
      models: ['gpt-5.6-sol', 'gpt-5.6-sol-high'],
      ok: true,
      reachable: true,
      transport_checked: 'codex_responses'
    })
    saveCustomEndpoint.mockResolvedValue(savedResponse)

    render(<CustomEndpointsSettings />)

    await screen.findByText('No custom endpoints')
    fireEvent.change(screen.getByPlaceholderText('Axet Proxy'), { target: { value: 'Responses gateway' } })
    fireEvent.change(screen.getByPlaceholderText('http://localhost:4000/v1'), {
      target: { value: 'https://responses-gateway.example.com/v1' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Responses API' }))
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Test' }))
    })
    fireEvent.change(screen.getByPlaceholderText('gpt-5.4'), { target: { value: 'gpt-5.6-sol-high' } })
    fireEvent.click(screen.getByRole('checkbox', { name: 'This endpoint requires no API key' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save and use' }))

    expect(validateCustomEndpoint).toHaveBeenCalledWith(
      expect.objectContaining({ api_mode: 'codex_responses' }),
      'default'
    )
    expect(screen.getByRole('status').textContent).toContain('No inference request is sent')
    expect(saveCustomEndpoint).toHaveBeenCalledWith(
      expect.objectContaining({
        api_mode: 'codex_responses',
        model: 'gpt-5.6-sol-high',
        model_details: expect.arrayContaining([
          expect.objectContaining({ canonical_model: 'gpt-5.6-sol', id: 'gpt-5.6-sol-high', reasoning_effort: 'high' })
        ]),
        models: ['gpt-5.6-sol', 'gpt-5.6-sol-high']
      }),
      'default'
    )
  })

  it('loads and saves endpoints for the Settings Applies-to profile, not only the active bot', async () => {
    $activeGatewayProfile.set('carousel-director')
    $settingsScopeOverride.set('content-studio')
    $profiles.set(
      ['carousel-director', 'content-studio'].map(name => ({
        name,
        has_env: false,
        is_default: false,
        model: null,
        path: '',
        provider: null,
        skill_count: 0
      }))
    )
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    saveCustomEndpoint.mockResolvedValue(savedResponse)

    render(<CustomEndpointsSettings />)

    await waitFor(() => expect(getCustomEndpoints).toHaveBeenCalledWith('content-studio'))
    expect(screen.getByText('Applies to')).toBeTruthy()

    fireEvent.change(await screen.findByPlaceholderText('Axet Proxy'), { target: { value: 'Studio gateway' } })
    fireEvent.change(await screen.findByPlaceholderText('http://localhost:4000/v1'), {
      target: { value: 'https://studio.example.com/v1' }
    })
    fireEvent.change(await screen.findByPlaceholderText('gpt-5.4'), { target: { value: 'studio-model' } })
    fireEvent.click(screen.getByRole('checkbox', { name: 'This endpoint requires no API key' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save and use' }))

    expect(saveCustomEndpoint).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'Studio gateway' }),
      'content-studio'
    )
  })

  it('hydrates the API mode from a saved endpoint', async () => {
    getCustomEndpoints.mockResolvedValue({
      ...savedResponse,
      endpoints: [{ ...savedResponse.endpoints[0], api_mode: 'anthropic_messages' }]
    })

    render(<CustomEndpointsSettings />)

    await screen.findByText('Profile A')
    expect(screen.getByRole('button', { name: 'Anthropic Messages' }).getAttribute('aria-pressed')).toBe('true')
  })

  it('drops a pending save completion after its profile-scoped view unmounts', async () => {
    let resolveSave!: (value: CustomEndpointsResponse) => void
    saveCustomEndpoint.mockReturnValue(new Promise(resolve => (resolveSave = resolve)))
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    const onConfigSaved = vi.fn()
    const onMainModelChanged = vi.fn()

    const view = render(
      <CustomEndpointsSettings onConfigSaved={onConfigSaved} onMainModelChanged={onMainModelChanged} />
    )

    await screen.findByText('No custom endpoints')
    fireEvent.change(screen.getByPlaceholderText('Axet Proxy'), { target: { value: 'Profile A' } })
    fireEvent.change(screen.getByPlaceholderText('http://localhost:4000/v1'), {
      target: { value: 'http://profile-a.test/v1' }
    })
    fireEvent.change(screen.getByPlaceholderText('gpt-5.4'), { target: { value: 'model-a' } })
    fireEvent.click(screen.getByRole('checkbox', { name: 'This endpoint requires no API key' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save and use' }))
    expect(saveCustomEndpoint).toHaveBeenCalledTimes(1)

    // SettingsView keys ProvidersSettings by the selected profile, so a profile
    // switch unmounts this instance while its already-routed write is pending.
    view.unmount()
    await act(async () => resolveSave(savedResponse))

    expect(onMainModelChanged).not.toHaveBeenCalled()
    expect(onConfigSaved).not.toHaveBeenCalled()
    expect(triggerHaptic).not.toHaveBeenCalled()
    expect(notify).not.toHaveBeenCalled()
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('Test retains the typed URL and shows the discovered suggestion without changing it', async () => {
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    validateCustomEndpoint.mockResolvedValue({
      ok: true,
      message: '',
      models: ['model-a'],
      resolved_base_url: 'http://h.test/v1'
    })
    render(<CustomEndpointsSettings onConfigSaved={vi.fn()} onMainModelChanged={vi.fn()} />)

    await screen.findByText('No custom endpoints')
    const urlInput = screen.getByPlaceholderText<HTMLInputElement>('http://localhost:4000/v1')
    fireEvent.change(urlInput, { target: { value: 'http://h.test' } })
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Test' })))

    // Save stores form.baseUrl verbatim and chat POSTs {base_url}/chat/completions, so the
    // typed bare root would 404 every request even though the test looked green.
    expect(urlInput.value).toBe('http://h.test')
    expect(screen.getByRole('status').textContent).toContain('http://h.test/v1')
  })
  it('preserves an independently running localhost endpoint without managed-model setup', async () => {
    getCustomEndpoints.mockResolvedValue(emptyResponse)
    saveCustomEndpoint.mockResolvedValue(savedResponse)
    render(<CustomEndpointsSettings />)
    await screen.findByText('No custom endpoints')
    fireEvent.change(screen.getByPlaceholderText('Axet Proxy'), { target: { value: 'Independent local server' } })
    fireEvent.change(screen.getByPlaceholderText('http://localhost:4000/v1'), {
      target: { value: 'http://localhost:8081/v1' }
    })
    fireEvent.change(screen.getByPlaceholderText('gpt-5.4'), { target: { value: 'my-existing-model' } })
    fireEvent.click(screen.getByRole('checkbox', { name: 'This endpoint requires no API key' }))
    fireEvent.click(screen.getByRole('button', { name: 'Save and use' }))
    expect(saveCustomEndpoint).toHaveBeenCalledWith(
      expect.objectContaining({ base_url: 'http://localhost:8081/v1', model: 'my-existing-model' }),
      'default'
    )
  })
})
