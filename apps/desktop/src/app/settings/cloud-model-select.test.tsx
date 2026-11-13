import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

import { ModelSelect } from './model-select'

beforeAll(() => {
  stubResizeObserver()
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})
afterEach(cleanup)

it('Settings searches provider/model terms together and commits the public id', async () => {
  const onChange = vi.fn()

  const provider = { slug: 'semantier-cloud', name: 'Semantier Model Provider', authenticated: true,
    eligible_models: ['gpt-4o', 'qwen3.5-plus'], models: ['gpt-4o', 'qwen3.5-plus'],
    model_profiles: { 'gpt-4o': { label: 'GPT-4o', provider_id: 'AZURE_OPENAI', provider_display: 'Azure OpenAI', provider_aliases: ['az'], source_model_id: 'deployment-gpt-4o' }, 'qwen3.5-plus': { label: 'Qwen', provider_id: 'DASHSCOPE', provider_display: 'Alibaba DashScope' } } }

  render(<ModelSelect aria-label="Main model" models={provider.models} onValueChange={onChange} provider={provider} providerSlug="semantier-cloud" value="qwen3.5-plus" />)
  fireEvent.click(screen.getByRole('combobox', { name: 'Main model' }))
  fireEvent.change(screen.getByPlaceholderText('Search models'), { target: { value: ' AZURE GpT-4o ' } })
  await waitFor(() => expect(screen.getAllByRole('option').map(n => n.textContent)).toEqual(['GPT-4o · Azure OpenAI']))
  fireEvent.click(screen.getByRole('option', { name: 'GPT-4o · Azure OpenAI' }))
  expect(onChange).toHaveBeenCalledWith('gpt-4o')
})

it('Settings searches combined BYOK provider/model terms without changing the manual route', async () => {
  const onChange = vi.fn()

  const provider = { slug: 'offer-mvl-azure-search', name: 'Azure local search fixture (synthetic)', authenticated: true,
    models: ['gpt-4o', 'gpt-4o-mini'], discovery_state: 'fresh' as const, discovery_source_id: 'fixture-source' }

  render(<ModelSelect aria-label="BYOK model" models={provider.models} onValueChange={onChange} provider={provider} providerSlug={provider.slug} searchable value="gpt-4o" />)
  fireEvent.click(screen.getByRole('combobox', { name: 'BYOK model' }))
  fireEvent.change(screen.getByPlaceholderText('Model'), { target: { value: '  AZURE gpt-4o  ' } })
  await waitFor(() => expect(screen.getAllByRole('option').map(n => n.textContent)).toEqual(['gpt-4o', 'gpt-4o-mini']))
  expect(onChange).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('option', { name: 'gpt-4o-mini' }))
  expect(onChange).toHaveBeenCalledWith('gpt-4o-mini')
})
