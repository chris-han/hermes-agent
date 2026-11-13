import { catalogModelSearchText, catalogSearchMatches } from '@hermes/shared'
import { describe, expect, it } from 'vitest'

const provider = {
  slug: 'semantier-cloud', name: 'Semantier Model Provider', aliases: ['cloud'],
  model_profiles: {
    'gpt-4o': { label: 'GPT-4o', provider_id: 'AZURE_OPENAI', provider_display: 'Azure OpenAI', provider_aliases: ['az', 'openai'], source_model_id: 'deployment-blue-gpt4o' },
    'qwen3.5-plus': { label: 'Qwen3.5 Plus', provider_id: 'DASHSCOPE', provider_display: 'Alibaba DashScope', provider_aliases: ['aliyun'], source_model_id: 'qwen3.5-plus' }
  }
}

describe('shared cloud catalog search contract', () => {
  it('matches every trimmed case-insensitive term across provider and model identities', () => {
    const text = catalogModelSearchText(provider, 'gpt-4o')

    for (const query of [' AZURE gPt-4o ', 'OPENAI', 'azure_openai', 'az blue', 'deployment-blue', 'gpt-4']) {
      expect(catalogSearchMatches(text, query), query).toBe(true)
    }

    expect(catalogSearchMatches(text, 'azure qwen')).toBe(false)
    expect(catalogSearchMatches(catalogModelSearchText(provider, 'qwen3.5-plus'), 'aliyun qwen')).toBe(true)
  })

  it('keeps wire ids and legacy provider aliases without inventing missing metadata', () => {
    const text = catalogModelSearchText({ slug: 'private', name: 'Private', aliases: ['custom:private'] }, 'k3')
    expect(catalogSearchMatches(text, 'custom:private kimi')).toBe(true)
    expect(catalogSearchMatches(text, 'azure')).toBe(false)
    expect(catalogSearchMatches(text, '   ')).toBe(true)
  })
})


it('a declared non-chat modality never appears as a Cloud chat choice', async () => {
  const { providerSelectableModels } = await import('./model-options')
  const result = providerSelectableModels({ slug: 'semantier-cloud', name: 'Semantier', authenticated: true, eligible_models: ['chat', 'image'], models: ['chat', 'image'], model_profiles: { chat: { modality: 'chat' }, image: { modality: 'image' } } })
  expect(result).toEqual(['chat'])
})
