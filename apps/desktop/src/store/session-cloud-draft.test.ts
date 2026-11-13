import { afterEach, expect, it, vi } from 'vitest'

import { desktopSessionCreateParams } from '@/app/session/hooks/use-session-actions/session-create-request'
import {
  $currentModel,
  resetCloudComposerForFreshDraft,
  setCurrentModel,
  setCurrentModelSource,
  setCurrentProvider
} from '@/store/session'

vi.mock('@/store/profile', async importOriginal => ({
  ...await importOriginal<Record<string, unknown>>() ,
  ensureGatewayProfile: vi.fn(async () => undefined)
}))

afterEach(() => {
  setCurrentModel('')
  setCurrentProvider('')
  setCurrentModelSource('')
})

it.each(['semantier-cloud', 'custom:byok'])('fresh draft has the correct model policy for %s', async provider => {
  setCurrentModel('explicit-session-model')
  setCurrentProvider(provider)
  setCurrentModelSource('manual')
  resetCloudComposerForFreshDraft()
  const params = await desktopSessionCreateParams('', null, 'default')

  if (provider === 'semantier-cloud') {
    expect($currentModel.get()).toBe('')
    expect(params).not.toHaveProperty('model')
    expect(params).not.toHaveProperty('provider')
  } else {
    expect($currentModel.get()).toBe('explicit-session-model')
    expect(params).toMatchObject({model:'explicit-session-model',provider})
  }
})
