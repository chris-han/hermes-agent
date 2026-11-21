import { describe, expect, it } from 'vitest'

import { resolveAccountModelDefault } from './account-model-default'

const readiness = { account_id: 'acct-a', account_provisioning: { inference_ready: true },
  catalog: { status: 'available', data: [{ id: 'model-a', account_id: 'acct-a', access: 'authorized' }] } }

describe('account model default', () => {
  it('uses the exact account preference only when the device catalog authorizes it', () => {
    expect(resolveAccountModelDefault({ account_id: 'acct-a', model: 'model-a' }, null, readiness)).toBe('model-a')
    expect(resolveAccountModelDefault({ account_id: 'acct-a', model: 'platform_recommended' }, { recommended_model: 'model-a' }, readiness)).toBe('model-a')
    expect(resolveAccountModelDefault({ account_id: 'acct-a', model: null }, { recommended_model: 'model-a' }, readiness)).toBe('model-a')
  })

  it('does not invent a default for missing, foreign, ambiguous or unavailable authority', () => {
    expect(resolveAccountModelDefault({ account_id: 'other', model: 'model-a' }, null, readiness)).toBeNull()
    expect(resolveAccountModelDefault({ account_id: 'acct-a', model: 'other' }, null, readiness)).toBeNull()
    expect(resolveAccountModelDefault({ account_id: 'acct-a', model: 'platform_recommended' }, null, readiness)).toBeNull()
    expect(resolveAccountModelDefault({ account_id: 'acct-a', model: 'model-a' }, null,
      { ...readiness, account_provisioning: { inference_ready: false } })).toBeNull()
    expect(resolveAccountModelDefault({ account_id: 'acct-a', model: 'model-a' }, null,
      { ...readiness, catalog: { status: 'available', data: [...readiness.catalog.data, ...readiness.catalog.data] } })).toBeNull()
    expect(resolveAccountModelDefault({ account_id: 'acct-a', model: 'model-a' }, null,
      { ...readiness, catalog: { status: 'available', data: [{ ...readiness.catalog.data[0], access: 'denied' }] } })).toBeNull()
  })
})
