import { describe, expect, it } from 'vitest'

import type { SemantierBillingV2Response } from '@/app/settings/billing/types'
import { BUNDLED_LOCALES } from '@/i18n'
import { modelServiceCopy } from '@/i18n/model-service'

import { cloudServiceStatus } from './model-service-status'
import { evaluateRuntimeReadiness } from './runtime-readiness'

const amount = { amount: '100.00', currency: 'CNY' }

const ready: SemantierBillingV2Response = {
  ok: true,
  backend: 'semantier',
  schema_version: 'semantier.billing.v2',
  billing_account_id: 'account-a',
  organization: { id: 'org-a' },
  merchant: null,
  currency: 'CNY',
  environment: 'DEVELOPMENT',
  subscription: {
    plan: 'Professional',
    term: 'monthly',
    status: 'active',
    starts_at: '2026-10-20T00:00:00Z',
    ends_at: '2026-11-20T00:00:00Z',
    manual_renewal: true
  },
  credits: { allowance: amount, purchased: amount, available: amount, held: { amount: '0', currency: 'CNY' } },
  usage: { known_cost: { amount: '0', currency: 'CNY' }, period: 'lifetime' },
  capabilities: {
    can_read_usage: true,
    can_manage_billing: false,
    can_open_portal: true,
    can_charge: false,
    can_change_plan: false,
    can_auto_top_up: false
  },
  portal: { url: 'https://cloud.semantier.com' },
  as_of: '2026-10-25T00:00:00Z',
  account_provisioning: { local_status: 'READY', inference_ready: true, reason: null }
}

describe('selected model-service facts', () => {
  it('keeps transport, authentication and authorization separate from missing subscription', () => {
    for (const kind of ['transport', 'timeout', 'billing_unavailable']) {
      expect(cloudServiceStatus({ ok: false, refusal: { kind, message: 'Unavailable' } }).state).toBe('unavailable')
    }

    expect(
      cloudServiceStatus({ ok: false, refusal: { kind: 'provider_not_configured', message: 'Not configured' } }).state
    ).toBe('setup')
    expect(cloudServiceStatus({ ok: false, refusal: { kind: 'unauthenticated', message: 'Expired' } }).state).toBe(
      'reauth'
    )
    expect(cloudServiceStatus({ ok: false, refusal: { kind: 'membership_revoked', message: 'Denied' } }).state).toBe(
      'denied'
    )
  })

  it('projects real rights and provisioning independently', () => {
    const pending = {
      ...ready,
      account_provisioning: { local_status: 'READY', inference_ready: false, reason: 'NO_ACCESS' }
    }
    expect(cloudServiceStatus({ ok: true, data: ready }).state).toBe('ready')
    expect(cloudServiceStatus({ ok: true, data: { ...pending, subscription: null } }).state).toBe('plan')
    expect(
      cloudServiceStatus({
        ok: true,
        data: { ...pending, account_provisioning: { ...pending.account_provisioning, reason: 'PREPARING' } }
      }).state
    ).toBe('preparing')
    expect(
      cloudServiceStatus({
        ok: true,
        data: { ...pending, subscription: { ...ready.subscription!, status: 'expired' } }
      }).state
    ).toBe('expired')
    expect(
      cloudServiceStatus({
        ok: true,
        data: { ...pending, credits: { ...ready.credits, available: { amount: '0.000', currency: 'CNY' } } }
      }).state
    ).toBe('credit')
  })

  it('keeps a null wallet unavailable without inventing exhaustion or overriding independent model readiness', () => {
    const nonWallet = { ...ready, credits: { ...ready.credits, available: null } }
    expect(cloudServiceStatus({ ok: true, data: nonWallet }).state).toBe('ready')
    const notReady = {
      ...nonWallet,
      account_provisioning: { local_status: 'READY', inference_ready: false, reason: 'NO_ACCESS' }
    }
    expect(cloudServiceStatus({ ok: true, data: notReady }).state).toBe('unavailable')
    expect(cloudServiceStatus({ ok: true, data: { ...notReady, subscription: null } }).state).toBe('unavailable')
  })

  it('reads Billing only for the actual selected Cloud route and never invokes a model', async () => {
    const methods: string[] = []
    let cloud = false

    const request = async <T>(method: string): Promise<T> => {
      methods.push(method)

      if (method === 'setup.status') {
        return { provider_configured: true } as T
      }

      if (method === 'setup.runtime_check') {
        return { ok: true, source: cloud ? 'semantier-cloud' : 'env' } as T
      }

      if (method === 'billing.state') {
        return ready as T
      }
      throw new Error(`Forbidden startup method: ${method}`)
    }

    expect((await evaluateRuntimeReadiness(request)).ready).toBe(true)
    expect(methods).toEqual(['setup.status', 'setup.runtime_check'])
    methods.length = 0
    cloud = true
    expect((await evaluateRuntimeReadiness(request)).modelService?.state).toBe('ready')
    expect(methods).toEqual(['setup.status', 'setup.runtime_check', 'billing.state'])
  })

  it('has the same complete copy keys for every bundled locale', () => {
    expect(Object.keys(modelServiceCopy).sort()).toEqual([...BUNDLED_LOCALES].sort())

    for (const locale of BUNDLED_LOCALES) {
      expect(Object.keys(modelServiceCopy[locale]).sort()).toEqual(Object.keys(modelServiceCopy.en).sort())
    }
  })
})
