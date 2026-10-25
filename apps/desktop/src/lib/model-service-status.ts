import type { BillingResult } from '@/app/settings/billing/api'
import { semantierPortalTask } from '@/app/settings/billing/semantier-portal'
import type { BillingBackendState } from '@/app/settings/billing/types'

/** Presentation of existing authoritative Billing v2 facts; never an entitlement authority. */
export interface ModelServiceStatus {
  state: 'setup' | 'ready' | 'plan' | 'preparing' | 'expired' | 'credit' | 'reauth' | 'denied' | 'unavailable'
  detail?: string
  portalUrl?: string
}

export function cloudServiceStatus(result: BillingResult<BillingBackendState>): ModelServiceStatus {
  if (!result.ok) {
    return {
      state: result.refusal.kind === 'provider_not_configured' ? 'setup'
        : result.refusal.kind === 'unauthenticated' ? 'reauth'
        : ['insufficient_scope', 'membership_revoked'].includes(result.refusal.kind) ? 'denied' : 'unavailable',
      detail: result.refusal.message
    }
  }

  const state = result.data

  if (!('schema_version' in state) || state.schema_version !== 'semantier.billing.v2' || !state.account_provisioning) {
    return { state: 'unavailable' }
  }

  // Billing v2 permits a non-wallet account's availability to be null. Never
  // turn that into a zero balance; its model readiness is a separate owner fact.
  const available = state.credits.available

  if (available !== null && (!available || typeof available.amount !== 'string')) {
    return { state: 'unavailable' }
  }

  const projection = state.account_provisioning

  const portalUrl = state.capabilities.can_open_portal
    ? semantierPortalTask(state.portal.url, state.organization.id, state.billing_account_id, 'plans') ?? undefined
    : undefined

  if (projection.inference_ready) { return { state: 'ready' } }

  if (projection.local_status !== 'READY') { return { state: 'denied' } }

  if (projection.reason === 'PREPARING') { return { state: 'preparing' } }

  if (available === null) { return { state: 'unavailable' } }

  if (!state.subscription) { return { state: 'plan', portalUrl } }

  if (state.subscription.status === 'expired') { return { state: 'expired', portalUrl } }

  // Exact decimal zero comparison; never round balances or derive spend permission here.
  if (/^-?0+(?:\.0+)?$/.test(available.amount)) { return { state: 'credit', portalUrl } }

  return { state: 'unavailable' }
}
