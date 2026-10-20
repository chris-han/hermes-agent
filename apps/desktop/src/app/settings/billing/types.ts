import type {
  BillingAutoReload,
  BillingBackendState,
  BillingCardInfo,
  BillingChargeResponse,
  BillingChargeStatusResponse,
  BillingErrorPayload,
  BillingMonthlyCap,
  BillingMutationResponse,
  BillingRefusalCode,
  ChargeFailureReason,
  SemantierBillingResponse,
  SemantierBillingStateResponse,
  SemantierBillingV2Response,
  SemantierSubscriptionStateResponse,
  BillingStateResponse as SharedBillingStateResponse,
  SubscriptionBackendState,
  SubscriptionPreviewResponse,
  SubscriptionStateResponse,
  SubscriptionTierOption,
  UsageBarData,
  UsageModelData
} from '@hermes/shared/billing'

/**
 * The gateway's `billing.state` payload as THIS app reads it: the shared shape
 * plus the free-tier fields newer gateways add. When `free_tier_account` is true there
 * is no account behind the call — `logged_in` is false and there is no balance,
 * card or usage — so the free-tier branch must be read before the logged-out
 * one. Both fields are absent on older gateways.
 */
export type BillingStateResponse = SharedBillingStateResponse & {
  free_tier_account?: boolean
  free_tier_model?: null | string
}

export type {
  BillingAutoReload,
  BillingBackendState,
  BillingCardInfo,
  BillingChargeResponse,
  BillingChargeStatusResponse,
  BillingErrorPayload,
  BillingMonthlyCap,
  BillingMutationResponse,
  BillingRefusalCode,
  ChargeFailureReason,
  SemantierBillingResponse,
  SemantierBillingStateResponse,
  SemantierBillingV2Response,
  SemantierSubscriptionStateResponse,
  SharedBillingStateResponse,
  SubscriptionBackendState,
  SubscriptionPreviewResponse,
  SubscriptionStateResponse,
  SubscriptionTierOption,
  UsageBarData,
  UsageModelData
}
