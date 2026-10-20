import { isSemantierBilling } from '@hermes/shared/billing'
import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import type { SemantierBillingCopy } from '@/i18n/semantier-billing'
import { semantierCommerceCopy } from '@/i18n/semantier-commerce'
import { BarChart3, Package } from '@/lib/icons'
import { $activeGatewayRoute } from '@/store/gateway'
import { $connection } from '@/store/session'

import { ListRow, SettingsSection } from '../primitives'

import type { BillingRefusal, BillingResult } from './api'
import { openExternal } from './open-external'
import { semantierPortalTask } from './semantier-portal'
import type { BillingBackendState, SemantierBillingResponse, SemantierBillingStateResponse } from './types'

interface SemantierBillingProps {
  state?: SemantierBillingResponse
  refusal?: BillingRefusal
}

export function semantierBillingProps(result?: BillingResult<BillingBackendState>): SemantierBillingProps | null {
  if (!result) {return null}

  if (!result.ok) {return result.refusal.backend === 'semantier' ? { refusal: result.refusal } : null}

  return isSemantierBilling(result.data) ? { state: result.data } : null
}

const money = (value: unknown): string | null => {
  if (typeof value !== 'object' || value === null || !('amount' in value) || !('currency' in value)) {return null}

  return typeof value.amount === 'string' && typeof value.currency === 'string'
    ? `${value.currency} ${value.amount}` : null
}

function row(title: string, value: string) {
  return <ListRow action={<span className="min-w-0 break-words text-right tabular-nums">{value}</span>} key={title} title={title} />
}

function scopeRows(state: SemantierBillingStateResponse, c: SemantierBillingCopy, connection: string) {
  const funding: Record<string, string> = { CUSTOMER_CLOUD: c.customerCloud, SEMANTIER_CREDITS: c.platformCredits }
  const mode: Record<string, string> = { shadow: c.shadow, enforced: c.enforced }

  return <>
    {row(c.organization, state.organization.display_name || state.organization.id)}
    {state.organization.display_name && row(c.organization + ' ID', state.organization.id)}
    {row(c.account, state.billing_account_id)}
    {row(c.connection, connection)}
    {row(c.funding, funding[state.funding_source] || c.unavailable)}
    {row(c.mode, mode[state.mode] || c.unavailable)}
  </>
}

function budgetValue(state: SemantierBillingStateResponse, c: SemantierBillingCopy): string {
  const budget = state.account.budget

  if (typeof budget !== 'object' || budget === null || !('currency' in budget) || !('limit' in budget) || !('spent' in budget)) {return c.unavailable}

  return `${String(budget.currency)} ${String(budget.limit)} / ${String(budget.spent)}`
}

function usageRows(state: SemantierBillingStateResponse, c: SemantierBillingCopy) {
  const statuses: Record<string, string> = {
    ESTIMATED: c.estimated, COMPLETE: c.complete, KNOWN: c.complete, RATED: c.complete,
    UNPRICED: c.unpriced, PENDING: c.pendingState, PARTIAL: c.pendingState, UNAVAILABLE: c.unavailable
  }

  const costLabel = state.usage.period === 'lifetime' ? `${c.knownCost} · ${c.lifetime}` : c.knownCost

  return <>
    {row(costLabel, money(state.usage.known_cost) || c.unavailable)}
    {row(c.status, statuses[state.usage.status] || c.unavailable)}
    {row(c.pending, state.usage.unresolved_attempts === null ? c.unavailable : String(state.usage.unresolved_attempts))}
    {row(c.budget, budgetValue(state, c))}
    {state.balances.length ? state.balances.map((balance, index) => row(`${c.balance} ${index + 1}`, money(balance) || c.unavailable)) : (
      <p className="py-3 text-sm text-(--ui-text-tertiary)">{c.walletUnavailable}</p>
    )}
    {state.balances.length > 0 && row(c.available, money(state.account.available) || c.unavailable)}
    {state.balances.length > 0 && row(c.held, money(state.account.held) || c.unavailable)}
    {row(c.updated, state.as_of)}
  </>
}

export function SemantierBilling({ state, refusal }: SemantierBillingProps) {
  const { t, locale } = useI18n()
  const b = t.settings.billing
  const c = b.semantier
  const connection = useStore($connection)
  const profile = useStore($activeGatewayRoute)
  const connectionLabel = `${connection?.baseUrl ?? connection?.mode ?? '—'} / ${profile || 'default'}`

  if (!state) {return <SettingsSection icon={Package} title="Semantier">
    {row(c.connection, connectionLabel)}
    <p className="py-3 text-sm text-(--ui-text-tertiary)" role="status">{refusal?.message || c.unavailable}</p>
    <p className="text-sm text-(--ui-text-tertiary)">{c.recovery}</p>
  </SettingsSection>}

  if (state.schema_version === 'semantier.billing.v2') {
    const commerce = semantierCommerceCopy[locale] ?? semantierCommerceCopy.en
    const tasks = ['manage', 'plans', 'topUp', 'orders'] as const

    return <>
      <SettingsSection icon={Package} title="Semantier">
        {row(c.organization, state.organization.display_name || state.organization.id)}
        {row(c.account, state.billing_account_id)}
        {row(commerce.merchant, state.merchant?.display_name ?? c.unavailable)}
        {row(commerce.currency, state.currency)}
        {row(c.connection, connectionLabel)}
        {row(commerce.package, state.subscription?.plan ?? commerce.noPackage)}
        {state.subscription && row(c.status, state.subscription.status === 'active' ? commerce.active : commerce.expired)}
        {state.subscription && row(commerce.term, commerce[state.subscription.term])}
        {state.subscription && row(commerce.expires, state.subscription.ends_at)}
        {state.subscription && row(commerce.renewal, commerce.manual)}
        {row(commerce.environment, state.environment)}
      </SettingsSection>
      <SettingsSection icon={BarChart3} title={b.sections.usage}>
        {row(commerce.allowance, money(state.credits.allowance) || c.unavailable)}
        {row(commerce.purchased, money(state.credits.purchased) || c.unavailable)}
        {row(c.held, money(state.credits.held) || c.unavailable)}
        {row(c.available, money(state.credits.available) || c.unavailable)}
        {row(`${c.knownCost} · ${c.lifetime}`, money(state.usage.known_cost) || c.unavailable)}
        {row(c.updated, state.as_of)}
      </SettingsSection>
      {state.capabilities.can_open_portal && <div className="flex flex-wrap gap-2">
        {tasks.map(task => {
          const url = semantierPortalTask(state.portal.url, state.organization.id, state.billing_account_id, task)

          return url && <Button key={task} onClick={() => openExternal(url)} variant="outline">{commerce[task]}</Button>
        })}
      </div>}
      <p className="py-3 text-sm text-(--ui-text-tertiary)">{state.capabilities.can_open_portal && state.portal.url ? commerce.browserAuth : c.commercialUnavailable}</p>
    </>
  }

  return <>
    <SettingsSection icon={Package} title="Semantier">{scopeRows(state, c, connectionLabel)}</SettingsSection>
    <SettingsSection icon={BarChart3} title={b.sections.usage}>{usageRows(state, c)}</SettingsSection>
    <p className="text-sm text-(--ui-text-tertiary)">{c.commercialUnavailable}</p>
  </>
}
