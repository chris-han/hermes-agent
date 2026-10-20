import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { openExternal } from './open-external'
import { SemantierBilling } from './semantier-billing'
import type { SemantierBillingV2Response } from './types'

vi.mock('./open-external', () => ({ openExternal: vi.fn() }))
afterEach(cleanup)

const amount = { currency: 'CNY', amount: '100.00' }

const state: SemantierBillingV2Response = {
  ok: true, backend: 'semantier', schema_version: 'semantier.billing.v2', billing_account_id: 'account-a',
  organization: { id: 'org-a' }, merchant: null, currency: 'CNY', environment: 'DEVELOPMENT',
  subscription: { plan: 'Professional', term: 'annual', status: 'active', starts_at: '2026-10-20T00:00:00Z',
    ends_at: '2027-10-20T00:00:00Z', manual_renewal: true },
  credits: { allowance: amount, purchased: { currency: 'CNY', amount: '20.00' },
    available: amount, held: { currency: 'CNY', amount: '1.00' } },
  usage: { known_cost: { currency: 'CNY', amount: '0.25' }, period: 'lifetime' },
  capabilities: { can_read_usage: true, can_manage_billing: false, can_open_portal: true,
    can_charge: false, can_change_plan: false, can_auto_top_up: false },
  portal: { url: 'https://billing.example' }, as_of: '2026-10-20T00:00:00Z'
}

it('renders actual rights separately and opens an account task with independent browser sign-in', () => {
  render(<SemantierBilling state={state} />)
  expect(screen.getByText('Professional')).toBeTruthy()
  expect(screen.getByText('Manual renewal')).toBeTruthy()
  expect(screen.getByText('2027-10-20T00:00:00Z')).toBeTruthy()
  expect(screen.getByText('CNY 20.00')).toBeTruthy()
  expect(screen.getByText('CNY 1.00')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Orders' }))
  expect(openExternal).toHaveBeenCalledWith('https://billing.example/billing/orders?account_id=account-a')
})

it('missing portal offers no commercial links and never substitutes Nous', () => {
  render(<SemantierBilling state={{ ...state, portal: { url: null } }} />)
  expect(screen.queryByRole('button')).toBeNull()
  expect(screen.getByText('Payments, plan changes and automatic top-up are unavailable.')).toBeTruthy()
})
