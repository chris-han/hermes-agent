import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { fmtDateTime } from '@/lib/time'

import { openExternal } from './open-external'
import { formatSemantierMoney, SemantierBilling } from './semantier-billing'
import type { SemantierBillingV2Response } from './types'

vi.mock('./open-external', () => ({ openExternal: vi.fn() }))
afterEach(cleanup)

const amount = { currency: 'CNY', amount: '100.000000000000' }

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
  expect(screen.getByText(fmtDateTime.format(new Date(state.subscription!.ends_at)))).toBeTruthy()
  expect(screen.getByText(fmtDateTime.format(new Date(state.as_of)))).toBeTruthy()
  expect(screen.getAllByText('CNY 100.00')).toHaveLength(2)
  expect(screen.queryByText('2027-10-20T00:00:00Z')).toBeNull()
  expect(screen.getByText('CNY 20.00')).toBeTruthy()
  expect(screen.getByText('CNY 1.00')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Orders' }))
  expect(openExternal).toHaveBeenCalledWith('https://billing.example/billing/orders?organization_id=org-a&account_id=account-a')
  fireEvent.click(screen.getByRole('button', { name: 'Choose / renew package' }))
  expect(openExternal).toHaveBeenLastCalledWith('https://billing.example/pricing?organization_id=org-a&account_id=account-a')
})

it('renders unavailable customer-cloud credit without a crash or invented zero balance', () => {
  render(<SemantierBilling state={{ ...state,
    credits: { ...state.credits, available: null } }} />)
  expect(screen.getByText('Professional')).toBeTruthy()
  expect(screen.getByText('CNY 1.00')).toBeTruthy()
  expect(screen.queryByText('CNY 0.00')).toBeNull()
  expect(screen.getByRole('button', { name: 'Orders' })).toBeTruthy()
})

it('missing portal offers no commercial links and never substitutes Nous', () => {
  render(<SemantierBilling state={{ ...state, portal: { url: null } }} />)
  expect(screen.queryByRole('button')).toBeNull()
  expect(screen.getByText('Payments, plan changes and automatic top-up are unavailable.')).toBeTruthy()
})


it('formats exact credit decimals without floating-point conversion or rounding', () => {
  for (const [amount, expected] of [
    ['100.000000000000', '100.00'], ['1', '1.00'], ['1.1', '1.10'],
    ['9007199254740993.120000', '9007199254740993.12'],
    ['0.000000000000000001000', '0.000000000000000001'], ['-0.0100', '-0.01']
  ]) {
    expect(formatSemantierMoney({ amount, currency: 'CNY' })).toBe(`CNY ${expected}`)
  }

  expect(state.credits.allowance.amount).toBe('100.000000000000')
})

it('keeps paid rights visible while inference readiness is pending', () => {
  const {rerender}=render(<SemantierBilling state={{...state,account_provisioning:{local_status:'READY',gateway_sync_status:'PENDING',inference_ready:false,reason:'PREPARING'}}} />)
  expect(screen.getByText('Professional')).toBeTruthy()
  expect(screen.getAllByText('CNY 100.00')).toHaveLength(2)
  expect(screen.getByText('Preparing inference access')).toBeTruthy()
  expect(screen.queryByText('Ready for inference')).toBeNull()
  rerender(<SemantierBilling state={{...state,account_provisioning:{local_status:'READY',gateway_sync_status:'SYNCED',inference_ready:true}}} />)
  expect(screen.getByText('Ready for inference')).toBeTruthy()
})
