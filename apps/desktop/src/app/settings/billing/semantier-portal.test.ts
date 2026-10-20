import { expect, it } from 'vitest'

import { semantierPortalTask } from './semantier-portal'

it('opens only a configured origin and finite account-scoped tasks without secrets', () => {
  expect(semantierPortalTask('https://billing.example', 'account / a', 'orders'))
    .toBe('https://billing.example/billing/orders?account_id=account+%2F+a')
  expect(semantierPortalTask('http://127.0.0.1:5173', 'a', 'plans'))
    .toBe('http://127.0.0.1:5173/billing/plans?account_id=a')

  for (const unsafe of [null, 'https://user:token@billing.example', 'https://billing.example?key=secret',
    'https://billing.example/#token', 'https://billing.example/redirect', 'http://untrusted.example']) {
    expect(semantierPortalTask(unsafe, 'a', 'manage')).toBeNull()
  }
})
