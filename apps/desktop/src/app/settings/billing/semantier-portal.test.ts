import { expect, it } from 'vitest'

import { semantierPortalTask } from './semantier-portal'

it('opens only a configured origin and finite account-scoped tasks without secrets', () => {
  expect(semantierPortalTask('https://billing.example', 'org / a', 'account / a', 'orders'))
    .toBe('https://billing.example/billing/orders?organization_id=org+%2F+a&account_id=account+%2F+a')
  expect(semantierPortalTask('http://127.0.0.1:5173', 'org-a', 'a', 'plans'))
    .toBe('http://127.0.0.1:5173/billing/plans?organization_id=org-a&account_id=a')

  for (const unsafe of [null, 'https://user:token@billing.example', 'https://billing.example?key=secret',
    'https://billing.example/#token', 'https://billing.example/redirect', 'http://untrusted.example']) {
    expect(semantierPortalTask(unsafe, 'org-a', 'a', 'manage')).toBeNull()
  }
})


it('keeps same-named billing accounts in different organizations distinct', () => {
  const first = semantierPortalTask('https://billing.example', 'org-a', 'shared-account', 'plans')!
  const second = semantierPortalTask('https://billing.example', 'org-b', 'shared-account', 'plans')!

  expect(new URL(first).searchParams.get('organization_id')).toBe('org-a')
  expect(new URL(second).searchParams.get('organization_id')).toBe('org-b')
  expect(first).not.toBe(second)
  expect(semantierPortalTask('https://billing.example', '', 'shared-account', 'plans')).toBeNull()
})
