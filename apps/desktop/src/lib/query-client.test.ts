import { beforeEach, describe, expect, it } from 'vitest'

import { invalidateProfileScopedQueries, queryClient } from './query-client'

function invalidated(key: unknown[]): boolean {
  return queryClient.getQueryState(key)?.isInvalidated ?? false
}

describe('invalidateProfileScopedQueries', () => {
  beforeEach(() => {
    queryClient.clear()
  })

  it('invalidates profile-scoped caches and leaves account/global caches intact', () => {
    const profileScoped = [
      ['hermes-config-record'],
      ['hermes-config-schema'],
      ['skills-list'],
      ['toolsets-list'],
      ['model-options', 'global'],
      ['command-palette', 'sessions'],
      ['session-picker', 'sessions']
    ]

    const global = [
      ['marketplace-themes', 'all'],
      ['marketplace-themes-settings', 'x'],
      ['onboarding-model-options', 'y'],
      ['contrib-logs-tail']
    ]

    for (const key of [...profileScoped, ...global]) {
      queryClient.setQueryData(key, { seeded: true })
    }

    invalidateProfileScopedQueries()

    for (const key of profileScoped) {
      expect(invalidated(key), `${JSON.stringify(key)} should be invalidated`).toBe(true)
    }

    for (const key of global) {
      expect(invalidated(key), `${JSON.stringify(key)} should be left intact`).toBe(false)
    }
  })

  it('removes account data and cancels in-flight billing on source switches', async () => {
    let resolve!: (value: unknown) => void

    const pending = new Promise<unknown>(done => {
      resolve = done
    })

    queryClient.setQueryData(['billing', 'state', 'old'], { organization: 'old-org' })
    const request = queryClient.fetchQuery({ queryKey: ['billing', 'subscription', 'old'], queryFn: () => pending })
    const settled = request.catch(() => undefined)
    invalidateProfileScopedQueries()
    resolve({ organization: 'old-org' })
    await settled
    expect(queryClient.getQueriesData({ queryKey: ['billing'] })).toEqual([])
  })

  it('invalidates unknown/non-string-rooted keys by default (correctness-safe)', () => {
    queryClient.setQueryData(['some-future-profile-query'], 1)
    queryClient.setQueryData([{ scope: 'weird' }], 1)

    invalidateProfileScopedQueries()

    expect(invalidated(['some-future-profile-query'])).toBe(true)
    expect(invalidated([{ scope: 'weird' }])).toBe(true)
  })
})
