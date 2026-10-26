import { expect, test } from 'vitest'

import { UpdateWorkGuard } from './work-review'

test('review binds target, work revision and expiry; idle does not need consent', () => {
  let now = 1
  const state = { work: { count: 1, titles: ['Writing'] }, revision: 1, stopsWork: true }
  const guard = new UpdateWorkGuard(() => state, () => now)
  const review = guard.review('a', '1', '2')!
  expect(review.work.count).toBe(1)
  expect(guard.review('b', '1', '3', review.token)).not.toBeNull()
  const current = guard.review('a', '1', '2')!
  state.revision++
  expect(guard.review('a', '1', '2', current.token)).not.toBeNull()
  const expiring = guard.review('a', '1', '2')!
  now += 120_001
  expect(guard.review('a', '1', '2', expiring.token)).not.toBeNull()
  const fresh = guard.review('a', '1', '2')!
  expect(guard.review('a', '1', '2', fresh.token)).toBeNull()
  now += 120_001 // A large verified download does not expire accepted intent.
  expect(() => guard.assertBeforeStop()).not.toThrow()
  state.revision++
  expect(() => guard.assertBeforeStop()).toThrow('Active work changed')
  state.work = { count: 0, titles: [] }
  expect(guard.review('a', '1', '2')).toBeNull()
  state.work = { count: 1, titles: ['New work'] }
  expect(() => guard.assertBeforeStop()).toThrow('Active work changed')
})
