import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isCanaryTag } from './feature-flags'

test('isCanaryTag recognizes canary stamps and rejects stable/dev tags', () => {
  assert.equal(isCanaryTag('v0.28.0+canary.20260818T123456Z'), true)
  assert.equal(isCanaryTag('v0.28.0-canary.20260818'), false)
  assert.equal(isCanaryTag('v0.28.0+canary.20260818123456'), false)
  assert.equal(isCanaryTag('v0.28.0'), false)
  assert.equal(isCanaryTag(''), false)
  assert.equal(isCanaryTag(null), false)
  assert.equal(isCanaryTag(undefined), false)
})
