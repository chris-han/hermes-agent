import assert from 'node:assert/strict'
import { fromLocalGit } from './write-build-stamp.mjs'
const result = fromLocalGit(process.argv[2])
assert.ok(result && /^[0-9a-f]{40}$/.test(result.commit))
assert.equal(result.dirty, true)
assert.equal(result.source, 'local')
console.log('WSL_LINUX_GIT_NODE_TEST_PASS')
