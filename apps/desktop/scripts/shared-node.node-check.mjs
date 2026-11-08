import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createRequire } from 'node:module'
import { test } from 'node:test'
import { preparedBuilderConfiguration } from './run-electron-builder.mjs'

test('Windows MSIX omits only the private Playwright Node and preserves its driver', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'shared-node-msix-'))
  const app = path.join(root, 'apps/desktop')
  fs.mkdirSync(app, { recursive: true })
  const original = path.join(app, 'electron-builder.config.cjs')
  fs.writeFileSync(original, 'module.exports={extraResources:[{from:"build/agent-payload",to:"agent-payload",filter:["**/*"]}]}')
  const result = preparedBuilderConfiguration(root, ['--win', 'msix'])
  const require = createRequire(import.meta.url)
  const resource = require(result.config).extraResources[0]
  assert.ok(resource.filter.includes('!venv/Lib/site-packages/playwright/driver/node.exe'))
  assert.equal(resource.from, 'build/agent-payload')
  assert.equal(resource.to, 'agent-payload')
  assert.deepEqual(require(original).extraResources[0].filter, ['**/*'])
  // Keep ordinary Python driver modules, CLI JS, and the shared PM-owned runtime.
  for (const required of ['playwright/driver/package/cli.js', 'playwright/_impl/_driver.py', 'tools/node-pinned/node.exe']) {
    assert.ok(!resource.filter.some(pattern => pattern === `!venv/Lib/site-packages/${required}` || pattern === `!${required}`))
  }
})

test('directory builds retain the private Node until an explicit MSIX composition', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'shared-node-dir-'))
  const app = path.join(root, 'apps/desktop')
  fs.mkdirSync(app, { recursive: true })
  const original = path.join(app, 'electron-builder.config.cjs')
  fs.writeFileSync(original, 'module.exports={extraResources:[{from:"build/agent-payload",to:"agent-payload",filter:["**/*"]}]}')
  const result = preparedBuilderConfiguration(root, ['--dir'])
  assert.equal(result.config, 'electron-builder.config.cjs')
})
