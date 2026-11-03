import { mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, expect, test } from 'vitest'
import { workspaceTool } from '../scripts/build/frontend-common.mjs'

const roots = []
afterEach(() => {
  delete process.env.HERMES_DESKTOP_NODE_ROOT
  delete process.env.HERMES_DESKTOP_BUILD_DIR
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function packageAt(root, value) {
  const pkg = path.join(root, 'node_modules/probe-tool')
  mkdirSync(pkg, { recursive: true })
  writeFileSync(path.join(root, 'package.json'), '{}')
  writeFileSync(path.join(pkg, 'package.json'), '{"name":"probe-tool","main":"index.js"}')
  writeFileSync(path.join(pkg, 'index.js'), `module.exports=${JSON.stringify(value)}`)
}

test('external workspace tools resolve only from the selected dependency tree', () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'external-workspace-tool-'))
  roots.push(root)
  const source = path.join(root, 'source'), deps = path.join(root, 'deps')
  mkdirSync(source); mkdirSync(deps)
  packageAt(source, 'source'); packageAt(deps, 'external')
  process.env.HERMES_DESKTOP_BUILD_DIR = path.join(root, 'job')
  process.env.HERMES_DESKTOP_NODE_ROOT = deps
  expect(workspaceTool(source, 'apps/desktop', 'probe-tool')).toBe(path.join(deps, 'node_modules/probe-tool/index.js'))
  rmSync(path.join(deps, 'node_modules/probe-tool'), { recursive: true })
  expect(() => workspaceTool(source, 'apps/desktop', 'probe-tool')).toThrow(/Missing prepared/)
  symlinkSync(path.join(source, 'node_modules/probe-tool'), path.join(deps, 'node_modules/probe-tool'),
    process.platform === 'win32' ? 'junction' : 'dir')
  expect(() => workspaceTool(source, 'apps/desktop', 'probe-tool')).toThrow(/escaped/)
  process.env.HERMES_DESKTOP_NODE_ROOT = source
  expect(() => workspaceTool(source, 'apps/desktop', 'probe-tool')).toThrow(/overlap/)
})
