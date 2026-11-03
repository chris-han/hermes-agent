import { spawnSync } from 'node:child_process'
import { mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { afterEach, expect, test } from 'vitest'

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const hook = pathToFileURL(path.join(repo, 'scripts/build/external-node-resolver.mjs')).href
const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

function fixture() {
  const root = mkdtempSync(path.join(os.tmpdir(), 'external-node-resolution-'))
  roots.push(root)
  const source = path.join(root, 'source'), deps = path.join(root, 'deps')
  const app = path.join(source, 'apps/desktop'), mirror = path.join(deps, 'apps/desktop')
  mkdirSync(app, { recursive: true })
  mkdirSync(path.join(mirror, 'node_modules/probe'), { recursive: true })
  writeFileSync(path.join(mirror, 'package.json'), '{}')
  writeFileSync(path.join(mirror, 'node_modules/probe/package.json'), JSON.stringify({
    name: 'probe', type: 'module', exports: { import: './entry.mjs', require: './entry.cjs' }
  }))
  writeFileSync(path.join(mirror, 'node_modules/probe/entry.mjs'), 'export default "esm"')
  writeFileSync(path.join(mirror, 'node_modules/probe/entry.cjs'), 'module.exports="cjs"')
  const entry = path.join(app, 'entry.mjs')
  writeFileSync(entry, 'import x from "probe";import {createRequire} from "node:module";const r=createRequire(import.meta.url);console.log(x,r("probe"),r.resolve("probe"))')
  const run = (override = {}) => spawnSync(process.execPath, ['--import', hook, entry], {
    encoding: 'utf8', env: { ...process.env, HERMES_NODE_SOURCE_ROOT: source, HERMES_NODE_DEPENDENCY_ROOT: deps, ...override }
  })
  return { root, source, deps, app, mirror, run }
}

test('external Node resolver preserves ESM and CommonJS export conditions', () => {
  const t = fixture(), result = t.run()
  expect(result.status, result.stderr).toBe(0)
  expect(result.stdout).toContain(`esm cjs ${path.join(t.mirror, 'node_modules/probe/entry.cjs')}`)
})

test('external Node resolver refuses missing or overlapping roots', () => {
  const t = fixture()
  for (const override of [{ HERMES_NODE_DEPENDENCY_ROOT: '' }, { HERMES_NODE_DEPENDENCY_ROOT: t.source }]) {
    const result = t.run(override)
    expect(result.status).not.toBe(0)
    expect(result.stderr).toMatch(/absolute directory|must not overlap/)
  }
})

test('external Node resolver refuses a package redirected outside the selected roots', () => {
  const t = fixture()
  const outside = path.join(t.root, 'outside')
  mkdirSync(outside)
  writeFileSync(path.join(outside, 'package.json'), '{"name":"probe","type":"module","exports":"./entry.mjs"}')
  writeFileSync(path.join(outside, 'entry.mjs'), 'export default "unexpected"')
  const packagePath = path.join(t.mirror, 'node_modules/probe')
  rmSync(packagePath, { recursive: true })
  symlinkSync(outside, packagePath, process.platform === 'win32' ? 'junction' : 'dir')
  const result = t.run()
  expect(result.status).not.toBe(0)
  expect(result.stderr).toContain('escaped the selected roots')
})
