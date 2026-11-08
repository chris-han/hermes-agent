import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, test } from 'node:test'
import { checkDesktopLaunch } from './assert-dist-built.mjs'
import { stageDesktopIdentityClosure } from './stage-desktop-identity.mjs'
const projectRoot = path.resolve(import.meta.dirname, '../../..')

const created = []
afterEach(() => {
  for (const root of created.splice(0)) fs.rmSync(root, { recursive: true, force: true })
})

function fixture(reference = '../product-identity.cjs') {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'semantier-desktop-identity-'))
  created.push(root)
  const source = path.join(root, 'source')
  const app = path.join(root, 'work', 'apps', 'desktop')
  const dist = path.join(app, 'dist')
  fs.mkdirSync(path.join(source, 'apps', 'desktop'), { recursive: true })
  fs.mkdirSync(path.join(source, 'scripts'), { recursive: true })
  fs.mkdirSync(path.join(dist, 'assets'), { recursive: true })
  fs.writeFileSync(path.join(dist, 'index.html'), '<script type="module" src="./assets/main.js"></script>')
  fs.writeFileSync(path.join(dist, 'assets', 'main.js'), 'export const ok = true')
  fs.writeFileSync(path.join(dist, 'electron-main.mjs'),
    'const id = typeof __HERMES_PRODUCT_IDENTITY__ === "undefined" ? require("' + reference + '") : __HERMES_PRODUCT_IDENTITY__')
  for (const name of ['electron-preload.js', 'preview-guest-preload.js']) fs.writeFileSync(path.join(dist, name), '')
  fs.writeFileSync(path.join(source, 'apps', 'desktop', 'product-identity.cjs'),
    "module.exports = require('../../scripts/msix-shared.mjs').identity()\n")
  fs.writeFileSync(path.join(source, 'scripts', 'msix-shared.mjs'),
    "import { createRequire } from 'node:module'; const require = createRequire(import.meta.url); const types = require('./release-content-types.json'); export function identity() { return types.name }\n")
  fs.writeFileSync(path.join(source, 'scripts', 'release-content-types.json'), '{"name":"Semantier Test"}\n')
  return { source, app, dist }
}

for (const reference of ['../product-identity.cjs', './product-identity.cjs']) {
  test('copies the identity closure for ' + reference + ' relative to final emitted main', () => {
    const { source, app, dist } = fixture(reference)
    assert.match(checkDesktopLaunch(dist).error, /development identity dependency failed/)
    const first = stageDesktopIdentityClosure({ source, app })
    assert.equal(first.copied.length, 3)
    assert.deepEqual(checkDesktopLaunch(dist), { ok: true })
    const root = path.resolve(dist, reference)
    assert.ok(fs.existsSync(root))
    assert.ok(fs.existsSync(path.resolve(path.dirname(root), '../../scripts/msix-shared.mjs')))
    assert.ok(fs.existsSync(path.resolve(path.dirname(root), '../../scripts/release-content-types.json')))
    assert.deepEqual(stageDesktopIdentityClosure({ source, app }).copied, first.copied)
  })
}

test('staged identity resolves with the actual Hermes source dependency closure', () => {
  const { app, dist } = fixture('../product-identity.cjs')
  const source = path.resolve(import.meta.dirname, '../../..')
  const result = stageDesktopIdentityClosure({ source, app })
  assert.equal(result.copied.length, 3)
  assert.deepEqual(checkDesktopLaunch(dist), { ok: true })
})

test('DesktopOnly automatically stages the identity closure before final launch', () => {
  const script = fs.readFileSync(path.join(projectRoot, 'scripts', 'deploy-ecs.ps1'), 'utf8')
  const copy = script.indexOf('stageDesktopIdentityClosure({source,app})')
  const check = script.indexOf("checkDesktopLaunch(path.join(app,'dist'))")
  assert.ok(copy > 0, 'automatic staging is required')
  assert.ok(check > copy, 'staging must precede final runtime-resolution check')
  assert.match(script, /identityClosure:identityClosure\.copied\.map/)
})

test('packaged-injected bundle needs no development identity copies', () => {
  const { source, app, dist } = fixture()
  fs.writeFileSync(path.join(dist, 'electron-main.mjs'), 'const id = { name: "baked identity" }')
  assert.deepEqual(stageDesktopIdentityClosure({ source, app }), { copied: [] })
  assert.deepEqual(checkDesktopLaunch(dist), { ok: true })
})

test('fails closed on an unknown relative identity specifier rather than launching a broken bundle', () => {
  const { source, app, dist } = fixture()
  fs.writeFileSync(path.join(dist, 'electron-main.mjs'),
    'const id = typeof __HERMES_PRODUCT_IDENTITY__ === "undefined" ? require("../../../product-identity.cjs") : __HERMES_PRODUCT_IDENTITY__')
  assert.throws(() => stageDesktopIdentityClosure({ source, app }), /Unsupported development identity path/)
})

test('missing source dependency stops staging before a broken app can run', () => {
  const { source, app } = fixture()
  fs.rmSync(path.join(source, 'scripts', 'release-content-types.json'))
  assert.throws(() => stageDesktopIdentityClosure({ source, app }), /Missing source dependency/)
})

