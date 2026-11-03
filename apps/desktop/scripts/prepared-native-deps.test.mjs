import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'
import * as native from './prepared-native-deps.mjs'
import beforePack from './before-pack.mjs'

test('beforePack refuses absent native preparation rather than staging from npm', async () => {
  const source = fs.mkdtempSync(path.join(os.tmpdir(), 'native-hook-'))
  try {
    await assert.rejects(beforePack({ appOutDir: '', electronPlatformName: 'linux', arch: 1,
      packager: { projectDir: path.join(source, 'apps/desktop') } }), /run preparation again/)
  } finally {
    fs.rmSync(source, { recursive: true, force: true })
  }
})

test('beforePack uses the external product directory without touching checkout dist', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'external-native-hook-'))
  const previous = { build: process.env.HERMES_DESKTOP_BUILD_DIR, native: process.env.HERMES_PREPARED_NATIVE_DEPS }
  try {
    const source = path.join(root, 'source')
    const app = path.join(source, 'apps/desktop')
    const nativeDeps = path.join(root, 'native-input')
    const build = path.join(root, 'job')
    fs.mkdirSync(path.join(nativeDeps, 'node-pty'), { recursive: true })
    fs.mkdirSync(path.join(app, 'dist/node_modules'), { recursive: true })
    fs.mkdirSync(build)
    fs.cpSync(path.join(import.meta.dirname, '../electron/native'), path.join(app, 'electron/native'), { recursive: true })
    fs.writeFileSync(path.join(source, 'package-lock.json'), '{}')
    fs.writeFileSync(path.join(app, 'package.json'), '{}')
    fs.writeFileSync(path.join(nativeDeps, 'node-pty/package.json'), '{}')
    fs.writeFileSync(path.join(app, 'dist/node_modules/preserve'), 'other owner')
    native.recordNativeInputs({ source, out: nativeDeps, platform: 'linux', arch: 'x64' })
    process.env.HERMES_DESKTOP_BUILD_DIR = build
    process.env.HERMES_PREPARED_NATIVE_DEPS = nativeDeps
    await beforePack({ appOutDir: '', electronPlatformName: 'linux', arch: 1, packager: { projectDir: app } })
    assert.equal(fs.readFileSync(path.join(app, 'dist/node_modules/preserve'), 'utf8'), 'other owner')
    assert.equal(fs.readFileSync(path.join(build, 'dist/node_modules/node-pty/package.json'), 'utf8'), '{}')
  } finally {
    for (const [name, value] of [['HERMES_DESKTOP_BUILD_DIR', previous.build], ['HERMES_PREPARED_NATIVE_DEPS', previous.native]]) {
      if (value === undefined) delete process.env[name]; else process.env[name] = value
    }
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('native consumption copies admitted modules and helpers, rejecting changed sources, targets or bytes', async () => {
  const source = fs.mkdtempSync(path.join(os.tmpdir(), 'prepared-native-'))
  try {
    const app = path.join(source, 'apps/desktop')
    const out = path.join(app, 'build/native-deps')
    fs.mkdirSync(path.join(out, 'node-pty'), { recursive: true })
    fs.cpSync(path.join(import.meta.dirname, '../electron/native'), path.join(app, 'electron/native'), { recursive: true })
    fs.writeFileSync(path.join(source, 'package-lock.json'), '{}')
    fs.writeFileSync(path.join(app, 'package.json'), '{}')
    fs.writeFileSync(path.join(out, 'node-pty/package.json'), '{}')
    const binding = path.join(out, 'node-pty/pty.node')
    fs.writeFileSync(binding, 'native fixture')
    const helper = 'native/linux-x64/hud-modifier-monitor'
    fs.mkdirSync(path.dirname(path.join(out, helper)), { recursive: true })
    fs.writeFileSync(path.join(out, helper), 'helper fixture', { mode: 0o755 })
    const selection = { source, nativeDeps: out, platform: 'linux', arch: 'x64', nativeToolchain: 'compiler-a' }
    const record = () => native.recordNativeInputs({ ...selection, out })
    record()
    fs.writeFileSync(path.join(app,'package.json'),JSON.stringify({version:'0.0.2',productName:'Semantier Agent 0.0.2'}))
    assert.equal(native.readNativeInputs(selection), out)
    assert.throws(() => native.readNativeInputs({ ...selection, nativeToolchain: 'compiler-b' }), /run preparation again/)
    await beforePack({ appOutDir: '', electronPlatformName: 'linux', arch: 1, packager: { projectDir: app } })
    const destination = path.join(app, 'dist/node_modules')
    const copiedHelper = path.join(app, 'dist', helper)
    assert.equal(fs.readFileSync(copiedHelper, 'utf8'), 'helper fixture')
    assert.equal(fs.existsSync(path.join(destination, 'native')), false)
    if (process.platform !== 'win32') assert.equal(fs.statSync(copiedHelper).mode & 0o777, 0o755)
    fs.writeFileSync(path.join(destination, 'node-pty/pty.node'), 'product mutation')
    assert.equal(fs.readFileSync(binding, 'utf8'), 'native fixture')
    assert.throws(() => native.readNativeInputs({ ...selection, arch: 'arm64' }), /run preparation again/)
    const header = path.join(app, 'electron/native/hud-modifier-gesture.h')
    const originalHeader = fs.readFileSync(header)
    fs.appendFileSync(header, '\n/* changed gesture */\n')
    assert.throws(() => native.readNativeInputs(selection), /run preparation again/)
    fs.writeFileSync(header, originalHeader)
    assert.equal(native.readNativeInputs(selection), out)
    fs.writeFileSync(path.join(out, helper), 'corrupt helper')
    assert.throws(() => native.copyNativeInputs({ ...selection, out: destination }), /run preparation again/)
    assert.equal(fs.readFileSync(copiedHelper, 'utf8'), 'helper fixture')
    fs.writeFileSync(path.join(out, helper), 'helper fixture')
    fs.writeFileSync(binding, 'corrupt')
    assert.throws(() => native.copyNativeInputs({ ...selection, out: destination }), /run preparation again/)
    assert.equal(fs.readFileSync(path.join(destination, 'node-pty/pty.node'), 'utf8'), 'product mutation')
    // A subsequent admitted preparation without optional helpers removes old ones.
    fs.rmSync(path.join(out, 'native'), { recursive: true })
    record()
    native.copyNativeInputs({ ...selection, out: destination })
    assert.equal(fs.existsSync(path.join(app, 'dist/native')), false)
  } finally {
    fs.rmSync(source, { recursive: true, force: true })
  }
})
