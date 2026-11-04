// Hermes shares %LOCALAPPDATA% and HKCU state with the unpackaged CLI/gateway,
// so every MSIX the repo produces must opt out of write virtualisation and the
// builder must request the restricted capability that permits it.
import { execFileSync, spawnSync } from 'node:child_process'
import fs from 'node:fs'
import { createRequire } from 'node:module'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, expect, test } from 'vitest'
import { stageChannelManifest } from '../../../scripts/msix-shared.mjs'
import { stageReleaseManifest, stageStoreManifest } from './before-build.mjs'

const require = createRequire(import.meta.url)
const desktop = fileURLToPath(new URL('../', import.meta.url))
const roots = []
afterEach(() => { for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true }) })

function virtualization(xml) {
  const read = name => new RegExp(`<desktop6:${name}>([^<]*)</desktop6:${name}>`).exec(xml)?.[1]
  return { fileSystem: read('FileSystemWriteVirtualization'), registry: read('RegistryWriteVirtualization') }
}

// A tagged app dir: the real template plus just enough for appIdentity.
function taggedApp(tag) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'msix-virt-'))
  roots.push(root)
  const app = path.join(root, 'apps', 'desktop')
  fs.mkdirSync(path.join(app, 'assets'), { recursive: true })
  fs.copyFileSync(path.join(desktop, 'assets/msix-manifest.xml'), path.join(app, 'assets/msix-manifest.xml'))
  fs.writeFileSync(path.join(app, 'product-identity.cjs'), "module.exports={store:true,artifactNamePascal:'HermesBundled'}\n")
  fs.writeFileSync(path.join(app, 'package.json'), JSON.stringify({ name: 'hermes', version: '0.27.1' }))
  const env = { ...process.env, GIT_AUTHOR_NAME: 'Test', GIT_AUTHOR_EMAIL: 'test@example.invalid',
    GIT_COMMITTER_NAME: 'Test', GIT_COMMITTER_EMAIL: 'test@example.invalid',
    GIT_AUTHOR_DATE: '2026-09-07T00:18:00Z', GIT_COMMITTER_DATE: '2026-09-07T00:18:00Z' }
  for (const args of [['init', '-q'], ['add', '.'], ['-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture'], ['tag', tag]]) {
    execFileSync('git', args, { cwd: root, env, stdio: 'pipe' })
  }
  return app
}

// A channel app dir: stageChannelManifest loads scripts/before-build.mjs from it,
// so link the real scripts directory (a junction needs no privilege on Windows).
function channelApp() {
  const app = fs.mkdtempSync(path.join(os.tmpdir(), 'msix-virt-channel-'))
  roots.push(app)
  fs.mkdirSync(path.join(app, 'assets'))
  fs.mkdirSync(path.join(app, 'build/agent-payload'), { recursive: true })
  fs.copyFileSync(path.join(desktop, 'assets/msix-manifest.xml'), path.join(app, 'assets/msix-manifest.xml'))
  fs.writeFileSync(path.join(app, 'build/agent-payload/manifest.json'), JSON.stringify({ launchers: ['hermes'] }))
  fs.writeFileSync(path.join(app, 'build/msix-extensions.xml'), '<Extensions />')
  fs.symlinkSync(path.join(desktop, 'scripts'), path.join(app, 'scripts'), 'junction')
  return app
}

test('store, release, canary and channel MSIX manifests keep write virtualisation disabled', () => {
  const store = taggedApp('v0.27.1')
  const canary = taggedApp('v0.27.1+canary.20260907T001800Z')
  fs.mkdirSync(path.join(canary, 'build'), { recursive: true })
  fs.copyFileSync(path.join(canary, 'assets/msix-manifest.xml'), path.join(canary, 'build/msix-manifest.xml'))
  const channel = channelApp()
  stageChannelManifest(channel, {
    windowsVersion: '0.0.7.0', identity: { appNamePascal: 'HermesChannelab12', displayName: 'Hermes preview' }
  })
  const manifests = {
    store: stageStoreManifest(store, 'v0.27.1'),
    release: stageReleaseManifest(taggedApp('v0.27.1'), 'v0.27.1'),
    canary: stageReleaseManifest(canary, 'v0.27.1+canary.20260907T001800Z'),
    channel: path.join(channel, 'build/msix-manifest.xml')
  }
  for (const [kind, file] of Object.entries(manifests)) {
    expect([kind, virtualization(fs.readFileSync(file, 'utf8'))]).toEqual([kind, { fileSystem: 'disabled', registry: 'disabled' }])
  }
})

test('the builder config requests unvirtualizedResources for MSIX packages', () => {
  const config = require('../electron-builder.config.cjs')
  expect(config.msix.capabilities).toContain('unvirtualizedResources')
})

test('MSIX manifest staging accepts an external owned build directory', () => {
  const app = taggedApp('v0.27.1')
  const external = fs.mkdtempSync(path.join(os.tmpdir(), 'msix-owned-build-'))
  roots.push(external)
  const result = stageStoreManifest(app, 'v0.27.1', external)
  expect(result).toBe(path.join(external, 'store-msix-manifest.xml'))
  expect(virtualization(fs.readFileSync(result, 'utf8'))).toEqual({ fileSystem: 'disabled', registry: 'disabled' })
  expect(fs.existsSync(path.join(app, 'build'))).toBe(false)
  const release = stageReleaseManifest(app, 'v0.27.1', external)
  expect(release).toBe(path.join(external, 'msix-manifest.xml'))
  expect(virtualization(fs.readFileSync(release, 'utf8'))).toEqual({ fileSystem: 'disabled', registry: 'disabled' })
  expect(fs.existsSync(path.join(app, 'build'))).toBe(false)
})

test('channel MSIX manifest and extensions stay in the selected build directory', () => {
  const app = channelApp()
  const external = fs.mkdtempSync(path.join(os.tmpdir(), 'msix-channel-build-'))
  roots.push(external)
  fs.cpSync(path.join(app, 'build'), external, { recursive: true })
  fs.rmSync(path.join(app, 'build'), { recursive: true })
  stageChannelManifest(app, {
    windowsVersion: '0.0.7.0', identity: { appNamePascal: 'HermesChannelab12', displayName: 'Hermes preview' }
  }, external)
  expect(virtualization(fs.readFileSync(path.join(external, 'msix-manifest.xml'), 'utf8')))
    .toEqual({ fileSystem: 'disabled', registry: 'disabled' })
  expect(fs.existsSync(path.join(app, 'build'))).toBe(false)
})


test('before-build writes MSIX assets and extensions to the selected job directory', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'msix-external-hook-'))
  roots.push(root)
  const app = path.join(root, 'apps/desktop')
  const scripts = path.join(app, 'scripts')
  const external = path.join(root, 'job/build')
  fs.mkdirSync(scripts, { recursive: true })
  fs.mkdirSync(path.join(root, 'scripts'), { recursive: true })
  fs.cpSync(path.join(desktop, 'assets'), path.join(app, 'assets'), { recursive: true })
  fs.copyFileSync(path.join(desktop, 'product-identity.cjs'), path.join(app, 'product-identity.cjs'))
  fs.copyFileSync(path.join(desktop, 'scripts/before-build.mjs'), path.join(scripts, 'before-build.mjs'))
  for (const file of ['msix-shared.mjs', 'release-content-types.json']) {
    fs.copyFileSync(path.resolve(desktop, '../../scripts', file), path.join(root, 'scripts', file))
  }
  fs.cpSync(path.join(app, 'assets'), path.join(external, 'assets'), { recursive: true })
  fs.mkdirSync(path.join(external, 'agent-payload'), { recursive: true })
  fs.writeFileSync(path.join(external, 'agent-payload/manifest.json'), JSON.stringify({ launchers: ['hermes'] }))
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) => !/^(HERMES_|SEMANTIER_|GITHUB_)/.test(key)))
  Object.assign(env, { HERMES_DESKTOP_VARIANT: 'bundled', HERMES_DESKTOP_BUILD_DIR: external })
  const run = spawnSync(process.execPath, ['--input-type=module', '-e',
    `import { pathToFileURL } from 'node:url'; const m = await import(pathToFileURL(process.argv[1])); await m.default()` ,
    path.join(scripts, 'before-build.mjs')], { env, encoding: 'utf8' })
  expect(run.status, run.stderr).toBe(0)
  expect(fs.existsSync(path.join(external, 'appx/Square44x44Logo.png'))).toBe(true)
  expect(fs.readFileSync(path.join(external, 'msix-extensions.xml'), 'utf8')).toContain('hermes.exe')
  expect(fs.existsSync(path.join(app, 'build'))).toBe(false)
})

test('builder configuration consumes the selected external build directory', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'msix-external-config-'))
  roots.push(root)
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) => !/^(HERMES_|SEMANTIER_|GITHUB_)/.test(key)))
  Object.assign(env, { HERMES_DESKTOP_VARIANT: 'bundled', HERMES_DESKTOP_BUILD_DIR: root })
  const code = `const c=require(process.argv[1]);console.log(JSON.stringify({resources:c.directories.buildResources,extra:c.extraResources,extensions:c.msix.customExtensionsPath,files:c.files,icon:c.icon,unpack:c.asar.unpack}))`
  const run = spawnSync(process.execPath, ['-e', code, path.join(desktop, 'electron-builder.config.cjs')], { env, encoding: 'utf8' })
  expect(run.status, run.stderr).toBe(0)
  const actual = JSON.parse(run.stdout)
  expect(actual.resources).toBe(root)
  expect(actual.icon).toBe(path.join(root, 'assets/icon'))
  expect(actual.files).toContainEqual({ from: path.join(root, 'assets'), to: 'assets', filter: ['**/*'] })
  expect(actual.extra.map(item => item.from)).toContain(path.join(root, 'agent-payload'))
  expect(actual.extra.map(item => item.from)).toContain(path.join(root, 'install-stamp.json'))
  expect(actual.extensions).toBe(path.join(root, 'msix-extensions.xml'))
  expect(actual.files).toContainEqual({ from: path.join(root, 'dist'), to: 'dist', filter: ['**/*'] })
  for (const name of ['node-pty', 'get-windows']) {
    expect(actual.files).toContainEqual({ from: path.join(root, 'dist/node_modules', name), to: `dist/node_modules/${name}`, filter: ['**/*'] })
    expect(actual.unpack).toContain(`**/node_modules/${name}/**`)
  }
})
