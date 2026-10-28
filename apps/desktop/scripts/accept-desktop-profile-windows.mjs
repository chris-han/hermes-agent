// Real packaged Windows profile/channel regression; never touches normal user roots.
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import crypto from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { _electron, expect } from '@playwright/test'

const [bindingFile, outputArg] = process.argv.slice(2)
assert.equal(process.platform, 'win32')
const binding = JSON.parse(fs.readFileSync(bindingFile, 'utf8'))
const digest = file => {
  const hash = crypto.createHash('sha256'), buffer = Buffer.alloc(8 * 1024 * 1024)
  const fd = fs.openSync(file, 'r')
  try { let n; while ((n = fs.readSync(fd, buffer, 0, buffer.length, null))) hash.update(buffer.subarray(0, n)) }
  finally { fs.closeSync(fd) }
  return hash.digest('hex')
}
for (const name of ['executable', 'asar', 'msix']) assert.equal(digest(binding[name].path), binding[name].sha256)
assert.match(binding.source, /^[a-f0-9]{40}$/)
assert.match(binding.governing_plan_sha256, /^[a-f0-9]{64}$/)
assert.match(binding.candidate_inputs_sha256, /^[a-f0-9]{64}$/)
const output = path.resolve(outputArg)
assert.ok(!fs.existsSync(output), 'new owned output required')
fs.mkdirSync(output, { recursive: true })
fs.writeFileSync(path.join(output, 'owner.json'), JSON.stringify({ owner: 'windows-update-profile-regression-v1', source: binding.source }))
const results = {}, observations = [], errors = [], identities = []
let app, page
const baseEnv = Object.fromEntries(Object.entries(process.env).filter(([key]) => /^(PATH|SYSTEMROOT|WINDIR|COMSPEC|PATHEXT|USERPROFILE|SYSTEMDRIVE|PROCESSOR_ARCHITECTURE|NUMBER_OF_PROCESSORS)$/i.test(key)))
// Acceptance-family bundle clears ambient HERMES_HOME; the existing production
// resolver scopes its effective root beneath the explicit userData directory.
const profileHome = root => path.join(root, 'userdata', 'hermes-home')
function environment(root) {
  assert.ok(path.resolve(root).startsWith(output + path.sep))
  const dirs = Object.fromEntries(['home', 'appdata', 'localappdata', 'temp', 'userdata'].map(name => [name, path.join(root, name)]))
  dirs.appdata = path.join(dirs.home, 'AppData', 'Roaming')
  dirs.localappdata = path.join(dirs.home, 'AppData', 'Local')
  dirs.hermes = profileHome(root)
  for (const dir of Object.values(dirs)) fs.mkdirSync(dir, { recursive: true })
  return { ...baseEnv, HOME: dirs.home, USERPROFILE: dirs.home, APPDATA: dirs.appdata, LOCALAPPDATA: dirs.localappdata, TEMP: dirs.temp, TMP: dirs.temp,
    HERMES_HOME: dirs.hermes, HERMES_DESKTOP_USER_DATA_DIR: dirs.userdata,
    HERMES_DESKTOP_SKIP_QUIT_CONFIRM: '1', HERMES_DESKTOP_SKIP_PROTOCOL_REGISTRATION: '1', HERMES_DESKTOP_CDP_PORT: 'off',
    ELECTRON_ENABLE_LOGGING: '1', ELECTRON_ENABLE_STACK_DUMPING: '1', ELECTRON_LOG_FILE: path.join(dirs.temp, 'electron.log'),
    HTTP_PROXY: 'http://127.0.0.1:9', HTTPS_PROXY: 'http://127.0.0.1:9', ALL_PROXY: 'http://127.0.0.1:9', NO_PROXY: '127.0.0.1,localhost,::1' }
}
async function close() { if (app) { await app.close(); app = null; page = null } }
async function launch(root, label, executable = binding.executable.path) {
  app = await _electron.launch({ executablePath: executable, cwd: path.dirname(executable), env: environment(root), timeout: 120000 })
  const identity = await app.evaluate(({ app }) => ({ platform: process.platform, packaged: app.isPackaged, version: app.getVersion(), executable: process.execPath, userData: app.getPath('userData'), home: app.getPath('home'), locale: app.getLocale() }))
  assert.equal(identity.platform, 'win32'); assert.equal(identity.packaged, true)
  assert.equal(path.resolve(identity.executable), path.resolve(executable))
  assert.equal(path.resolve(identity.userData), path.resolve(root, 'userdata'))
  if (executable === binding.executable.path) assert.equal(identity.version, binding.version)
  identities.push({ ...identity, label, pid: app.process().pid })
  page = await app.firstWindow({ timeout: 120000 })
  page.on('pageerror', error => errors.push(error.message))
  await expect(page.locator('html')).toHaveAttribute('data-hermes-theme', /.+/, { timeout: 120000 })
  await expect.poll(() => page.evaluate(async () => (await window.hermesDesktop.api({ path: '/api/profiles/active' })).current), { timeout: 120000 }).toBe('semantier-local')
  await page.screenshot({ path: path.join(output, label + '.png'), fullPage: true })
  return identity
}
async function roster() {
  return page.evaluate(async () => ({
    names: (await window.hermesDesktop.api({ path: '/api/profiles' })).profiles.map(p => p.name).sort(),
    current: (await window.hermesDesktop.api({ path: '/api/profiles/active' })).current,
    native: await window.hermesDesktop.profile.get()
  }))
}
async function assertLocal(root, label, locale = 'zh') {
  await expect(page.locator('html')).toHaveAttribute('lang', locale, { timeout: 120000 })
  await expect(page.locator('html')).toHaveAttribute('data-hermes-theme', 'semantier')
  const state = await roster()
  assert.deepEqual(state.names, ['semantier-local']); assert.equal(state.current, 'semantier-local')
  const preference = JSON.parse(fs.readFileSync(path.join(root, 'userdata/active-profile.json'), 'utf8'))
  assert.equal(preference.profile, 'semantier-local'); assert.equal(preference.defaultRoute.profile, 'semantier-local')
  const named = fs.readdirSync(path.join(profileHome(root), 'profiles'), { withFileTypes: true }).filter(e => e.isDirectory() && !e.name.startsWith('.')).map(e => e.name).sort()
  assert.deepEqual(named, ['semantier-local'])
  assert.ok(!fs.existsSync(path.join(profileHome(root), 'eos.db')))
  observations.push({ label, state, preference, named })
}
function snapshot(root) {
  const files = {}
  function visit(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name)
      assert.ok(!entry.isSymbolicLink(), 'fixture has no links')
      if (entry.isDirectory()) visit(full)
      else files[path.relative(root, full)] = digest(full)
    }
  }
  visit(root); return files
}
function seedLegacy(root, withLocal) {
  environment(root)
  const legacy = path.join(profileHome(root), 'profiles/tender-runtime-expert')
  for (const dir of ['sessions', 'memories', 'plugins']) fs.mkdirSync(path.join(legacy, dir), { recursive: true })
  fs.writeFileSync(path.join(legacy, 'config.yaml'), 'display:\n  language: en\nmodel:\n  provider: custom\n  default: retained-fixture\nplugins:\n  enabled: [tender_contextgraph_runtime]\n')
  fs.writeFileSync(path.join(legacy, '.env'), 'CUSTOM_API_KEY=isolated-noncredential-fixture\n')
  fs.writeFileSync(path.join(legacy, 'memories/MEMORY.md'), 'retained memory 中文\n')
  fs.writeFileSync(path.join(legacy, 'PROFILE.md'), '# tender-runtime-expert\nGenerated internal profile fixture\n')
  fs.writeFileSync(path.join(legacy, 'sessions/retained.json'), JSON.stringify({ messages: [{ role: 'user', content: 'Retain this chat 中文' }] }))
  // The package's own interpreter and SessionDB create a genuine persisted chat.
  const payload = path.join(path.dirname(binding.executable.path), 'resources/agent-payload')
  const manifest = JSON.parse(fs.readFileSync(path.join(payload, 'manifest.json'), 'utf8'))
  const runtime = manifest.runtime
  const python = path.join(payload, runtime.storePython)
  const repo = path.join(payload, runtime.repoDir)
  const code = "from pathlib import Path; import sys; from hermes_state import SessionDB; db=SessionDB(Path(sys.argv[1])/'state.db'); db.create_session('profile-preservation', 'desktop'); db.append_message('profile-preservation','user','Retain this chat 中文'); db.close()"
  execFileSync(python, ['-P', '-c', code, legacy], { env: { ...environment(root), PYTHONPATH: [repo, path.join(payload, runtime.sitePackages)].join(path.delimiter), PYTHONIOENCODING: 'utf-8' }, cwd: repo, windowsHide: true, timeout: 120000 })
  if (withLocal) {
    const local = path.join(profileHome(root), 'profiles/semantier-local'); fs.mkdirSync(local, { recursive: true })
    fs.writeFileSync(path.join(local, 'config.yaml'), 'display:\n  language: zh\nmodel:\n  default: keep-local-provider\n')
    fs.writeFileSync(path.join(local, '.env'), 'CUSTOM_API_KEY=local-fixture-preserve\n')
    fs.writeFileSync(path.join(local, 'SOUL.md'), 'Existing local user data\n')
  }
  fs.writeFileSync(path.join(profileHome(root), 'active_profile'), 'tender-runtime-expert\n')
  fs.writeFileSync(path.join(root, 'userdata/active-profile.json'), JSON.stringify({ profile: 'tender-runtime-expert', defaultRoute: { connectionId: null, profile: 'tender-runtime-expert' } }))
  return { legacy, files: snapshot(legacy), local: withLocal ? snapshot(path.join(profileHome(root), 'profiles/semantier-local')) : null }
}
async function language(value) {
  await page.evaluate(() => { location.hash = '/settings?tab=config:appearance&page=general' })
  await page.getByRole('button', { name: /Switch language|切换语言/i }).click()
  const option = page.locator('[cmdk-item][data-value="' + value + '"]')
  const bounds = await option.boundingBox(); assert.ok(bounds)
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2)
  await option.click()
  await expect(page.locator('html')).toHaveAttribute('lang', value)
  await expect.poll(() => page.evaluate(async () => {
    const config = await window.hermesDesktop.api({ path: '/api/config?include_defaults=false' })
    return config.display?.language
  })).toBe(value)
}
try {
  const fresh = path.join(output, 'fresh')
  const identity = await launch(fresh, 'fresh-launch')
  assert.match(identity.locale, /^en/i)
  await assertLocal(fresh, 'fresh-Chinese-Semantier')
  results.P01 = 'PASS'
  await language('en'); await close(); await launch(fresh, 'fresh-explicit-English-restart'); await assertLocal(fresh, 'fresh-English-restart', 'en')
  await language('zh'); await close(); await launch(fresh, 'fresh-explicit-Chinese-restart'); await assertLocal(fresh, 'fresh-Chinese-restart')
  await close()
  for (const withLocal of [false, true]) {
    const root = path.join(output, withLocal ? 'legacy-and-local' : 'legacy-only')
    const seeded = seedLegacy(root, withLocal)
    await launch(root, withLocal ? 'legacy-local-migrated' : 'legacy-only-migrated')
    await assertLocal(root, 'legacy-selection', withLocal ? 'zh' : 'en')
    const batches = fs.readdirSync(path.join(profileHome(root), '.desktop-retired-profiles'))
    assert.equal(batches.length, 1)
    const backup = path.join(profileHome(root), '.desktop-retired-profiles', batches[0], 'tender-runtime-expert')
    assert.deepEqual(snapshot(backup), seeded.files)
    assert.ok(fs.existsSync(path.join(profileHome(root), 'profiles/.deleted/tender-runtime-expert')))
    if (seeded.local) {
      for (const file of ['.env', 'SOUL.md']) assert.equal(digest(path.join(profileHome(root), 'profiles/semantier-local', file)), seeded.local[file])
      assert.ok(fs.readFileSync(path.join(profileHome(root), 'profiles/semantier-local/config.yaml'), 'utf8').includes('keep-local-provider'))
    }
    observations.push({ retainedBackup: backup, hashes: seeded.files, localHashes: seeded.local })
    await close(); await launch(root, withLocal ? 'legacy-local-full-restart' : 'legacy-full-restart')
    await assertLocal(root, 'retirement-restart', withLocal ? 'zh' : 'en')
    assert.deepEqual(snapshot(backup), seeded.files)
    await close()
  }
  results.P02 = 'PASS'; results.P04 = 'PASS'
  results.P03 = 'NOT_RUN: full restart passed; actual predecessor-to-candidate update requires the separate bound update driver'
  // Live sources are exercised through real packaged IPC and rendered controls.
  await launch(fresh, 'channels')
  await page.evaluate(() => { location.hash = '/settings?tab=about' })
  const selection = await page.evaluate(() => window.hermesDesktop.updates.getDeliverySources())
  assert.deepEqual(selection.options.map(p => p.id), ['channel1', 'channel2', 'channel3'])
  const sourceControl = page.locator('#update-delivery-source')
  await expect(sourceControl).toBeVisible({ timeout: 30000 })
  await sourceControl.click()
  for (const label of [/通道 1|Channel 1/, /通道 2|Channel 2/, /通道 3|Channel 3/]) await expect(page.getByRole('option', { name: label })).toBeVisible()
  await page.keyboard.press('Escape')
  const sources = []
  for (const id of ['channel1', 'channel2', 'channel3']) {
    const value = await page.evaluate(async id => {
      try { return { ok: true, result: await window.hermesDesktop.updates.setDeliverySource(id) } }
      catch (error) { return { ok: false, error: String(error) } }
    }, id)
    sources.push({ id, value })
    await page.screenshot({ path: path.join(output, id + '.png'), fullPage: true })
  }
  observations.push({ sourceSelection: selection, sources })
  results.P05 = sources.every(x => x.value.ok && !x.value.result.status.error) ? 'PASS' : 'NOT_RUN: one or more real source endpoints unavailable; inspect source results'
  await close()
  assert.deepEqual(errors, [])
} catch (error) {
  errors.push(String(error.stack || error))
  if (page) await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {})
  throw error
} finally {
  await close()
  fs.writeFileSync(path.join(output, 'acceptance.json'), JSON.stringify({ produced_at: new Date().toISOString(), binding, harness_sha256: digest(fileURLToPath(import.meta.url)), results, observations, identities, errors,
    result: errors.length ? 'FAIL' : Object.values(results).every(x => x === 'PASS') ? 'PASS' : 'PARTIAL',
    registered_installation: 'NOT_RUN', os_upgrade: 'NOT_RUN', human_usability: 'UNMEASURED', business_benefit: 'UNMEASURED' }, null, 2))
}
