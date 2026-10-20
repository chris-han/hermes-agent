import { dependencyContents } from '../../../scripts/build/node-deps.mjs'
import { resolveWindowsFrameworkCompiler } from './build-hud-modifier-monitor.mjs'
import fs from 'node:fs'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { fileDigest, treeDigest, preparationRequired } from './prepared-packaging.mjs'

/** @typedef {{ source: string, nativeDeps: string, platform?: string, arch?: string, nativeToolchain?: string }} NativeSelection */
/** @param {string} source @returns {string} */
function nativeIdentity(source) {
  return createHash('sha256').update(JSON.stringify([
    dependencyContents(fs.readFileSync(path.join(source,'package-lock.json')),true),
    dependencyContents(fs.readFileSync(path.join(source,'apps/desktop/package.json'))),
    process.versions.node,process.versions.modules,
    process.platform === 'win32' ? fileDigest(resolveWindowsFrameworkCompiler()) : null,
    fileDigest(path.join(import.meta.dirname, 'stage-native-deps.mjs')),
    fileDigest(path.join(import.meta.dirname, 'prepared-native-deps.mjs')),
    ...['build-command-screenshot-monitor.mjs', 'build-hud-modifier-monitor.mjs']
      .map(name => fileDigest(path.join(import.meta.dirname, name))),
    ...['command-screenshot-monitor.m', 'hud-modifier-gesture.h', 'hud-modifier-gesture.cs',
      'hud-modifier-monitor.m', 'hud-modifier-monitor-win.cs', 'hud-modifier-monitor-x11.c']
      .map(name => fileDigest(path.join(source, 'apps/desktop/electron/native', name))),
  ])).digest('hex')
}

/**
 * The sidecar stays outside node_modules so it never ships in the application.
 * ``degraded`` records a soft-failed component (no get-windows, no X11 toolchain): the tree is
 * consumable, but the identity cannot see the host fix that would complete it.
 * @param {{ source: string, out: string, platform: string, arch: string, nativeToolchain?: string, degraded?: boolean }} inputs
 * @returns {void}
 */
export function recordNativeInputs({ source, out, platform, arch, nativeToolchain, degraded = false }) {
  fs.writeFileSync(`${out}.prepared.json`, JSON.stringify({
    schema: 1, source: fs.realpathSync(source), out: fs.realpathSync(out),
    platform, arch, nativeToolchain, degraded, identity: nativeIdentity(source), digest: treeDigest(out),
  }) + '\n')
}

/** A local rebuild may skip restaging: the receipt is admitted and nothing in it soft-failed.
 * @param {NativeSelection} inputs @returns {boolean} */
export function nativeInputsReason(inputs) {
  try {
    readNativeInputs(inputs)
    return JSON.parse(fs.readFileSync(`${inputs.nativeDeps}.prepared.json`, 'utf8')).degraded === false ? null : 'degraded-native-tree'
  } catch (error) { return error.cacheReason || 'missing-or-unreadable-native-receipt' }
}

export function nativeInputsComplete(inputs) { return nativeInputsReason(inputs) === null }

/** @param {NativeSelection} inputs @returns {string} */
export function readNativeInputs({ source, nativeDeps, platform = process.platform, arch = process.arch, nativeToolchain }) {
  try {
    const record = JSON.parse(fs.readFileSync(`${nativeDeps}.prepared.json`, 'utf8'))
    const reason = [
      [record.schema !== 1, 'native-receipt-schema'],
      [record.source !== fs.realpathSync(source) || record.out !== fs.realpathSync(nativeDeps), 'native-source-or-output-path'],
      [record.platform !== platform || record.arch !== arch, 'native-platform-or-architecture'],
      [nativeToolchain !== undefined && record.nativeToolchain !== nativeToolchain, 'native-toolchain'],
      [record.identity !== nativeIdentity(source), 'native-dependencies-abi-or-recipe'],
      [record.digest !== treeDigest(nativeDeps), 'native-artifact-digest'],
    ].find(([changed]) => changed)?.[1]
    if (reason) {const error=preparationRequired('Stale native inputs: '+reason);error.cacheReason=reason;throw error}
    if (!fs.statSync(path.join(nativeDeps, 'node-pty/package.json')).isFile()) throw preparationRequired('Missing prepared node-pty')
    return record.out
  } catch (error) {
    const rejected=preparationRequired(`Cannot consume native inputs: ${error instanceof Error ? error.message : String(error)}`)
    rejected.cacheReason=error.cacheReason || 'missing-or-unreadable-native-inputs'
    throw rejected
  }
}

/** @param {NativeSelection & { out: string }} inputs @returns {void} */
export function copyNativeInputs({ out, ...inputs }) {
  copyNativeTree({ nativeDeps: readNativeInputs(inputs), out })
}

/** Copy admitted modules and executable resources without rebuilding either.
 * @param {{ nativeDeps: string, out: string }} inputs out is the product's node_modules.
 * @returns {void}
 */
export function copyNativeTree({ nativeDeps, out }) {
  nativeDeps = fs.realpathSync(nativeDeps)
  const destination = path.resolve(out)
  const helpers = path.join(path.dirname(destination), 'native')
  for (const target of [destination, helpers]) {
    if (target === nativeDeps || target.startsWith(nativeDeps + path.sep) || nativeDeps.startsWith(target + path.sep)) {
      throw preparationRequired('Native input and product directories overlap')
    }
  }
  fs.rmSync(destination, { recursive: true, force: true })
  fs.rmSync(helpers, { recursive: true, force: true })
  const preparedHelpers = path.join(nativeDeps, 'native')
  fs.cpSync(nativeDeps, destination, { recursive: true, dereference: true,
    filter: file => file !== preparedHelpers })
  if (fs.existsSync(preparedHelpers)) fs.cpSync(preparedHelpers, helpers, { recursive: true, dereference: true })
}
