// Stage the source-relative product identity closure used by non-injected
// Desktop UI check bundles. A packaged bundle has its identity injected
// and must not carry these development-only files.
import { createHash } from 'node:crypto'
import { copyFileSync, existsSync, lstatSync, mkdirSync, readFileSync } from 'node:fs'
import path from 'node:path'

const IDENTITY_REFERENCE = /["'](\.\.?\/product-identity\.cjs)["']/g

function sourceFile(source, relative) {
  const file = path.join(source, relative)
  if (!existsSync(file) || !lstatSync(file).isFile() || lstatSync(file).isSymbolicLink()) {
    throw new Error(`Missing source dependency: ${relative}`)
  }
  return file
}

function copyBoundFile(source, relative, destination, copied) {
  const input = sourceFile(source, relative)
  mkdirSync(path.dirname(destination), { recursive: true })
  // Repeat runs are deterministic and do not need to rewrite already-matching files.
  const digest = value => createHash('sha256').update(value).digest('hex')
  const expected = digest(readFileSync(input))
  if (!existsSync(destination) || digest(readFileSync(destination)) !== expected) {
    copyFileSync(input, destination)
  }
  if (digest(readFileSync(destination)) !== expected) {
    throw new Error(`Staged dependency differs from source: ${relative}`)
  }
  copied.push(destination)
}

/**
 * Copy only the discovered development identity dependency chain, relative
 * to the *final* emitted electron-main.mjs. No user flags or manual staging.
 * Runtime Electron/node-pty/get-windows assets remain owned by the existing
 * packaging toolchain; this function does not pretend to qualify them.
 */
export function stageDesktopIdentityClosure({ source, app }) {
  const main = path.join(app, 'dist', 'electron-main.mjs')
  if (!existsSync(main)) {
    throw new Error('Desktop electron-main.mjs is missing from the final staging directory')
  }
  const code = readFileSync(main, 'utf8')
  if (!code.includes('typeof __HERMES_PRODUCT_IDENTITY__')) {
    return { copied: [] }
  }
  const references = [...new Set([...code.matchAll(IDENTITY_REFERENCE)].map(match => match[1]))]
  if (references.length !== 1 || !['./product-identity.cjs', '../product-identity.cjs'].includes(references[0])) {
    throw new Error('Unsupported development identity path in staged Electron main')
  }

  // Guard the documented transitive closure: a future added import must be
  // inspected and admitted rather than silently copied from arbitrary paths.
  const identitySource = sourceFile(source, 'apps/desktop/product-identity.cjs')
  const bridgeSource = sourceFile(source, 'scripts/msix-shared.mjs')
  if (!/require\(['"]\.\.\/\.\.\/scripts\/msix-shared\.mjs['"]\)/.test(readFileSync(identitySource, 'utf8')) ||
      !/require\(['"]\.\/release-content-types\.json['"]\)/.test(readFileSync(bridgeSource, 'utf8'))) {
    throw new Error('Desktop identity source dependencies changed; inspect the transitive staging closure')
  }

  const identity = path.resolve(path.dirname(main), references[0])
  const bridge = path.resolve(path.dirname(identity), '../../scripts/msix-shared.mjs')
  const types = path.resolve(path.dirname(bridge), 'release-content-types.json')
  const copied = []
  copyBoundFile(source, 'apps/desktop/product-identity.cjs', identity, copied)
  copyBoundFile(source, 'scripts/msix-shared.mjs', bridge, copied)
  copyBoundFile(source, 'scripts/release-content-types.json', types, copied)
  return { copied }
}
