import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { expect, test } from 'vitest'

// The freshness contract for OS file-manager metadata (#122632, #122803): writing
// .DS_Store / AppleDouble sidecars / .localized / Thumbs.db / Desktop.ini into any
// hashed tree must never flip a recorded product from current to stale. Exercises
// the real buildInputs/recordProduct/productCurrent path against a fixture tree
// instead of a full compiler run, so the regression shows up in seconds.
const fixture = (name) => {
  const dir = mkdtempSync(path.join(tmpdir(), name))
  mkdirSync(path.join(dir, 'web/src'), { recursive: true })
  writeFileSync(path.join(dir, 'package.json'), '{"private":true}')
  writeFileSync(path.join(dir, 'package-lock.json'), '{}')
  writeFileSync(path.join(dir, 'web/src/main.ts'), 'export const answer: string = "prepared web";')
  return dir
}

test('desktop freshness tracks delivered code rather than tests, documentation and acceptance tools', async () => {
  const { buildInputs, recordProduct, productCurrent } = await import('../scripts/build/freshness.mjs')
  const source = fixture('desktop-freshness-')
  const out = mkdtempSync(path.join(tmpdir(), 'desktop-freshness-out-'))
  const put = (name, text) => {
    mkdirSync(path.dirname(path.join(source, name)), { recursive: true })
    writeFileSync(path.join(source, name), text)
  }
  const inert = [
    'apps/desktop/electron/updater/native.test.ts',
    'apps/desktop/scripts/native-package-update.windows.test.ps1',
    'apps/desktop/scripts/semantier-native-package.test.cjs',
    'apps/desktop/src/__tests__/fixture.ts',
    'apps/desktop/e2e/update.e2e.mts',
    'apps/desktop/scripts/perf/probe-model-picker.mjs',
    'apps/desktop/ENGINEERING.md',
    'apps/desktop/docs/qualification.md',
    'scripts/build/desktop.test.mjs',
    'tests/install/e2e-assets/accept-desktop-profile-windows.mjs',
    'scripts/deploy-ecs.ps1',
  ]
  const delivered = [
    'apps/desktop/src/main.tsx',
    'apps/desktop/electron/main.ts',
    'apps/desktop/scripts/native-package-update.ps1',
    'apps/desktop/scripts/native-package-digest.ps1',
    'apps/desktop/scripts/update-relaunch-waiter.ps1',
    'apps/desktop/scripts/before-pack.mjs',
    'apps/desktop/electron-builder.config.cjs',
    'apps/desktop/public/third-party/README.md',
    'scripts/build/desktop.mjs',
  ]
  try {
    for (const name of [...inert, ...delivered]) put(name, 'original')
    const inputs = buildInputs(source, 'desktop')
    recordProduct({ source, product: 'desktop', out, inputs })
    for (const name of inert) {
      put(name, 'diagnostic or test change')
      expect(productCurrent({ source, product: 'desktop', out }), name).toBe(true)
    }
    put('apps/desktop/electron/new-regression.test.ts', 'new test')
    expect(productCurrent({ source, product: 'desktop', out })).toBe(true)
    for (const name of delivered) {
      put(name, 'production change')
      expect(productCurrent({ source, product: 'desktop', out }), name).toBe(false)
      put(name, 'original')
      expect(productCurrent({ source, product: 'desktop', out }), name).toBe(true)
    }
  } finally {
    rmSync(source, { recursive: true, force: true })
    rmSync(out, { recursive: true, force: true })
  }
})

test('OS file-manager metadata landing in hashed trees never invalidates a recorded product', async () => {
  const { buildInputs, recordProduct, productCurrent } = await import('../scripts/build/freshness.mjs')
  const source = fixture('freshness-metadata-')
  const out = mkdtempSync(path.join(tmpdir(), 'freshness-out-'))
  try {
    mkdirSync(path.join(out, 'dist'), { recursive: true })
    const inputs = buildInputs(source, 'web')
    recordProduct({ source, product: 'web', out, inputs })
    expect(productCurrent({ source, product: 'web', out })).toBe(true)
    // A real source change must still invalidate — the guard stays fail-closed.
    writeFileSync(path.join(source, 'web/src/main.ts'), 'export const answer: string = "changed";')
    expect(productCurrent({ source, product: 'web', out })).toBe(false)
    writeFileSync(path.join(source, 'web/src/main.ts'), 'export const answer: string = "prepared web";')
    expect(productCurrent({ source, product: 'web', out })).toBe(true)
    // macOS Finder noise (#122632) and Windows file-manager noise (#122803) in
    // every hashed tree: source inputs, the output, prepared directories.
    for (const [name, content] of [
      ['.DS_Store', 'finder metadata'],
      ['web/src/._main.ts', 'apple double sidecar'],
      ['web/.localized', ''],
      ['web/Thumbs.db', 'thumb cache'],
      ['web/Desktop.ini', 'folder view settings'],
      ['._.DS_Store', 'double finder metadata'],
    ]) {
      writeFileSync(path.join(source, name), content)
      expect(productCurrent({ source, product: 'web', out }), name).toBe(true)
    }
    // Same noise inside the output tree and a prepared tree must not flip it either.
    writeFileSync(path.join(out, 'dist/.DS_Store'), 'finder metadata')
    expect(productCurrent({ source, product: 'web', out })).toBe(true)
    const prepared = mkdtempSync(path.join(tmpdir(), 'freshness-prepared-'))
    writeFileSync(path.join(prepared, 'icon.ico'), 'icon')
    writeFileSync(path.join(prepared, 'Thumbs.db'), 'thumb cache')
    const withPrepared = buildInputs(source, 'web', { icons: prepared })
    recordProduct({ source, product: 'web', out, inputs: withPrepared })
    writeFileSync(path.join(prepared, '.DS_Store'), 'finder metadata')
    expect(productCurrent({ source, product: 'web', out, prepared: { icons: prepared } })).toBe(true)
    // ...while a genuine prepared-input change still invalidates.
    writeFileSync(path.join(prepared, 'icon.ico'), 'changed icon')
    expect(productCurrent({ source, product: 'web', out, prepared: { icons: prepared } })).toBe(false)
  } finally {
    rmSync(source, { recursive: true, force: true })
    rmSync(out, { recursive: true, force: true })
  }
})
