// Four complete Windows Electron process launches, with production IPC/preload/store.
import { build } from 'esbuild'
import { mkdtempSync, rmSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { resolve, join } from 'node:path'
import { spawnSync } from 'node:child_process'
const require = createRequire(import.meta.url)
const electron = require('electron')
const temporary = mkdtempSync(join(tmpdir(), 'hermes-branding-native-'))
try {
  for (const name of ['main', 'preload']) await build({
    entryPoints: [resolve(`electron/desktop-branding-native-fixture/${name}.ts`)],
    bundle: true, platform: 'node', format: 'cjs', external: ['electron'], outfile: join(temporary, `${name}.cjs`)
  })
  for (const mode of ['commit', 'remove', 'reset', 'verify']) {
    const result = spawnSync(electron, [join(temporary, 'main.cjs'), join(temporary, 'user-data'), mode], { stdio: 'inherit', timeout: 40000 })
    if (result.error || result.status !== 0) throw result.error ?? new Error(`Native ${mode} failed: ${result.status}`)
  }
} finally { rmSync(temporary, { recursive: true, force: true }) }
