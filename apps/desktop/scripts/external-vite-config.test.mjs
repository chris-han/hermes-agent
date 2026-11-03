import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { test } from 'vitest'

const source = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..')

test('external dependency resolution keeps the compiler and offline emoji assets', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'external-vite-config-'))
  try {
    const fixture = path.join(root, 'source')
    const app = path.join(fixture, 'apps/desktop')
    fs.mkdirSync(path.join(app, 'scripts'), { recursive: true })
    for (const relative of ['apps/desktop/vite.config.ts', 'apps/desktop/scripts/compiler-cache.mjs', 'package-lock.json']) {
      fs.copyFileSync(path.join(source, relative), path.join(fixture, relative))
    }
    fs.writeFileSync(path.join(app, 'index.html'), '<script type="module" src="/entry.js"></script>')
    fs.writeFileSync(path.join(app, 'entry.js'), 'import React from "react";console.log(React.version)')
    fs.writeFileSync(path.join(app, 'build.mjs'), 'import {build} from "vite";await build({root:process.cwd(),configLoader:"native",build:{outDir:process.env.PROBE_OUT,emptyOutDir:true}})')
    // Fixture-only resolver: this proves the config's behavior after external
    // resolution, not readiness of the real dependency installer or build.
    const hook = path.join(root, 'hook.mjs')
    fs.writeFileSync(hook, `import {registerHooks,isBuiltin,createRequire} from 'node:module';
import path from 'node:path';import {fileURLToPath,pathToFileURL} from 'node:url';
const source=process.env.PROBE_SOURCE,deps=process.env.PROBE_DEPS;
registerHooks({resolve(s,c,n){if(c.parentURL?.startsWith('file:')&&!s.startsWith('.')&&!s.startsWith('/')&&!s.includes(':')&&!isBuiltin(s)){
 const rel=path.relative(source,fileURLToPath(c.parentURL));if(rel&&!rel.startsWith('..'+path.sep)&&!path.isAbsolute(rel)){
 const anchor=path.join(deps,rel);if(c.conditions.includes('require'))return {url:pathToFileURL(createRequire(anchor).resolve(s)).href,shortCircuit:true};
 return n(s,{...c,parentURL:pathToFileURL(anchor).href})}}return n(s,c)}});`)
    const output = path.join(root, 'output')
    const deps = path.dirname(fs.realpathSync(path.join(source, 'node_modules')))
    const result = spawnSync(process.execPath, ['--import', hook, path.join(app, 'build.mjs')], {
      cwd: app, encoding: 'utf8', timeout: 45000,
      env: { ...process.env, PROBE_SOURCE: fixture, PROBE_DEPS: deps, PROBE_OUT: output,
        HERMES_DESKTOP_BUILD_DIR: path.join(root, 'job') }
    })
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`)
    for (const name of ['data.json', 'messages.json', 'shortcodes/emojibase.json']) {
      assert.ok(fs.statSync(path.join(output, 'emojibase/en', name)).size > 0)
    }
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
})
