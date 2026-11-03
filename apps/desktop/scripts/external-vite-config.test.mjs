import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { fileURLToPath, pathToFileURL } from 'node:url'
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
    fs.mkdirSync(path.join(app, 'src'))
    fs.writeFileSync(path.join(app, 'entry.js'), 'import React from "react";import {atom} from "nanostores";import "./src/styles.css";console.log(React.version,atom(1).get())')
    fs.writeFileSync(path.join(app, 'src/styles.css'), "@import 'tailwindcss';\n@import 'tw-shimmer';\n@font-face{font-family:fixture;src:url('../../../node_modules/@nous-research/ui/dist/fonts/Collapse-Bold.woff2')}\n.fixture{font-family:fixture}\n")
    fs.writeFileSync(path.join(app, 'build.mjs'), 'import {build} from "vite";await build({root:process.cwd(),configLoader:"native",build:{outDir:process.env.PROBE_OUT,emptyOutDir:true}})')
    const hook = pathToFileURL(path.join(source, 'scripts/build/external-node-resolver.mjs')).href
    const output = path.join(root, 'output')
    const deps = path.dirname(fs.realpathSync(path.join(source, 'node_modules')))
    const result = spawnSync(process.execPath, ['--import', hook, path.join(app, 'build.mjs')], {
      cwd: app, encoding: 'utf8', timeout: 45000,
      env: { ...process.env, HERMES_NODE_SOURCE_ROOT: fixture, HERMES_NODE_DEPENDENCY_ROOT: deps, PROBE_OUT: output,
        HERMES_DESKTOP_BUILD_DIR: path.join(root, 'job'), HERMES_DESKTOP_NODE_ROOT: deps }
    })
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`)
    for (const name of ['data.json', 'messages.json', 'shortcodes/emojibase.json']) {
      assert.ok(fs.statSync(path.join(output, 'emojibase/en', name)).size > 0)
    }
    assert.ok(fs.readdirSync(path.join(output, 'assets')).some(name => name.startsWith('Collapse-Bold') && name.endsWith('.woff2')))
    const overlap = spawnSync(process.execPath, ['--import', hook, path.join(app, 'build.mjs')], {
      cwd: app, encoding: 'utf8', timeout: 45000,
      env: { ...process.env, HERMES_NODE_SOURCE_ROOT: fixture, HERMES_NODE_DEPENDENCY_ROOT: deps, PROBE_OUT: path.join(root, 'refused'),
        HERMES_DESKTOP_BUILD_DIR: path.join(root, 'job'), HERMES_DESKTOP_NODE_ROOT: fixture }
    })
    assert.notEqual(overlap.status, 0)
    assert.match(overlap.stderr, /dependencies must not overlap source/)
    assert.equal(fs.existsSync(path.join(root, 'refused')), false)
    const wrong = path.join(root, 'wrong-dependencies')
    fs.mkdirSync(wrong)
    const mismatch = spawnSync(process.execPath, ['--import', hook, path.join(app, 'build.mjs')], {
      cwd: app, encoding: 'utf8', timeout: 45000,
      env: { ...process.env, HERMES_NODE_SOURCE_ROOT: fixture, HERMES_NODE_DEPENDENCY_ROOT: deps, PROBE_OUT: path.join(root, 'mismatched'),
        HERMES_DESKTOP_BUILD_DIR: path.join(root, 'job'), HERMES_DESKTOP_NODE_ROOT: wrong }
    })
    assert.notEqual(mismatch.status, 0)
    assert.match(mismatch.stderr, /outside selected dependency root/)
    assert.equal(fs.existsSync(path.join(root, 'mismatched')), false)
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
})
