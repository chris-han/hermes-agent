import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import { checkDistBuilt } from '../scripts/assert-dist-built.mjs'

function makeDist(extra) {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-assert-dist-'))
  const distDir = path.join(tempRoot, 'dist')
  fs.mkdirSync(distDir, { recursive: true })
  if (extra) extra(distDir)
  return { tempRoot, distDir }
}

function writeRouterAsset(distDir, name) {
  fs.writeFileSync(
    path.join(distDir, 'assets', name),
    `throw new Error('may be used only in the context of a <Router>')`,
    'utf8',
  )
}

function writeQueryClientAsset(distDir, name) {
  fs.writeFileSync(
    path.join(distDir, 'assets', name),
    `throw new Error('No QueryClient set, use QueryClientProvider to set one')`,
    'utf8',
  )
}

test('checkDistBuilt passes when index.html + an assets JS bundle exist', () => {
  const { tempRoot, distDir } = makeDist(d => {
    fs.writeFileSync(path.join(d, 'index.html'), '<!doctype html><div id=root></div>', 'utf8')
    fs.mkdirSync(path.join(d, 'assets'))
    fs.writeFileSync(path.join(d, 'assets', 'index-abc123.js'), 'console.log(1)', 'utf8')
  })
  try {
    assert.deepEqual(checkDistBuilt(distDir), { ok: true })
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkDistBuilt fails when the dist directory is absent', () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-assert-dist-'))
  try {
    const result = checkDistBuilt(path.join(tempRoot, 'dist'))
    assert.equal(result.ok, false)
    assert.match(result.error, /no dist directory/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkDistBuilt fails when index.html is missing', () => {
  const { tempRoot, distDir } = makeDist(d => {
    fs.mkdirSync(path.join(d, 'assets'))
    fs.writeFileSync(path.join(d, 'assets', 'index-abc123.js'), 'console.log(1)', 'utf8')
  })
  try {
    const result = checkDistBuilt(distDir)
    assert.equal(result.ok, false)
    assert.match(result.error, /index\.html is missing/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkDistBuilt fails when index.html is empty', () => {
  const { tempRoot, distDir } = makeDist(d => {
    fs.writeFileSync(path.join(d, 'index.html'), '', 'utf8')
    fs.mkdirSync(path.join(d, 'assets'))
    fs.writeFileSync(path.join(d, 'assets', 'index-abc123.js'), 'console.log(1)', 'utf8')
  })
  try {
    const result = checkDistBuilt(distDir)
    assert.equal(result.ok, false)
    assert.match(result.error, /index\.html is empty/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkDistBuilt fails when assets/ has no JS bundle', () => {
  const { tempRoot, distDir } = makeDist(d => {
    fs.writeFileSync(path.join(d, 'index.html'), '<!doctype html>', 'utf8')
    fs.mkdirSync(path.join(d, 'assets'))
    // CSS only, no JS — still a blank page at runtime.
    fs.writeFileSync(path.join(d, 'assets', 'index-abc123.css'), 'body{}', 'utf8')
  })
  try {
    const result = checkDistBuilt(distDir)
    assert.equal(result.ok, false)
    assert.match(result.error, /no built JS bundle/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkDistBuilt passes when the Router context invariant is in one JS asset', () => {
  const { tempRoot, distDir } = makeDist(d => {
    fs.writeFileSync(path.join(d, 'index.html'), '<!doctype html>', 'utf8')
    fs.mkdirSync(path.join(d, 'assets'))
    writeRouterAsset(d, 'vendor-react-abc123.js')
    fs.writeFileSync(path.join(d, 'assets', 'command-def456.js'), 'console.log(1)', 'utf8')
  })
  try {
    assert.deepEqual(checkDistBuilt(distDir), { ok: true })
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkDistBuilt fails when the Router context invariant is in multiple JS assets', () => {
  const { tempRoot, distDir } = makeDist(d => {
    fs.writeFileSync(path.join(d, 'index.html'), '<!doctype html>', 'utf8')
    fs.mkdirSync(path.join(d, 'assets'))
    writeRouterAsset(d, 'vendor-react-abc123.js')
    writeRouterAsset(d, 'command-def456.js')
  })
  try {
    const result = checkDistBuilt(distDir)
    assert.equal(result.ok, false)
    assert.match(result.error, /react-router context invariant found in multiple JS assets/)
    assert.match(result.error, /vendor-react-abc123\.js/)
    assert.match(result.error, /command-def456\.js/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkDistBuilt passes when the QueryClient context invariant is in one JS asset', () => {
  const { tempRoot, distDir } = makeDist(d => {
    fs.writeFileSync(path.join(d, 'index.html'), '<!doctype html>', 'utf8')
    fs.mkdirSync(path.join(d, 'assets'))
    writeQueryClientAsset(d, 'vendor-react-abc123.js')
    fs.writeFileSync(path.join(d, 'assets', 'command-def456.js'), 'console.log(1)', 'utf8')
  })
  try {
    assert.deepEqual(checkDistBuilt(distDir), { ok: true })
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkDistBuilt fails when the QueryClient context invariant is in multiple JS assets (#95560)', () => {
  const { tempRoot, distDir } = makeDist(d => {
    fs.writeFileSync(path.join(d, 'index.html'), '<!doctype html>', 'utf8')
    fs.mkdirSync(path.join(d, 'assets'))
    writeQueryClientAsset(d, 'vendor-react-abc123.js')
    writeQueryClientAsset(d, 'session-list-density-def456.js')
  })
  try {
    const result = checkDistBuilt(distDir)
    assert.equal(result.ok, false)
    assert.match(result.error, /@tanstack\/react-query context invariant found in multiple JS assets/)
    assert.match(result.error, /vendor-react-abc123\.js/)
    assert.match(result.error, /session-list-density-def456\.js/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkDistBuilt fails when a chunk is not valid ES module syntax', () => {
  const { tempRoot, distDir } = makeDist(d => {
    fs.writeFileSync(path.join(d, 'index.html'), '<!doctype html>', 'utf8')
    fs.mkdirSync(path.join(d, 'assets'))
    // Invalid: destructuring pattern with a hole, like a bundle that lost an
    // identifier token mid-minification (`{categories:n}` minus `categories`).
    fs.writeFileSync(path.join(d, 'assets', 'index-abc123.js'), 'let {,:n}=x', 'utf8')
    fs.writeFileSync(path.join(d, 'assets', 'vendor-def456.js'), 'export const ok = 1', 'utf8')
  })
  try {
    const result = checkDistBuilt(distDir)
    assert.equal(result.ok, false)
    assert.match(result.error, /not valid ES module syntax/)
    assert.match(result.error, /index-abc123\.js/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('checkDistBuilt passes when every chunk parses as an ES module', () => {
  const { tempRoot, distDir } = makeDist(d => {
    fs.writeFileSync(path.join(d, 'index.html'), '<!doctype html>', 'utf8')
    fs.mkdirSync(path.join(d, 'assets'))
    fs.writeFileSync(path.join(d, 'assets', 'index-abc123.js'), 'export const a = `x`', 'utf8')
    fs.writeFileSync(path.join(d, 'assets', 'vendor-def456.js'), 'import.meta.url', 'utf8')
  })
  try {
    assert.deepEqual(checkDistBuilt(distDir), { ok: true })
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('launch smoke rejects a detached development bundle and verifies its transitive identity module', async () => {
  const { checkDesktopLaunch } = await import('./assert-dist-built.mjs')
  const { tempRoot, distDir } = makeDist(d => {
    fs.writeFileSync(path.join(d, 'index.html'), '<script type="module" src="./assets/main.js"></script>')
    fs.mkdirSync(path.join(d, 'assets'))
    fs.writeFileSync(path.join(d, 'assets/main.js'), 'export const ready = true')
    fs.writeFileSync(path.join(d, 'electron-main.mjs'), 'const x = typeof __HERMES_PRODUCT_IDENTITY__ === "undefined" ? require("../product-identity.cjs") : __HERMES_PRODUCT_IDENTITY__')
    for (const name of ['electron-preload.js', 'preview-guest-preload.js']) fs.writeFileSync(path.join(d, name), '')
  })
  try {
    assert.match(checkDesktopLaunch(distDir).error, /development identity/)
    fs.writeFileSync(path.join(tempRoot, 'product-identity.cjs'), 'module.exports = require("./identity-input.cjs")')
    assert.match(checkDesktopLaunch(distDir).error, /development identity/)
    fs.writeFileSync(path.join(tempRoot, 'identity-input.cjs'), 'module.exports = { name: "fixture" }')
    assert.deepEqual(checkDesktopLaunch(distDir), { ok: true })
    fs.rmSync(path.join(distDir, 'electron-preload.js'))
    assert.match(checkDesktopLaunch(distDir).error, /electron-preload/)
  } finally { fs.rmSync(tempRoot, { recursive: true, force: true }) }
})

test('launch smoke accepts injected packaged identity without source modules and rejects missing HTML assets', async () => {
  const { checkDesktopLaunch } = await import('./assert-dist-built.mjs')
  const { tempRoot, distDir } = makeDist(d => {
    fs.writeFileSync(path.join(d, 'index.html'), '<script src="./assets/missing.js"></script>')
    fs.mkdirSync(path.join(d, 'assets'))
    fs.writeFileSync(path.join(d, 'assets/main.js'), 'export const ready = true')
    fs.writeFileSync(path.join(d, 'electron-main.mjs'), 'const identity = { name: "injected" }')
    for (const name of ['electron-preload.js', 'preview-guest-preload.js']) fs.writeFileSync(path.join(d, name), '')
  })
  try {
    assert.match(checkDesktopLaunch(distDir).error, /missing.js/)
    fs.writeFileSync(path.join(distDir, 'index.html'), '<script src="./assets/main.js"></script>')
    assert.deepEqual(checkDesktopLaunch(distDir), { ok: true })
  } finally { fs.rmSync(tempRoot, { recursive: true, force: true }) }
})
