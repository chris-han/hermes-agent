import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'
import { stampExeIdentity } from './set-exe-identity.mjs'

const root = path.resolve(import.meta.dirname, '../../..')

test('all non-Store release variants use the Semantier certificate subject without legacy version selectors', () => {
  for (const variant of ['', 'bundled', 'light']) {
    const env = Object.fromEntries(Object.entries(process.env).filter(([key]) =>
      !/^(HERMES_|SEMANTIER_DESKTOP_VERSION$|_HERMES_|AZURE_)/.test(key)))
    const result = JSON.parse(execFileSync(process.execPath, ['-e', `
      const config = require('./apps/desktop/electron-builder.config.cjs');
      const identity = require('./apps/desktop/product-identity.cjs');
      console.log(JSON.stringify({ publisher: config.msix.publisher, publisherDisplayName: config.msix.publisherDisplayName,
        author: require('./apps/desktop/package.json').author, nativeIdentity: identity.msixAppIdWithOrg }));
    `], { cwd: root, env: { ...env, HERMES_DESKTOP_VARIANT: variant }, encoding: 'utf8' }))
    assert.equal(result.publisher, 'CN=Semantier Technology, O=Semantier Technology', variant)
    assert.equal(result.publisherDisplayName, 'Semantier Technology', variant)
    assert.equal(result.author, 'Semantier Technology', variant)
    assert.match(result.nativeIdentity, /^Semantier\./, variant)
  }
})

test('Windows executable resources identify Semantier as the product company', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'semantier-publisher-'))
  try {
    fs.mkdirSync(path.join(directory, 'assets'))
    fs.writeFileSync(path.join(directory, 'assets/icon.ico'), 'fixture')
    const exe = path.join(directory, 'test.exe')
    fs.writeFileSync(exe, 'fixture')
    let stamped
    await stampExeIdentity(exe, directory, { rcedit: async (_path, options) => { stamped = options['version-string'] } })
    assert.equal(stamped.CompanyName, 'Semantier Technology')
    assert.match(stamped.LegalCopyright, /Semantier Technology$/)
  } finally { fs.rmSync(directory, { recursive: true, force: true }) }
})

test('Setup packaging identifies Semantier as publisher', () => {
  const config = JSON.parse(fs.readFileSync(path.join(root, 'apps/bootstrap-installer/src-tauri/tauri.conf.json'), 'utf8'))
  assert.equal(config.bundle.publisher, 'Semantier Technology')
  assert.match(config.bundle.copyright, /Semantier Technology$/)
  assert.match(config.identifier, /^com\.semantier\./)
})
