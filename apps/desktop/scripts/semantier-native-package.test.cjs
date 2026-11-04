'use strict'
const { test } = require('node:test')
const assert = require('node:assert/strict')
const config = require('../electron-builder.config.cjs')

test('packager includes and unpacks each prepared native module', () => {
  for (const name of ['node-pty', 'get-windows']) {
    assert.ok(config.files.some(item => typeof item === 'object' && item.from === `dist/node_modules/${name}` && item.to === `dist/node_modules/${name}` && item.filter.includes('**/*')), `${name} must be explicitly included`)
    assert.ok(config.asar.unpack.includes(`**/node_modules/${name}/**`), `${name} must be unpacked`)
  }
})

test('desktop package includes the HTTP/3 client license files', () => {
  const fs = require('node:fs')
  const path = require('node:path')
  const notices = path.join(__dirname, '../public/third-party/semantier-cloud-http3')
  assert.ok(config.files.includes('public/**'))
  for (const name of ['README.txt', 'curl-cffi.txt', 'curl-impersonate.txt', 'curl.txt', 'boringssl.txt', 'fiat-crypto.txt', 'zlib.txt', 'brotli.txt', 'zstd.txt', 'nghttp2.txt', 'ngtcp2.txt', 'nghttp3.txt', 'cffi.txt', 'certifi.txt', 'pycparser.txt']) {
    assert.ok(fs.statSync(path.join(notices, name)).size > 50, `${name} must be included`)
  }
})
