import { test } from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { startNativePackageOrigin } from './native-package-origin.mjs'

test('only pinned native bytes are served, with actual HTTP range and mutation refusal', async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'native-origin-'))
  const body = Buffer.from('signed-byte fixture; no native qualification')
  const key = `releases/channel-builds/${'a'.repeat(32)}/main.msix`
  const row = { key, size: body.length, sha256: createHash('sha256').update(body).digest('hex') }
  const file = path.join(dir, 'main.msix')
  await fs.writeFile(file, body)
  const events = []
  let origin
  try {
    await assert.rejects(startNativePackageOrigin(dir, { root: key, objects: [{ ...row, sha256: '0'.repeat(64) }] }), /digest/)
    origin = await startNativePackageOrigin(dir, { root: key, objects: [row] }, event => events.push(event))
    const url = origin.base + '/' + key
    const response = await fetch(url, { headers: { Range: 'bytes=7-10' } })
    assert.equal(response.status, 206)
    assert.equal(response.headers.get('content-range'), `bytes 7-10/${body.length}`)
    assert.equal(await response.text(), body.subarray(7, 11).toString())
    assert.equal((await fetch(url, { method: 'HEAD' })).headers.get('content-length'), String(body.length))
    assert.equal((await fetch(url, { headers: { Range: 'bytes=99999-' } })).status, 416)
    const changedValidator = await fetch(url, { headers: { Range: 'bytes=7-10', 'If-Range': '"old"' } })
    assert.equal(changedValidator.status, 200)
    assert.equal(await changedValidator.text(), body.toString())
    assert.equal((await fetch(url + '?candidate=other')).status, 404)
    assert.equal((await fetch(origin.base + '/main.msix')).status, 404)
    await fs.writeFile(file, Buffer.alloc(body.length, 120))
    assert.equal((await fetch(url)).status, 409)
    assert.ok(events.some(event => event.status === 206 && event.bytesRead === 4))
    const base = origin.base
    await origin.close(); origin = null
    await fs.writeFile(file, body)
    origin = await startNativePackageOrigin(dir, { root: key, objects: [row] }, () => {}, Number(new URL(base).port))
    assert.equal(origin.base, base)
    assert.equal(await (await fetch(url)).text(), body.toString())
  } finally { await origin?.close(); await fs.rm(dir, { recursive: true, force: true }) }
})
