const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const { verifyFilexInputs, isPinnedFilex } = require('./filex-inputs.cjs')
test('clean checkout and wrong executable cannot yield a sync-enabled package', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'filex-inputs-'))
  try {
    assert.throws(() => verifyFilexInputs(root, 'win32', 'x64'), /ENOENT/)
    fs.mkdirSync(path.join(root, 'resources', 'filex'), { recursive: true })
    fs.writeFileSync(path.join(root, 'resources', 'filex', 'filex.exe'), 'wrong-platform-or-corrupt')
    assert.throws(() => verifyFilexInputs(root, 'win32', 'x64'), /checksum/)
    assert.throws(() => verifyFilexInputs(root, 'win32', 'arm64'), /Unsupported/)
  } finally { fs.rmSync(root, { recursive: true, force: true }) }
})
test('signing exception covers only the pinned package destination', () => {
  assert.equal(isPinnedFilex('C:\\app\\resources\\filex\\filex.exe', 'C:\\app'), true)
  assert.equal(isPinnedFilex('/app/resources/filex/filex.exe', '/app'), true)
  for (const file of ['/app/filex.exe', '/app/resources/agent-payload/filex.exe', '/app/resources/agent-payload/tools/resources/filex/filex.exe', '/app/resources/filex/other.exe']) assert.equal(isPinnedFilex(file, '/app'), false)
})
test('prepared pinned payload validates read-only, and stale helper/receipt are rejected', { skip: !process.env.FILEX_INPUTS_DESKTOP }, () => {
  const desktop = process.env.FILEX_INPUTS_DESKTOP
  const root = path.join(desktop, 'resources', 'filex')
  const helper = path.join(root, 'filex-child.mjs')
  const receipt = path.join(root, 'inputs.json')
  const helperBytes = fs.readFileSync(helper), receiptBytes = fs.readFileSync(receipt)
  try {
    fs.chmodSync(root, 0o500)
    assert.equal(verifyFilexInputs(desktop, 'win32', 'x64').version, '0.61.1')
    fs.chmodSync(root, 0o700)
    fs.writeFileSync(helper, 'stale helper')
    assert.throws(() => verifyFilexInputs(desktop, 'win32', 'x64'), /Stale.*helper/)
    fs.writeFileSync(helper, helperBytes)
    fs.writeFileSync(receipt, JSON.stringify({ ...JSON.parse(receiptBytes), platform: 'linux' }))
    assert.throws(() => verifyFilexInputs(desktop, 'win32', 'x64'), /wrong-platform/)
  } finally {
    fs.chmodSync(root, 0o700)
    fs.writeFileSync(helper, helperBytes); fs.writeFileSync(receipt, receiptBytes)
  }
})
