const test = require('node:test')
const assert = require('node:assert/strict')
const { execFileSync } = require('node:child_process')
const path = require('node:path')
const config = path.resolve(__dirname, '..', 'electron-builder.config.cjs')
function output(env) {
  return execFileSync(process.execPath, ['-e', `process.stdout.write(require(${JSON.stringify(config)}).directories.output)`],
    { encoding: 'utf8', env: { ...process.env, ...env } })
}
test('desktop output defaults to legacy location outside the controlled Windows build', () => {
  assert.equal(output({ SEMANTIER_DESKTOP_BUILD_OUTPUT: '' }), 'release')
})
test('controlled Windows build can redirect packaging out of the checkout', () => {
  const target = String.raw`D:\repo\deliverables\work\job-1\electron-output`
  assert.equal(output({ SEMANTIER_DESKTOP_BUILD_OUTPUT: target }), target)
})
