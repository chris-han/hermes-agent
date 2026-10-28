import { execFileSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { expect, test } from 'vitest'

const identity = fileURLToPath(new URL('../product-identity.cjs', import.meta.url))

const derive = (version: string) =>
  JSON.parse(
    execFileSync(process.execPath, ['-e', 'console.log(JSON.stringify(require(process.argv[1])))', identity], {
      cwd: path.dirname(identity),
      encoding: 'utf8',
      env: {
        ...process.env,
        HERMES_DESKTOP_VARIANT: 'bundled',
        HERMES_BUILD_COMMIT: 'a'.repeat(40),
        HERMES_PAYLOAD_TAG: '',
        SEMANTIER_DESKTOP_VERSION: version
      }
    })
  )

test('readable Semantier version changes display and executable names while preserving native identity', () => {
  const upstream = derive('')
  const semantier = derive('0.0.1')
  expect(semantier.displayName).toBe('Semantier Agent 0.0.1')
  expect(semantier.windowsExecutableName).toBe('Semantier Agent')

  for (const key of ['appId', 'appNamePascal', 'artifactNamePascal', 'cliName', 'msixAppIdWithOrg', 'channel']) {
    expect(semantier[key]).toEqual(upstream[key])
  }

  expect(() => derive('invalid')).toThrow()
})
