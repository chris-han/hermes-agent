import { fileURLToPath } from 'node:url'

import { expect, it } from 'vitest'

import { runManagedDeviceKey } from './managed-device-key-process'

it('uses the real bounded Python helper and does not turn a missing Windows key into enrollment', async () => {
  if (process.platform === 'win32') {
    return
  } // This test asserts the non-Windows boundary only.

  await expect(
    runManagedDeviceKey(
      {
        command: process.env.HERMES_PYTHON || 'python3',
        cwd: fileURLToPath(new URL('../../../', import.meta.url)),
        env: { ...process.env, PYTHONPATH: fileURLToPath(new URL('../../../', import.meta.url)) }
      },
      { operation: 'public', key_id: 'a'.repeat(32) }
    )
  ).rejects.toMatchObject({ nativeDeviceCode: 'NATIVE_DEVICE_KEY_UNAVAILABLE' })
})
it('rejects an arbitrary native operation before launching an executable', async () => {
  await expect(
    runManagedDeviceKey({ command: 'must-not-launch', cwd: process.cwd(), env: {} }, {
      operation: 'export_private',
      key_id: 'a'.repeat(32)
    } as never)
  ).rejects.toThrow('Invalid native device request')
})
