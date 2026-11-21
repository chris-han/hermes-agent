import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { expect, it } from 'vitest'

import { runAccountFence } from './account-fence-process'

it('does not report success when native protected storage is unavailable', async () => {
  if (process.platform === 'win32') {return}
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'account-fence-test-'))
  const source = fileURLToPath(new URL('../../../', import.meta.url))
  try {
    await expect(runAccountFence({ command: process.env.HERMES_PYTHON || 'python3', cwd: source,
      env: { ...process.env, HERMES_HOME: root, PYTHONPATH: source } }, null)).rejects.toThrow('unavailable')
  } finally {await fs.rm(root, { recursive: true, force: true })}
})
