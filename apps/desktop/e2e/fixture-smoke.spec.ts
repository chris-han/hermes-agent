import fs from 'node:fs'

import { expect, test } from '@playwright/test'

import { DesktopFixtureSession } from './fixture-session'
import { buildAppEnv, createSandbox, setupNoProvider } from './fixtures'

// eslint-disable-next-line no-empty-pattern -- Electron supplies its own page; only Playwright TestInfo is used.
test('source Electron fixture boots canonical default and produces useful failure diagnostics', async ({}, info) => {
  test.skip(!process.env.HERMES_DESKTOP_DEV_SERVER, 'Use scripts/run-fixture-e2e.mjs')
  // Native Windows source transforms can exceed the WSL startup budget on a cold renderer.
  test.setTimeout(process.platform === 'win32' ? 300_000 : 150_000)
  const sandbox = createSandbox('smoke')
  fs.writeFileSync(sandbox.hermesHome + '/config.yaml', 'language: en\n')

  const env = buildAppEnv(sandbox, {
    HERMES_PYTHON: process.env.HERMES_PYTHON!,
    HERMES_DESKTOP_DEV_SERVER: process.env.HERMES_DESKTOP_DEV_SERVER!,
    HERMES_DESKTOP_CDP_PORT: 'off'
  })

  const fixture = new DesktopFixtureSession(sandbox, env, info)

  try {
    const { page } = await fixture.launch()
    await fixture.stage('desktop.ready', async () => {
      await expect(page.getByRole('button', { name: 'default', exact: true })).toHaveAttribute('aria-pressed', 'true', {
        timeout: process.platform === 'win32' ? 180_000 : 90_000
      })
      await expect(fixture.button('Sign in to Semantier Model Provider').or(fixture.button('登录 Semantier 模型服务'))).toBeVisible({ timeout: 30_000 })
    })
    // A deliberate caught failure qualifies the diagnostic path on actual Electron DOM.
    await expect(
      fixture.stage('diagnostic.counterexample', async () => {
        throw new Error('synthetic diagnostic counterexample')
      })
    ).rejects.toThrow('synthetic diagnostic counterexample')
    expect(fixture.records.at(-1)?.accessibility).toMatch(/Sign in to Semantier Model Provider|登录 Semantier 模型服务/)
    expect(fs.existsSync(sandbox.hermesHome + '/profiles/semantier-local')).toBe(false)
  } finally {
    await fixture.close()
  }
})


test('existing no-provider fixture also uses the source-renderer lane without a full build', async () => {
  test.skip(!process.env.HERMES_DESKTOP_DEV_SERVER, 'Use scripts/run-fixture-e2e.mjs')
  test.setTimeout(120_000)
  const fixture = await setupNoProvider()

  try {
    await expect(fixture.page.getByRole('button', { name: 'default', exact: true })).toHaveAttribute('aria-pressed', 'true', { timeout: 90_000 })
  } finally { await fixture.cleanup() }
})
