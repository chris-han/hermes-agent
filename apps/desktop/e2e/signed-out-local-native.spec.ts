/** Actual Windows source Electron with a signed-out local runtime and loopback inference. */
import path from 'node:path'

import { expect, test } from '@playwright/test'
import { startMockServer } from '../../../tests-js/scripts/mock-server'
import { runDesktopChatSmoke } from '../../../tests-js/scripts/desktop-chat-smoke'
import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { DesktopFixtureSession } from './fixture-session'
import { buildAppEnv, createSandbox } from './fixtures'

// eslint-disable-next-line no-empty-pattern -- Electron supplies its own page.
test('signed-out Local completes a turn without Semantier cloud', async ({}, info) => {
  test.skip(process.platform !== 'win32' || !process.env.HERMES_DESKTOP_DEV_SERVER, 'Requires isolated Windows source Electron')
  test.setTimeout(300_000)
  const mock = await startMockServer()
  const sandbox = createSandbox('signed-out-local')
  writeMockProviderConfig(sandbox.hermesHome, mock.url, undefined, 'language: en\ndisplay:\n  language: en\n')
  writeEnvFile(sandbox.hermesHome)
  const fixture = new DesktopFixtureSession(sandbox, buildAppEnv(sandbox, {
    HERMES_PYTHON: process.env.HERMES_PYTHON!, HERMES_DESKTOP_PYTHON: process.env.HERMES_PYTHON!,
    HERMES_DESKTOP_DEV_SERVER: process.env.HERMES_DESKTOP_DEV_SERVER!, HERMES_DESKTOP_CDP_PORT: 'off'
  }), info)
  try {
    const { app, page } = await fixture.launch()
    await expect(page.getByRole('button', { name: 'default', exact: true })).toHaveAttribute('aria-pressed', 'true', { timeout: 180_000 })
    const before = await page.evaluate(() => (window as any).hermesDesktop.connections.list())
    expect(before.primary).toBe('local')
    expect(before.accountContext).toBeFalsy()
    // The account origin is deliberately unreachable for the renderer's session.
    await app.evaluate(({ session }) => {
      session.defaultSession.webRequest.onBeforeRequest({ urls: ['https://cloud.semantier.cn/*'] },
        (_details: unknown, callback: (value: { cancel: boolean }) => void) => callback({ cancel: true }))
    })
    const turn = await runDesktopChatSmoke(page, { mockUrl: mock.url, phase: 'installed', outDir: info.outputPath('local-turn') })
    expect(turn.status).toBe('passed')
    expect(mock.receivedPrompts.some(value => value.includes(turn.prompt))).toBe(true)
    const after = await page.evaluate(() => (window as any).hermesDesktop.connections.list())
    expect(after.primary).toBe('local')
    expect(after.accountContext).toBeFalsy()
    expect(path.resolve(sandbox.hermesHome)).toContain(path.resolve(sandbox.root))
  } finally {
    await fixture.close()
    await mock.close()
  }
})
