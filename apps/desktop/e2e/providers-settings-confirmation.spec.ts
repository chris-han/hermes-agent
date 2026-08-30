import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from './test'
import {
  buildAppEnv,
  createSandbox,
  launchDesktop,
  type MockBackendFixture,
  waitForAppReady,
  writeEnvFile,
  writeMockProviderConfig
} from './fixtures'
import { startMockServer } from './mock-server'

let fixture: MockBackendFixture | null = null

function findCredentialEnv(root: string): string | null {
  for (const relativeEntry of fs.readdirSync(root, { recursive: true })) {
    const relative = relativeEntry.toString()
    if (path.basename(relative) !== '.env') {
      continue
    }

    const candidate = path.join(root, relative)

    if (fs.readFileSync(candidate, 'utf8').includes('OPENROUTER_API_KEY=e2e-openrouter-key')) {
      return candidate
    }
  }

  return null
}

test.beforeAll(async () => {
  const mock = await startMockServer()
  const sandbox = createSandbox('provider-confirm')
  writeMockProviderConfig(sandbox.hermesHome, mock.url)
  writeEnvFile(sandbox.hermesHome)
  const { app, page } = await launchDesktop(buildAppEnv(sandbox))
  fixture = {
    app,
    page,
    mock,
    mockUrl: mock.url,
    sandbox,
    cleanup: async () => {
      await app.close().catch(() => undefined)
      await mock.close()
      sandbox.cleanup()
    }
  }
  await waitForAppReady(fixture, 120_000)
})

test.afterAll(async () => {
  await fixture?.cleanup()
  fixture = null
})

test('API-key removal is cancel-safe, keyboard accessible, and confirmed once', async () => {
  const { page, sandbox } = fixture!

  await page.getByRole('button', { name: 'Open settings' }).click()
  await page.getByRole('button', { name: 'Providers' }).click()
  await page.getByRole('button', { name: 'API keys' }).click()

  const keyInput = page.getByPlaceholder('Paste OpenRouter key')
  await keyInput.fill('e2e-openrouter-key')
  await page.getByRole('button', { exact: true, name: 'Save' }).click()

  const maskedKey = page.locator('input[readonly]').first()
  await expect(maskedKey).toBeVisible({ timeout: 30_000 })
  let envPath: string | null = null
  await expect.poll(() => {
    envPath = findCredentialEnv(sandbox.hermesHome)

    return envPath
  }, { timeout: 30_000 }).not.toBeNull()

  await maskedKey.focus()
  const removeButton = page.getByRole('button', { exact: true, name: 'Remove' })
  await removeButton.click()

  const dialog = page.getByRole('dialog')
  await expect(dialog).toBeVisible()
  await expect(dialog).toContainText(/Remove .* from \.env\?/)
  await expect(dialog.getByRole('button', { name: 'Confirm' })).toBeFocused()

  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()
  expect(fs.readFileSync(envPath!, 'utf8')).toContain('OPENROUTER_API_KEY=e2e-openrouter-key')

  await removeButton.click()
  await expect(dialog).toBeVisible()
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).toBeHidden()
  expect(fs.readFileSync(envPath!, 'utf8')).toContain('OPENROUTER_API_KEY=e2e-openrouter-key')

  await removeButton.click()
  await expect(dialog).toBeVisible()
  await dialog.getByRole('button', { name: 'Confirm' }).click()
  await expect(dialog).toBeHidden()
  await expect.poll(() => fs.readFileSync(envPath!, 'utf8')).not.toContain('OPENROUTER_API_KEY=')
  await expect(removeButton).toHaveCount(0)
})
