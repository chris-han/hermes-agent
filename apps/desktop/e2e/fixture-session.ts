/** Owned source-Electron lifecycle and bounded diagnostics for synthetic local fixtures. */
import { execFile } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { _electron, type ElectronApplication, type Page, type TestInfo } from '@playwright/test'

import { resolveElectronBinary } from './electron-binary'
import { assertFixtureOwner } from './fixture-environment'
import type { Sandbox } from './fixtures'

export function buttonLabelPattern(label: string): RegExp {
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

  return new RegExp(`^\\s*(?:\\[\\s*)?${escaped}(?:\\s*\\])?\\s*$`)
}

export function sanitizeDiagnostic(text: string): string {
  return text
    .replace(/Bearer\s+[^\s"']+/gi, 'Bearer [redacted]')
    .replace(/https?:\/\/[^\s<>"']+/g, raw => {
      try {
        const url = new URL(raw)
        url.username = ''
        url.password = ''
        url.search = ''
        url.hash = ''

        return url.toString()
      } catch {
        return '[invalid URL]'
      }
    })
    .replace(/(textbox "[^"\n]*(?:password|token|secret)[^"\n]*":)[^\n]*/gi, '$1 [redacted]')
    .slice(0, 12_000)
}

export interface FixtureStageRecord {
  name: string
  status: 'passed' | 'failed'
  durationMs: number
  at: string
  error?: string
  url?: string
  accessibility?: string
}

export async function recordFixtureStage<T>(
  records: FixtureStageRecord[],
  name: string,
  action: () => Promise<T>,
  page: () => Page | undefined,
  persist: (records: FixtureStageRecord[]) => Promise<void>
): Promise<T> {
  const start = performance.now()

  try {
    const value = await action()
    records.push({
      name,
      status: 'passed',
      durationMs: Math.round(performance.now() - start),
      at: new Date().toISOString()
    })
    await persist(records)

    return value
  } catch (error) {
    const current = page()

    const accessibility = current
      ? await current
          .locator('body')
          .ariaSnapshot({ timeout: 2000 })
          .catch(() => '[page unavailable]')
      : '[no renderer page]'

    records.push({
      name,
      status: 'failed',
      durationMs: Math.round(performance.now() - start),
      at: new Date().toISOString(),
      error: sanitizeDiagnostic(error instanceof Error ? error.message : String(error)),
      url: current ? sanitizeDiagnostic(current.url()) : undefined,
      accessibility: sanitizeDiagnostic(accessibility)
    })
    await persist(records)
    throw error
  }
}

/** Native lifecycle authority, shared by source fixtures; never infer a process from argv. */
export async function stopFixtureGateway(root: string, sandbox: Sandbox, env: Record<string, string>): Promise<void> {
  assertFixtureOwner(sandbox.root, root, process.env.HERMES_E2E_RUN_TMP || os.tmpdir())

  if (
    env.HERMES_HOME !== sandbox.hermesHome ||
    env.HOME !== sandbox.root ||
    env.HERMES_DESKTOP_ISOLATED_BACKEND !== '1'
  ) {
    throw new Error('Gateway stop requires the owned isolated fixture home')
  }

  const python =
    env.HERMES_DESKTOP_PYTHON ||
    env.HERMES_PYTHON ||
    path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')

  try {
    await promisify(execFile)(python, ['-P', '-m', 'hermes_cli.main', 'gateway', 'stop'], {
      cwd: root,
      env: { ...env, PYTHONPATH: root },
      // CLI cold imports plus Windows graceful gateway shutdown can exceed20s.
      // Ownership checks above remain mandatory; this wait is still bounded.
      timeout: 60_000
    })
  } catch (error) {
    if (process.env.HERMES_E2E_RUN_TMP)
      {fs.writeFileSync(path.join(process.env.HERMES_E2E_RUN_TMP, '.retain-fixtures'), '')}

    throw error
  }
}

export class DesktopFixtureSession {
  app?: ElectronApplication
  page?: Page
  readonly records: FixtureStageRecord[] = []
  private readonly sandbox: Sandbox
  private readonly env: Record<string, string>
  private readonly root: string
  private readonly info: TestInfo

  constructor(sandbox: Sandbox, env: Record<string, string>, info: TestInfo) {
    this.sandbox = sandbox
    this.env = env
    this.root = path.resolve(import.meta.dirname, '../../..')
    this.info = info
    assertFixtureOwner(sandbox.root, this.root, process.env.HERMES_E2E_RUN_TMP || os.tmpdir())

    if (
      env.HERMES_HOME !== sandbox.hermesHome ||
      env.HOME !== sandbox.root ||
      env.HERMES_DESKTOP_ISOLATED_BACKEND !== '1'
    ) {
      throw new Error('Fixture environment must use its owned isolated home')
    }

    if (env.HERMES_DESKTOP_HERMES_ROOT !== this.root) {throw new Error('Fixture must launch its selected source root')}

    if (fs.existsSync(path.join(this.root, '.env'))) {throw new Error('Source .env is unsafe for synthetic fixtures')}

    if (!env.HERMES_DESKTOP_PYTHON) {throw new Error('Fixture requires an explicit Desktop Python interpreter')}
  }

  async stage<T>(name: string, action: () => Promise<T>): Promise<T> {
    return recordFixtureStage(
      this.records,
      name,
      action,
      () => this.page,
      async records => {
        const output = this.info.outputPath('fixture-stages.json')
        fs.mkdirSync(path.dirname(output), { recursive: true })
        fs.writeFileSync(
          output,
          JSON.stringify(
            {
              source: this.root,
              python: this.env.HERMES_DESKTOP_PYTHON,
              home: this.sandbox.hermesHome,
              stages: records
            },
            null,
            2
          )
        )
        const last = records.at(-1)!
        console.log(
          'FIXTURE_STAGE',
          JSON.stringify({ name: last.name, status: last.status, durationMs: last.durationMs })
        )

        if (last.status === 'failed')
          {await this.info.attach('fixture-failure', { path: output, contentType: 'application/json' })}
      }
    )
  }

  button(label: string) {
    if (!this.page) {throw new Error('Fixture renderer has not launched')}

    return this.page.getByRole('button', { name: buttonLabelPattern(label) })
  }

  async launch(): Promise<{ app: ElectronApplication; page: Page }> {
    return this.stage('electron.launch', async () => {
      this.app = await _electron.launch({
        executablePath: resolveElectronBinary([path.join(this.root, 'apps/desktop'), this.root]),
        args: [
          path.join(this.root, 'apps/desktop'),
          ...(process.platform === 'linux' ? ['--no-sandbox', '--disable-gpu', '--ozone-platform=x11'] : [])
        ],
        cwd: path.join(this.root, 'apps/desktop'),
        env: this.env
      })
      this.page = await this.app.firstWindow({ timeout: 90_000 })
      this.page.setDefaultTimeout(10_000)

      return { app: this.app, page: this.page }
    })
  }

  private async stop(): Promise<void> {
    if (this.app) {
      await this.app.close().catch(() => undefined)
      this.app = undefined
    }

    await this.stage('gateway.stop', async () => {
      await stopFixtureGateway(this.root, this.sandbox, this.env)
    })
  }

  async restart(): Promise<{ app: ElectronApplication; page: Page }> {
    await this.stop()

    return this.launch()
  }

  async close(): Promise<void> {
    await this.stop()
    this.sandbox.cleanup()
  }
}
