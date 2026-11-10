#!/usr/bin/env node
/** Optional source-Electron lane. No dependency installation, production credentials or package build. */
import { execFileSync, spawn } from 'node:child_process'
import fs from 'node:fs'
import { createServer as createHttpServer } from 'node:http'
import path from 'node:path'
import os from 'node:os'
import { fileURLToPath } from 'node:url'

import { assertFixtureOwner, FIXTURE_OWNER, stripCredentials } from '../e2e/fixture-environment.ts'

const source = path.resolve(import.meta.dirname, '../../..')

export function parseFixtureArgs(argv) {
  const result = { specs: [], timeoutMs: 600_000 }
  const names = { '--python': 'python', '--output': 'output', '--timeout-ms': 'timeoutMs' }
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]
    if (arg === '--help') {
      result.help = true
      continue
    }
    if (names[arg]) {
      const value = argv[++i]
      if (!value || value.startsWith('--')) throw new Error(`${arg} requires a value`)
      result[names[arg]] = arg === '--timeout-ms' ? Number(value) : value
    } else {
      if (arg.startsWith('-')) throw new Error(`Unknown option: ${arg}`)
      if (!/^e2e\/[a-zA-Z0-9_/-]+\.spec\.ts$/.test(arg) || arg.includes('..')) {
        throw new Error('Select a repository-owned e2e/*.spec.ts path')
      }
      result.specs.push(arg)
    }
  }
  if (!Number.isSafeInteger(result.timeoutMs) || result.timeoutMs <= 0) throw new Error('Invalid timeout-ms value')
  if (!result.specs.length) result.specs.push('e2e/fixture-smoke.spec.ts')
  return result
}

export function validateFixtureSource(root) {
  if (!fs.existsSync(path.join(root, 'hermes_cli/main.py'))) throw new Error('Not a Hermes source checkout')
  if (fs.existsSync(path.join(root, '.env'))) {
    throw new Error('Source .env could hydrate developer credentials. Run from a clean source worktree.')
  }
}

export function runFixtureChild(command, args, options, timeoutMs, graceMs = 5000) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { ...options, detached: process.platform !== 'win32', stdio: 'inherit' })
    let stopped = 0
    let force
    const signal = name => {
      if (!child.pid || child.exitCode !== null || child.signalCode !== null) return
      try {
        if (process.platform === 'win32') child.kill(name)
        else process.kill(-child.pid, name)
      } catch (error) {
        if (error.code !== 'ESRCH') throw error
      }
    }
    const stop = code => {
      stopped ||= code
      signal('SIGTERM')
      force ||= setTimeout(() => signal('SIGKILL'), graceMs)
    }
    const timer = setTimeout(() => stop(124), timeoutMs)
    const interrupt = () => stop(130)
    const dispose = () => {
      clearTimeout(timer)
      clearTimeout(force)
      process.off('SIGINT', interrupt)
      process.off('SIGTERM', interrupt)
    }
    process.on('SIGINT', interrupt)
    process.on('SIGTERM', interrupt)
    child.once('error', error => {
      dispose()
      reject(error)
    })
    child.once('exit', (code, signal) => {
      dispose()
      resolve(stopped || code || (signal ? 130 : 0))
    })
  })
}

/** Public middleware API avoids Vite interpreting configured port zero as its default port. */
export async function startFixtureRenderer(desktop) {
  const { createServer } = await import('vite')
  const http = createHttpServer()
  const vite = await createServer({
    root: desktop,
    configFile: path.join(desktop, 'vite.config.ts'),
    server: { middlewareMode: true, hmr: false, watch: null },
    clearScreen: false
  })
  http.on('request', vite.middlewares)
  try {
    await new Promise((resolve, reject) => {
      http.once('error', reject)
      http.listen(0, '127.0.0.1', resolve)
    })
  } catch (error) {
    await vite.close()
    throw error
  }
  return {
    url: `http://127.0.0.1:${http.address().port}`,
    close: async () => {
      await vite.close()
      http.closeAllConnections()
      await new Promise((resolve, reject) => http.close(error => (error ? reject(error) : resolve())))
    }
  }
}

export function fixtureRunnerEnv(env) {
  const clean = stripCredentials(env)
  for (const key of Object.keys(clean)) {
    if (key.startsWith('HERMES_DESKTOP_') || key === 'ELECTRON_RUN_AS_NODE') delete clean[key]
  }
  return clean
}

/** Recover only marked homes below this invocation's own temporary directory. */
export function stopOwnedGateways(tmp, python, root, stop = execFileSync) {
  const retained = []
  for (const item of fs.readdirSync(tmp, { withFileTypes: true })) {
    const candidate = path.join(tmp, item.name)
    if (!item.isDirectory() || !fs.existsSync(path.join(candidate, FIXTURE_OWNER))) continue
    try {
      const owner = assertFixtureOwner(candidate, root, tmp)
      stop(python, ['-P', '-m', 'hermes_cli.main', 'gateway', 'stop'], {
        cwd: root,
        env: { ...stripCredentials(process.env), HOME: owner.root, HERMES_HOME: owner.hermesHome, PYTHONPATH: root },
        timeout: 20_000,
        stdio: 'pipe'
      })
      fs.rmSync(candidate, { recursive: true })
    } catch {
      retained.push(candidate)
    }
  }
  return retained
}

export function readFixtureResults(file) {
  const stats = JSON.parse(fs.readFileSync(file, 'utf8')).stats
  if (!stats || !(stats.expected + stats.unexpected + stats.flaky > 0)) {
    throw new Error('No Electron tests executed; skipped tests are not verification')
  }
  return stats
}

export async function runFixtureE2E(options) {
  validateFixtureSource(source)
  const desktop = path.join(source, 'apps/desktop')
  const python =
    options.python ||
    process.env.HERMES_DESKTOP_PYTHON ||
    process.env.HERMES_PYTHON ||
    path.join(source, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')
  if (!path.isAbsolute(python) || !fs.existsSync(python))
    throw new Error('Select an existing absolute --python interpreter path')
  for (const spec of options.specs) {
    if (!fs.existsSync(path.join(desktop, spec))) throw new Error(`Missing spec: ${spec}`)
  }
  const parent = path.join(desktop, 'test-results')
  fs.mkdirSync(parent, { recursive: true })
  let output = options.output ? path.resolve(options.output) : fs.mkdtempSync(path.join(parent, 'fixture-'))
  if (options.output && fs.existsSync(output) && fs.readdirSync(output).length)
    throw new Error('--output must be new or empty')
  fs.mkdirSync(output, { recursive: true })
  output = fs.realpathSync(output)
  // Chromium's Unix-domain sockets cannot live under arbitrarily long artifact paths.
  const tmp = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'ef-')))
  const receipt = { source, python, output, tmp, stage: 'python', status: 'running', stages: [] }
  const save = () => fs.writeFileSync(path.join(output, 'runner.json'), JSON.stringify(receipt, null, 2))
  const stage = async (name, action) => {
    receipt.stage = name
    save()
    const start = Date.now()
    try {
      const result = await action()
      receipt.stages.push({ name, status: 'passed', durationMs: Date.now() - start })
      return result
    } catch (error) {
      receipt.stages.push({ name, status: 'failed', durationMs: Date.now() - start, error: error.name })
      throw error
    } finally {
      save()
    }
  }
  let renderer
  let code = 1
  try {
    const probeHome = path.join(tmp, 'python-probe')
    fs.mkdirSync(probeHome)
    receipt.interpreter = await stage('python', () =>
      JSON.parse(
        execFileSync(
          python,
          ['-c', 'import json,sys; print(json.dumps({"executable":sys.executable,"version":sys.version.split()[0]}))'],
          {
            env: { ...stripCredentials(process.env), HOME: probeHome, HERMES_HOME: probeHome },
            timeout: 15_000,
            encoding: 'utf8'
          }
        )
      )
    )
    await stage('dev-bundle', async () => {
      const { bundleElectronMain } = await import('./bundle-electron-main.mjs')
      await bundleElectronMain({ source, out: path.join(desktop, 'dist'), dev: true })
    })
    await stage('vite', async () => {
      renderer = await startFixtureRenderer(desktop)
      receipt.renderer = renderer.url
    })
    const env = {
      ...fixtureRunnerEnv(process.env),
      HERMES_PYTHON: python,
      HERMES_DESKTOP_PYTHON: python,
      HERMES_DESKTOP_DEV_SERVER: receipt.renderer,
      PLAYWRIGHT_JSON_OUTPUT_FILE: path.join(output, 'results.json'),
      HERMES_E2E_RUN_TMP: tmp,
      TMPDIR: tmp,
      TMP: tmp,
      TEMP: tmp
    }
    const args = [
      path.join(source, 'node_modules/@playwright/test/cli.js'),
      'test',
      ...options.specs,
      '--workers=1',
      '--retries=0',
      '--reporter=list,json',
      '--output',
      path.join(output, 'playwright')
    ]
    code = await stage('playwright', async () => {
      const exit =
        process.platform === 'linux'
          ? await runFixtureChild(
              'xvfb-run',
              ['-a', process.execPath, ...args],
              { cwd: desktop, env },
              options.timeoutMs
            )
          : await runFixtureChild(process.execPath, args, { cwd: desktop, env }, options.timeoutMs)
      if (exit !== 0) throw Object.assign(new Error(`Playwright exited ${exit}`), { exitCode: exit })
      receipt.tests = readFixtureResults(path.join(output, 'results.json'))
      return exit
    })
    receipt.status = code === 0 ? 'passed' : 'failed'
  } catch (error) {
    receipt.status = 'failed'
    code = error.exitCode || 1
    console.error(`[fixture:${receipt.stage}] ${error.message}`)
  } finally {
    receipt.retained = stopOwnedGateways(tmp, python, source)
    if (receipt.retained.length) {
      code = 1
      receipt.status = 'cleanup_failed'
    }
    if (renderer) await renderer.close()
    if (!receipt.retained.length) fs.rmSync(tmp, { recursive: true })
    save()
    console.log(
      `FIXTURE_RESULT ${JSON.stringify({ status: receipt.status, stage: receipt.stage, tests: receipt.tests, output, retained: receipt.retained })}`
    )
  }
  return code
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const options = parseFixtureArgs(process.argv.slice(2))
    if (options.help)
      console.log(
        'node apps/desktop/scripts/run-fixture-e2e.mjs [--python /absolute/python] [--output /empty/directory] [--timeout-ms 600000] [e2e/name.spec.ts ...]'
      )
    else process.exitCode = await runFixtureE2E(options)
  } catch (error) {
    console.error(`[fixture:preflight] ${error.message}`)
    process.exitCode = 1
  }
}
