import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test } from 'vitest'

import { parseFixtureArgs, validateFixtureSource } from './run-fixture-e2e.mjs'

const roots = []
afterEach(() => {
  for (const root of roots) fs.rmSync(root, { recursive: true, force: true })
})

test('defaults to the actual Electron smoke and accepts explicit existing specs', () => {
  expect(parseFixtureArgs([]).specs).toEqual(['e2e/fixture-smoke.spec.ts'])
  expect(parseFixtureArgs(['--python', '/fixture/python', 'e2e/cloud-sign-in.spec.ts'])).toMatchObject({
    python: '/fixture/python',
    specs: ['e2e/cloud-sign-in.spec.ts']
  })
  expect(() => parseFixtureArgs(['--python'])).toThrow(/value/)
  expect(() => parseFixtureArgs(['--unknown'])).toThrow(/Unknown/)
  expect(() => parseFixtureArgs(['../outside.spec.ts'])).toThrow(/e2e/)
})

test('refuses a source checkout that could hydrate developer credentials', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'fixture-source-'))
  roots.push(root)
  fs.mkdirSync(path.join(root, 'hermes_cli'))
  fs.writeFileSync(path.join(root, 'hermes_cli/main.py'), '')
  expect(() => validateFixtureSource(root)).not.toThrow()
  fs.writeFileSync(path.join(root, '.env'), 'SYNTHETIC_ONLY=value')
  expect(() => validateFixtureSource(root)).toThrow(/\.env/)
})

test('runner removes inherited fake boot/source/runtime flags rather than trusting them', async () => {
  const { fixtureRunnerEnv } = await import('./run-fixture-e2e.mjs')
  expect(
    fixtureRunnerEnv({
      HERMES_DESKTOP_BOOT_FAKE: '1',
      HERMES_DESKTOP_HERMES_ROOT: '/other',
      ELECTRON_RUN_AS_NODE: '1',
      HERMES_HOME: '/real',
      OPENAI_API_KEY: 'synthetic',
      PATH: '/bin'
    })
  ).toEqual({ PATH: '/bin' })
})

test('timeout remains failure when a child handles termination and exits zero', async () => {
  const { runFixtureChild } = await import('./run-fixture-e2e.mjs')
  const code = await runFixtureChild(
    process.execPath,
    ['-e', "process.on('SIGTERM',()=>process.exit(0));setInterval(()=>{},1000)"],
    {},
    250,
    100
  )
  expect(code).toBe(124)
})

test('cleanup retains failed or forged fixture homes and never stops an escaped home', async () => {
  const { stopOwnedGateways } = await import('./run-fixture-e2e.mjs')
  const { writeFixtureOwner } = await import('../e2e/fixture-environment.ts')
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'fixture-cleanup-'))
  roots.push(base)
  for (const name of ['failed', 'forged']) {
    const root = path.join(base, name)
    fs.mkdirSync(root)
    fs.mkdirSync(path.join(root, 'hermes-home'))
    fs.mkdirSync(path.join(root, 'electron-user-data'))
    writeFixtureOwner(
      { root, hermesHome: path.join(root, 'hermes-home'), userDataDir: path.join(root, 'electron-user-data') },
      name === 'forged' ? '/foreign' : '/source'
    )
  }
  const homes = []
  const retained = stopOwnedGateways(base, '/python', '/source', (_python, _args, options) => {
    homes.push(options.env.HERMES_HOME)
    throw new Error('Synthetic stop failure')
  })
  expect(homes).toEqual([path.join(base, 'failed/hermes-home')])
  expect(retained).toHaveLength(2)
  expect(fs.existsSync(path.join(base, 'failed/hermes-home'))).toBe(true)
})

test('fresh renderers use distinct OS-assigned ports and serve the selected source', async () => {
  const { startFixtureRenderer } = await import('./run-fixture-e2e.mjs')
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'fixture-renderer-'))
  roots.push(root)
  fs.writeFileSync(path.join(root, 'vite.config.ts'), 'export default {}')
  fs.writeFileSync(path.join(root, 'index.html'), '<h1>owned renderer fixture</h1>')
  const first = await startFixtureRenderer(root)
  const second = await startFixtureRenderer(root)
  try {
    expect(first.url).not.toBe(second.url)
    expect(await (await fetch(first.url)).text()).toContain('owned renderer fixture')
    expect(await (await fetch(second.url)).text()).toContain('owned renderer fixture')
  } finally {
    await first.close()
    await second.close()
  }
})

test('an all-skipped run is not accepted as executed Electron verification', async () => {
  const { readFixtureResults } = await import('./run-fixture-e2e.mjs')
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'fixture-results-'))
  roots.push(root)
  const file = path.join(root, 'results.json')
  fs.writeFileSync(file, JSON.stringify({ stats: { expected: 0, unexpected: 0, flaky: 0, skipped: 2 } }))
  expect(() => readFixtureResults(file)).toThrow(/No Electron tests executed/)
  fs.writeFileSync(file, JSON.stringify({ stats: { expected: 1, unexpected: 0, flaky: 0, skipped: 0 } }))
  expect(readFixtureResults(file).expected).toBe(1)
})
