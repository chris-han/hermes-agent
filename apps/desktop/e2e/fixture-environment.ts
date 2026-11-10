import fs from 'node:fs'
import path from 'node:path'

const CREDENTIAL_SUFFIXES: string[] = [
  '_API_KEY',
  '_TOKEN',
  '_SECRET',
  '_PASSWORD',
  '_CREDENTIALS',
  '_ACCESS_KEY',
  '_PRIVATE_KEY',
  '_OAUTH_TOKEN'
]

const CREDENTIAL_NAMES = new Set([
  'ANTHROPIC_BASE_URL',
  'ANTHROPIC_TOKEN',
  'AWS_ACCESS_KEY_ID',
  'AWS_SECRET_ACCESS_KEY',
  'AWS_SESSION_TOKEN',
  'CUSTOM_API_KEY',
  'GEMINI_BASE_URL',
  'OPENAI_BASE_URL',
  'OPENROUTER_BASE_URL',
  'OLLAMA_BASE_URL',
  'GROQ_BASE_URL',
  'XAI_BASE_URL'
])

function isCredentialEnvVar(name: string): boolean {
  if (CREDENTIAL_NAMES.has(name)) {
    return true
  }

  return CREDENTIAL_SUFFIXES.some(suffix => name.endsWith(suffix))
}

// Runtime state of whatever Hermes launched this run. A spec driven from inside
// an agent's terminal inherits HERMES_YOLO_MODE, HERMES_INTERACTIVE,
// HERMES_SESSION_ID…, and the sandboxed backend then skips approvals or binds
// the caller's session — the approval spec failed locally on the leaked yolo
// flag while CI (which never has these) stayed green. The fixtures set every
// HERMES_* the app needs themselves; only the harness's own knobs pass.
function isInheritedHermesRuntimeVar(name: string): boolean {
  return name.startsWith('HERMES_') && !name.startsWith('HERMES_DESKTOP_') && !name.startsWith('HERMES_E2E_')
}

export function stripCredentials(env: Record<string, string | undefined>): Record<string, string> {
  const clean: Record<string, string> = {}

  for (const [key, value] of Object.entries(env)) {
    if (!value) {
      continue
    }

    if (isCredentialEnvVar(key) || isInheritedHermesRuntimeVar(key)) {
      continue
    }

    clean[key] = value
  }

  return clean
}

/** Both names are needed: Desktop selects HERMES_DESKTOP_PYTHON, CLI helpers use HERMES_PYTHON. */
export function fixturePythonEnv(extra: Record<string, string>): Record<string, string> {
  const python = extra.HERMES_DESKTOP_PYTHON || extra.HERMES_PYTHON

  return { ...extra, ...(python ? { HERMES_PYTHON: python, HERMES_DESKTOP_PYTHON: python } : {}) }
}

export interface FixtureOwner {
  root: string
  hermesHome: string
  userDataDir: string
}

export const FIXTURE_OWNER = '.electron-fixture.json'

export function writeFixtureOwner(owner: FixtureOwner, source: string): void {
  fs.writeFileSync(path.join(owner.root, FIXTURE_OWNER), JSON.stringify({ ...owner, source }), { mode: 0o600 })
}

/** A marker alone grants nothing: every destructive path must remain under this run's owned root. */
export function assertFixtureOwner(root: string, source: string, base: string): FixtureOwner {
  const relative = path.relative(fs.realpathSync(base), path.resolve(root))

  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error('Fixture owner is outside the owned run root')
  }

  const home = path.join(root, 'hermes-home')
  const userDataDir = path.join(root, 'electron-user-data')

  for (const file of [root, path.join(root, FIXTURE_OWNER), home, userDataDir]) {
    if (fs.lstatSync(file).isSymbolicLink()) {throw new Error('Fixture owner contains a symlink')}
  }

  const marker = JSON.parse(fs.readFileSync(path.join(root, FIXTURE_OWNER), 'utf8'))

  if (
    marker.root !== root ||
    marker.hermesHome !== home ||
    marker.userDataDir !== userDataDir ||
    marker.source !== source
  ) {
    throw new Error('Fixture owner does not match the selected source and home')
  }

  return { root, hermesHome: home, userDataDir }
}
