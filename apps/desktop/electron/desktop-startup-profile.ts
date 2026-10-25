import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

import { DESKTOP_PROFILE_NAME_RE, type DesktopProfileRoute } from './desktop-profile'
import { ticketRuntimeFor } from './local-gateway-python'
import { hiddenWindowsChildOptions } from './windows-child-options'

/** A private, bounded call to the selected runtime's profile lifecycle owner. */
export async function resolveDesktopStartupProfile(
  backend: Parameters<typeof ticketRuntimeFor>[0],
  ownerRoot: string,
  profile: string | null,
  explicit: boolean
): Promise<{ profile: string; recovered: boolean }> {
  const runtime = await ticketRuntimeFor(backend)
  const args = ['-P', '-m', 'hermes_cli.profiles_desktop', '--owner-root', ownerRoot]

  if (profile) { args.push('--profile', profile) }

  if (explicit) { args.push('--explicit') }

  const { stdout } = await promisify(execFile)(runtime.command, args, hiddenWindowsChildOptions({
    cwd: backend.root,
    env: { ...process.env, ...runtime.env, HERMES_HOME: ownerRoot, PYTHONIOENCODING: 'utf-8' },
    timeout: 90_000,
    maxBuffer: 1024 * 1024
  }))

  const result = JSON.parse(stdout)

  if (result.error) { throw new Error(result.error) }

  if (typeof result.profile !== 'string' || !DESKTOP_PROFILE_NAME_RE.test(result.profile) || typeof result.recovered !== 'boolean') {
    throw new Error('Invalid local startup profile result')
  }

  return result
}

/** Resolve ordinary local intent before a window/device flow can capture it. */
export async function prepareDesktopStartupProfile(deps: {
  preference: { profile: string | null; defaultRoute: DesktopProfileRoute | null }
  explicitProfile: string | null
  isLocalConnection: (id: string) => boolean
  isRemote: (profile: string | null) => boolean
  resolve: (profile: string | null, explicit: boolean) => Promise<{ profile: string; recovered: boolean }>
  remember: (profile: string) => void
  setDefault: (route: DesktopProfileRoute) => void
  log: (message: string) => void
}): Promise<void> {
  const route = deps.preference.defaultRoute
  const selected = deps.explicitProfile || route?.profile || deps.preference.profile

  if (route?.connectionId && !deps.isLocalConnection(route.connectionId)) { return }

  if (deps.isRemote(selected)) { return }

  const resolved = await deps.resolve(selected, Boolean(deps.explicitProfile))

  deps.remember(resolved.profile)

  if (route || resolved.profile === 'semantier-local') {
    deps.setDefault({ connectionId: route?.connectionId ?? null, profile: resolved.profile })
  }

  if (resolved.recovered) {
    deps.log(`[startup-profile] resolved ordinary local startup to ${resolved.profile}`)
  }
}
