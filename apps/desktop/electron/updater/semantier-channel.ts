import { readFileSync } from 'node:fs'

import { type ChannelBuild, channelPublicBase } from './channel-protocol'

export type PackagedReleaseChannel = 'production' | 'development'
const channels = { production: 'semantier-production', development: 'semantier-acceptance' } as const

/** Only the Semantier-owned Windows package policy may opt into these streams. */
export function supportsPackagedChannels(build: ChannelBuild | undefined): boolean {
  return Boolean(build && build.repository === 'chris-han/hermes-agent' &&
    ['Semantier.Agent', 'Semantier.Agent.Acceptance'].includes(build.identity.msixAppIdWithOrg) &&
    Object.values(channels).includes(build.channel as typeof channels[PackagedReleaseChannel]))
}

function choice(value: unknown): PackagedReleaseChannel {
  if (value !== 'production' && value !== 'development') {
    throw new Error('Unsupported packaged update channel')
  }

  return value
}

function preferences(path: string): Record<string, unknown> {
  let text: string

  try {
    text = readFileSync(path, 'utf8')
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') { return {} }
    throw error
  }

  const value: unknown = JSON.parse(text)

  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Invalid desktop update preferences')
  }

  return value as Record<string, unknown>
}

export function packagedChannelSelection(build: ChannelBuild | undefined, path: string, base = ''):
  { choice: PackagedReleaseChannel; build: ChannelBuild } | null {
  if (!build || !supportsPackagedChannels(build)) { return null }
  const saved = preferences(path).releaseChannel
  const selected = saved === undefined ? (build.channel === channels.production ? 'production' : 'development') : choice(saved)

  return { choice: selected, build: { ...build, channel: channels[selected], publicBase: channelPublicBase(base || build.publicBase) } }
}

/** The existing atomic desktop preference writer owns the actual write. */
export function savePackagedChannel(path: string, value: unknown, write: (path: string, text: string) => void): void {
  const selected = choice(value)
  write(path, JSON.stringify({ ...preferences(path), releaseChannel: selected }, null, 2))
}

/** Reject a stale renderer selection before the native installation handoff. */
export function assertPackagedChannelSelection(selected: PackagedReleaseChannel, expected: unknown): void {
  if (selected !== expected) { throw new Error('Update channel changed. Check for updates again before installing.') }
}
