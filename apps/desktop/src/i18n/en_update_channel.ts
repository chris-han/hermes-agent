import type { UpdateChannelCopy } from './types_update_channel'

export const enUpdateChannel: UpdateChannelCopy = {
  production: "Production",
  development: "Development",
  packagedDescription: "Production follows published releases. Development includes test builds. Switching only checks for updates; installation stays your choice.",
  title: 'Update channel',
  description: 'Stable installs published releases only. Every commit follows the main branch as changes land.',
  stable: 'Stable releases',
  main: 'Every commit',
  stableConfirmTitle: 'Switch to stable releases?',
  stableConfirmBody:
    'Semantier will move to the latest published release, which can be older than the code you run now. Data written by newer code may not read correctly, so back it up first.',
  stableConfirm: 'Switch to stable',
  failed: 'Could not change the update channel',
  detailsLabel: 'Channel',
  branch: (b: string) => `Branch: ${b}`,
  change: 'Change',
  latestRelease: (v: string) => `You’re on the latest stable release (${v}).`,
  rollbackTitle: (v: string) => `Back to stable release ${v}`,
  rollbackBody: (v: string) =>
    `This build is newer than the stable release ${v}. Installing it moves Semantier back to that release.`,
  rollbackAction: (v: string) => `Roll back to ${v}`
}
