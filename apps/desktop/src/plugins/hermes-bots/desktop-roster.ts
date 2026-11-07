import type { host } from '@hermes/plugin-sdk'

import type { RosterRow } from './types'

export type NativeDefaultRoute = ReturnType<typeof host.state.defaultProfileRoute.get>

const managedDefault = (route: NativeDefaultRoute) =>
  route?.profile === 'semantier-local' && (!route.connectionId || route.connectionId === 'local')

const localManagedBot = (bot: RosterRow) =>
  bot.name === 'semantier-local' &&
  bot.connectionId === 'local' &&
  bot.connectionKind === 'local' &&
  !bot.ghost &&
  !bot.retained

/** Presentation only. Full owner rows remain in the shared roster for chats,
 * groups, mentions and explicitly selected historical workspaces. */
export function desktopBotRoster(roster: RosterRow[], route: NativeDefaultRoute): RosterRow[] {
  if (!managedDefault(route) || !roster.some(localManagedBot)) {
    return roster
  }

  return roster.filter(
    bot => !(bot.name === 'default' && bot.connectionId === 'local' && bot.connectionKind === 'local')
  )
}

/** Initial selection only; never replaces an existing valid selected owner. */
export function preferredDesktopBot(roster: RosterRow[], route: NativeDefaultRoute): RosterRow | undefined {
  return managedDefault(route) ? roster.find(localManagedBot) : undefined
}
