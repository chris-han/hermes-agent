import type { host } from '@hermes/plugin-sdk'

import type { RosterRow } from './types'

export type NativeDefaultRoute = ReturnType<typeof host.state.defaultProfileRoute.get>

const managedDefault = (route: NativeDefaultRoute) =>
  route?.profile === 'default' && (!route.connectionId || route.connectionId === 'local')

const localManagedBot = (bot: RosterRow) =>
  bot.name === 'default' && bot.connectionId === 'local' && bot.connectionKind === 'local' &&
  !bot.ghost && !bot.retained

/** Retirement belongs to the profile lifecycle; still-live owners stay visible. */
export function desktopBotRoster(roster: RosterRow[], _route: NativeDefaultRoute): RosterRow[] {
  return roster
}

/** Initial selection only; never replaces an existing valid selected owner. */
export function preferredDesktopBot(roster: RosterRow[], route: NativeDefaultRoute): RosterRow | undefined {
  return managedDefault(route) ? roster.find(localManagedBot) : undefined
}
