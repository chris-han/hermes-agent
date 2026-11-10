import { describe, expect, it } from 'vitest'

import { desktopBotRoster, preferredDesktopBot } from './desktop-roster'
import type { RosterRow } from './types'

const local = (name: string): RosterRow => ({ name, connectionId: 'local', connectionKind: 'local' })
const route = { connectionId: null, profile: 'default' }

describe('managed desktop bot roster', () => {
  it('shows live profile owners until lifecycle migration retires them', () => {
    const root = local('semantier-local'), managed = local('default'), other = local('scout')
    const source = [root, other, managed]
    const visible = desktopBotRoster(source, route)
    expect(visible).toEqual([root, other, managed])
    expect(source).toEqual([root, other, managed])
    expect(visible[2]).toBe(managed)
    expect(preferredDesktopBot(visible, route)).toBe(managed)
  })

  it('does not drop remote roots or bots with the same names', () => {
    const remote = { name: 'semantier-local', connectionId: 'lab', connectionKind: 'remote' }
    const remoteManaged = { name: 'default', connectionId: 'lab', connectionKind: 'remote' }
    const source = [local('semantier-local'), local('default'), remote, remoteManaged]
    expect(desktopBotRoster(source, route)).toEqual(source)
  })

  it('keeps the roster unchanged until the managed local owner is confirmed', () => {
    const source = [local('semantier-local'), local('default')]
    expect(desktopBotRoster(source, null)).toBe(source)
    expect(desktopBotRoster(source, { connectionId: 'lab', profile: 'default' })).toBe(source)
    expect(desktopBotRoster([source[0]], route)).toEqual([source[0]])
    expect(preferredDesktopBot(source, null)).toBeUndefined()
  })

  it('does not suppress a real root on a stale or ghost-only managed roster', () => {
    const root = local('semantier-local')

    for (const managed of [{ ...local('default'), ghost: true }, { ...local('default'), retained: true }]) {
      const source = [root, managed]
      expect(desktopBotRoster(source, route)).toBe(source)
    }
  })
})
