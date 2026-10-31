import { describe, expect, it } from 'vitest'

import { desktopBotRoster, preferredDesktopBot } from './desktop-roster'
import type { RosterRow } from './types'

const local = (name: string): RosterRow => ({ name, connectionId: 'local', connectionKind: 'local' })
const route = { connectionId: null, profile: 'semantier-local' }

describe('managed desktop bot roster', () => {
  it('shows the managed bot once, omits the duplicate root, and preserves raw owner rows', () => {
    const root = local('default'),
      managed = local('semantier-local'),
      other = local('scout')
    const source = [root, other, managed]
    const visible = desktopBotRoster(source, route)
    expect(visible).toEqual([other, managed])
    expect(source).toEqual([root, other, managed])
    expect(visible[1]).toBe(managed)
    expect(preferredDesktopBot(visible, route)).toBe(managed)
  })

  it('does not drop remote roots or bots with the same names', () => {
    const remote = { name: 'default', connectionId: 'lab', connectionKind: 'remote' }
    const remoteManaged = { name: 'semantier-local', connectionId: 'lab', connectionKind: 'remote' }
    const source = [local('default'), local('semantier-local'), remote, remoteManaged]
    expect(desktopBotRoster(source, route)).toEqual([source[1], remote, remoteManaged])
  })

  it('keeps the roster unchanged until the managed local owner is confirmed', () => {
    const source = [local('default'), local('semantier-local')]
    expect(desktopBotRoster(source, null)).toBe(source)
    expect(desktopBotRoster(source, { connectionId: 'lab', profile: 'semantier-local' })).toBe(source)
    expect(desktopBotRoster([source[0]], route)).toEqual([source[0]])
    expect(preferredDesktopBot(source, null)).toBeUndefined()
  })

  it('does not suppress a real root on a stale or ghost-only managed roster', () => {
    const root = local('default')

    for (const managed of [
      { ...local('semantier-local'), ghost: true },
      { ...local('semantier-local'), retained: true }
    ]) {
      const source = [root, managed]
      expect(desktopBotRoster(source, route)).toBe(source)
    }
  })
})
