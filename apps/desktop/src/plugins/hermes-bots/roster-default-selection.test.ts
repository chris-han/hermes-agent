import { host } from '@hermes/plugin-sdk'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $openBotChat, $rosterHydrated, $selectedRosterHydrated, $selectedRosterKey } from './bot-state'
import { reconcileRosterSelection } from './roster-pane'
import type { RosterRow } from './types'

const root: RosterRow = { name: 'semantier-local', connectionId: 'local', connectionKind: 'local' }
const managed: RosterRow = { name: 'default', connectionId: 'local', connectionKind: 'local' }
const defaultRoute = { connectionId: null, profile: 'default' }

beforeEach(() => {
  $rosterHydrated.set(true)
  $selectedRosterHydrated.set(true)
  $selectedRosterKey.set('')
  $openBotChat.set(null)
})

afterEach(() => {
  vi.restoreAllMocks()
  $selectedRosterKey.set('')
  $openBotChat.set(null)
  $rosterHydrated.set(false)
  $selectedRosterHydrated.set(false)
})

describe('native desktop initial bot selection', () => {
  it('prefers default over an earlier root or recent bot without creating a chat', () => {
    const request = vi.spyOn(host, 'request')
    const newChat = vi.spyOn(host, 'newChat')
    const openSession = vi.spyOn(host, 'openSession')
    reconcileRosterSelection([root, { ...managed, name: 'scout' }, managed], [], {}, defaultRoute)
    expect($selectedRosterKey.get()).toBe('local::default')
    expect(request).not.toHaveBeenCalled()
    expect(newChat).not.toHaveBeenCalled()
    expect(openSession).not.toHaveBeenCalled()
  })

  it('preserves a valid existing root selection and its open canonical-chat identity', () => {
    $selectedRosterKey.set('local::semantier-local')
    const opened = { key: 'local::semantier-local', openedRegistryId: 'existing-root-chat', openedSessionId: 'existing-root-tip' }
    $openBotChat.set(opened)
    reconcileRosterSelection([root, managed], [], {}, defaultRoute)
    expect($selectedRosterKey.get()).toBe('local::semantier-local')
    expect($openBotChat.get()).toBe(opened)
  })
})
