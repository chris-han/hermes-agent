import { expect, it } from 'vitest'

import { $sessionTiles, dropTilesForConnection } from './session-states'
import { $connection } from './session'

it('drops old Cloud account tabs while preserving Local and other connection tabs', () => {
  $sessionTiles.set([
    { storedSessionId: 'cloud-private', workspaceMode: 'bots', ownerRoute: { connectionId: 'cloud', profile: 'default', mode: 'remote' } },
    { storedSessionId: 'local-private', workspaceMode: 'bots', ownerRoute: { connectionId: 'local', profile: 'default', mode: 'local' } },
    { storedSessionId: 'ssh-private', workspaceMode: 'bots', ownerRoute: { connectionId: 'ssh', profile: 'default', mode: 'remote' } }
  ])
  dropTilesForConnection('cloud')
  expect($sessionTiles.get().map(row => row.storedSessionId)).toEqual(['local-private', 'ssh-private'])
  dropTilesForConnection('local')
  expect($sessionTiles.get().map(row => row.storedSessionId)).toEqual(['local-private', 'ssh-private'])
})


it('preserves an explicitly Local session even when its tab is in the Cloud workspace', () => {
  $connection.set({ connectionId: 'cloud', profile: 'default', mode: 'remote', baseUrl: 'https://cloud.example' } as never)
  $sessionTiles.set([
    { storedSessionId: 'cloud-session', workspaceMode: 'sessions' },
    { storedSessionId: 'local-session', workspaceMode: 'sessions', ownerRoute: { connectionId: 'local', profile: 'default', mode: 'local' } }
  ])
  dropTilesForConnection('cloud')
  expect($sessionTiles.get().map(row => row.storedSessionId)).toEqual(['local-session'])
  $connection.set(null)
})
