import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from './catalog'

describe.each(Object.entries(TRANSLATIONS))('%s Semantier presentation', (_, t) => {
  it('identifies this desktop application consistently without changing external service or runtime identity', () => {
    expect(t.updates.appName).toBe('Semantier Agent')
    expect(t.boot.ready).toContain('Semantier')
    expect(t.notifications.native.turnDoneTitle).toContain('Semantier')
    expect(t.settings.gateway.cloudTitle).toBe('Hermes Cloud')
    expect(t.settings.connections.kindCloud).toBe('Hermes Cloud')
    expect(t.settings.connections.kindLocalDesc).toContain('Hermes')
    expect(t.settings.uninstallSection.options.gui.description).toContain('Hermes')
    expect(t.updates.bundleOutOfSyncDesc).toContain('Hermes')
    expect(t.composer.commandDescs['/version']).toContain('Hermes')
  })

  it('preserves values interpolated into application copy', () => {
    expect(t.updates.version('0.0.42')).toContain('0.0.42')
    expect(t.settings.gateway.connectedTo('https://gateway.example', '8.7.6')).toContain('8.7.6')
    expect(t.settings.gateway.connectedTo('https://gateway.example', '8.7.6')).toContain('Hermes')
  })
})
