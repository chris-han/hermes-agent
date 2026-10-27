import { contrastRatio } from '@hermes/shared/color'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'

import { __resetBackendSkinSync, ingestBackendSkin } from './backend-sync'
import { skinPref, ThemeProvider, useTheme } from './context'
import { BUILTIN_THEMES, DEFAULT_SKIN_NAME } from './presets'
import { useSkinCommand } from './use-skin-command'

let theme: ReturnType<typeof useTheme>
let skin: ReturnType<typeof useSkinCommand>

function Probe() {
  theme = useTheme()
  skin = useSkinCommand()

  return null
}

beforeEach(() => {
  localStorage.clear()
  $activeGatewayProfile.set('default')
  __resetBackendSkinSync()
})
afterEach(cleanup)

describe('Semantier desktop appearance', () => {
  it('starts with Semantier and keeps it through a stock backend connection', () => {
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>
    )
    expect(DEFAULT_SKIN_NAME).toBe('semantier')
    expect(theme.themeName).toBe('semantier')
    act(() => ingestBackendSkin({ name: 'default', colors: { ui_accent: '#ffbf00' } }, { apply: false }))
    expect(theme.themeName).toBe('semantier')
    expect(theme.availableThemes[0].name).toBe('semantier')
    expect(theme.availableThemes.some(item => item.name === 'nous-alt')).toBe(false)
  })

  it.each(['default', 'research'])('replaces a saved Nous Alt pick for %s and persists the new identity', profile => {
    skinPref.assign(profile, 'nous-alt')
    $activeGatewayProfile.set(profile)
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>
    )
    expect(theme.themeName).toBe('semantier')
    expect(skinPref.stored(profile)).toBe('semantier')
  })

  it('preserves an explicit alternative and lets /skin default restore Semantier', () => {
    skinPref.assign('default', 'mono')
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>
    )
    expect(theme.themeName).toBe('mono')
    act(() => void skin('default'))
    expect(theme.themeName).toBe('semantier')
    expect(skinPref.resolve('default')).toBe('semantier')
  })

  it('paints readable action fills and a stable design identity in both modes', () => {
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>
    )

    for (const mode of ['light', 'dark'] as const) {
      act(() => theme.setMode(mode))
      const root = window.document.documentElement
      expect(root.dataset.hermesDesign).toBe('asimov')
      expect(root.style.getPropertyValue('--dt-primary-solid')).toBe('#e8c92d')
      expect(root.style.getPropertyValue('--dt-primary-solid-foreground')).toBe('#1a2028')
      expect(root.style.getPropertyValue('--dt-font-sans')).toMatch(/^"Manrope", "Noto Sans SC"/)
    }

    act(() => theme.setTheme('mono'))
    expect(window.document.documentElement.dataset.hermesDesign).toBeUndefined()
  })

  it('keeps content, actions, destructive fills and focus readable on every shipped palette', () => {
    const seed = BUILTIN_THEMES.semantier
    expect(seed).toBeDefined()

    for (const c of [seed.colors, seed.darkColors!]) {
      for (const [foreground, background] of [
        [c.foreground, c.background],
        [c.cardForeground, c.card],
        [c.mutedForeground, c.muted],
        [c.primaryForeground, c.primary],
        [c.destructiveForeground, c.destructive]
      ]) {
        expect(contrastRatio(foreground, background)).toBeGreaterThanOrEqual(4.5)
      }

      for (const background of [c.background, c.card, c.muted, c.sidebarBackground!]) {
        expect(contrastRatio(c.ring, background)).toBeGreaterThanOrEqual(3)
      }

      expect(seed.terminal?.foreground).toBeTruthy()
      expect(seed.darkTerminal?.foreground).toBeTruthy()
    }
  })
})
