/**
 * Built-in desktop themes. Names match the CLI skins / dashboard presets.
 * Add new themes here — no code changes needed elsewhere.
 *
 * The palette-bearing skins (nous, catppuccin, everforest, solarized) are forks
 * of their VS Code originals, converted by `buildThemeFromMarketplace` (see
 * ./install.ts) from the extensions below — the same path a Marketplace import
 * takes, so each is identical to installing the extension by hand and costs the
 * user neither the download nor the install step.
 *
 *   nous       ← github.github-vscode-theme   (Light Default / Dark Default)
 *   catppuccin ← Catppuccin.catppuccin-vsc    (Latte / Mocha)
 *   everforest ← sainnhe.everforest
 *   solarized  ← ryanolsonx.solarized
 *
 * Re-convert marketplace forks from the upstream extension rather than
 * hand-editing hexes; hand edits drift from upstream silently and can't be
 * re-derived. `semantier` is the first-party Asimov adapter, not a marketplace fork.
 */

import { THEME_PRESET_PALETTES } from '@hermes/shared'

import { mixOklab } from './color'
import { skinToDesktopTheme } from './skin'
import type { DesktopTheme, DesktopThemeTypography } from './types'

// Color-emoji fonts to append to every stack as a last resort. None of the UI
// text/mono fonts carry emoji glyphs, so without this emoji render as tofu
// boxes on platforms whose default text font lacks them (e.g. Linux/#40364).
// Covers macOS, Windows, Linux, plus the `emoji` generic for anything else.
export const EMOJI_FALLBACK = '"Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji", emoji'

const SYSTEM_SANS =
  '"Segoe WPC", "Segoe UI", -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", system-ui, sans-serif, ' +
  EMOJI_FALLBACK

// Keep the bundled JetBrains Mono first in every mono fallback chain. Several
// theme display fonts (Courier Prime, IBM Plex Mono) do not cover Vietnamese /
// Latin Extended glyphs; falling directly to platform UI fonts makes code
// blocks visibly non-monospace. JetBrains Mono is bundled with the app and
// covers those glyphs, while the Linux fallbacks catch distros without it.
const SYSTEM_MONO =
  '"JetBrains Mono", "Cascadia Code", "Cascadia Mono", "DejaVu Sans Mono", "Liberation Mono", "Noto Sans Mono", ' +
  '"Noto Mono", "SF Mono", ui-monospace, Menlo, Monaco, Consolas, monospace, ' +
  EMOJI_FALLBACK

export const DEFAULT_TYPOGRAPHY: DesktopThemeTypography = { fontSans: SYSTEM_SANS, fontMono: SYSTEM_MONO }

/**
 * GitHub — the upstream palette, unmodified.
 *
 * `nous` is a fork of this with its own accent, so shipping both keeps the
 * original available on its own terms instead of only existing as the thing
 * nous diverged from. Everything but the accent family is identical between
 * them; separate presets are what let nous's accent move without silently
 * redefining what "GitHub" means.
 */
export const githubTheme: DesktopTheme = {
  name: 'github',
  label: 'GitHub',
  description: 'GitHub Light Default and Dark Default',
  ...THEME_PRESET_PALETTES.github,
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: SYSTEM_MONO,
    fontUrl: 'https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap'
  },
  terminal: {
    foreground: '#1f2328',
    black: '#24292f',
    red: '#cf222e',
    green: '#116329',
    yellow: '#4d2d00',
    blue: '#0969da',
    magenta: '#8250df',
    cyan: '#1b7c83',
    white: '#6e7781',
    brightBlack: '#57606a',
    brightRed: '#a40e26',
    brightGreen: '#1a7f37',
    brightYellow: '#633c01',
    brightBlue: '#218bff',
    brightMagenta: '#a475f9',
    brightCyan: '#3192aa',
    brightWhite: '#8c959f'
  },
  darkTerminal: {
    foreground: '#e6edf3',
    black: '#484f58',
    red: '#ff7b72',
    green: '#3fb950',
    yellow: '#d29922',
    blue: '#58a6ff',
    magenta: '#bc8cff',
    cyan: '#39c5cf',
    white: '#b1bac4',
    brightBlack: '#6e7681',
    brightRed: '#ffa198',
    brightGreen: '#56d364',
    brightYellow: '#e3b341',
    brightBlue: '#79c0ff',
    brightMagenta: '#d2a8ff',
    brightCyan: '#56d4dd',
    brightWhite: '#ffffff'
  }
}

/** Optional legacy Nous palette: retain GitHub neutral chrome and the persisted
 * `nous` identity, with its existing Asimov Action Yellow accent.
 * Light text/focus uses the contrast-safe agency role, not bright yellow. */
const ACTION_YELLOW = '#e8c92d'
const ACTION_INK = '#1a2028'

export const nousTheme: DesktopTheme = {
  name: 'nous',
  label: 'Nous',
  description: 'GitHub chrome, Asimov action-yellow accent',
  colors: {
    ...THEME_PRESET_PALETTES.nous.colors,
    primary: ACTION_YELLOW,
    primaryForeground: ACTION_INK,
    secondary: mixOklab(ACTION_YELLOW, THEME_PRESET_PALETTES.nous.colors.background, 0.86),
    accent: mixOklab(ACTION_YELLOW, THEME_PRESET_PALETTES.nous.colors.background, 0.88),
    ring: '#665300',
    midground: '#665300',
    midgroundForeground: '#ffffff',
    composerRing: '#665300',
    userBubble: mixOklab(THEME_PRESET_PALETTES.nous.colors.card, ACTION_YELLOW, 0.12)
  },
  darkColors: {
    ...THEME_PRESET_PALETTES.nous.darkColors,
    primary: ACTION_YELLOW,
    primaryForeground: ACTION_INK,
    secondary: mixOklab(ACTION_YELLOW, THEME_PRESET_PALETTES.nous.darkColors.background, 0.72),
    accent: mixOklab(ACTION_YELLOW, THEME_PRESET_PALETTES.nous.darkColors.background, 0.82),
    ring: ACTION_YELLOW,
    midground: ACTION_YELLOW,
    midgroundForeground: ACTION_INK,
    composerRing: ACTION_YELLOW,
    userBubble: mixOklab(THEME_PRESET_PALETTES.nous.darkColors.card, ACTION_YELLOW, 0.18)
  },
  typography: {
    fontSans: SYSTEM_SANS,
    fontMono: SYSTEM_MONO,
    fontUrl: 'https://fonts.googleapis.com/css2?family=Courier+Prime:wght@400;700&display=swap'
  },
  terminal: {
    foreground: '#1f2328',
    black: '#24292f',
    red: '#cf222e',
    green: '#116329',
    yellow: '#4d2d00',
    blue: '#0969da',
    magenta: '#8250df',
    cyan: '#1b7c83',
    white: '#6e7781',
    brightBlack: '#57606a',
    brightRed: '#a40e26',
    brightGreen: '#1a7f37',
    brightYellow: '#633c01',
    brightBlue: '#218bff',
    brightMagenta: '#a475f9',
    brightCyan: '#3192aa',
    brightWhite: '#8c959f'
  },
  darkTerminal: {
    foreground: '#e6edf3',
    black: '#484f58',
    red: '#ff7b72',
    green: '#3fb950',
    yellow: '#d29922',
    blue: '#58a6ff',
    magenta: '#bc8cff',
    cyan: '#39c5cf',
    white: '#b1bac4',
    brightBlack: '#6e7681',
    brightRed: '#ffa198',
    brightGreen: '#56d364',
    brightYellow: '#e3b341',
    brightBlue: '#79c0ff',
    brightMagenta: '#d2a8ff',
    brightCyan: '#56d4dd',
    brightWhite: '#ffffff'
  }
}

/** Catppuccin — Latte in light, Mocha in dark (Catppuccin.catppuccin-vsc). */
export const catppuccinTheme: DesktopTheme = {
  name: 'catppuccin',
  label: 'Catppuccin',
  description: 'Soothing pastels — Latte and Mocha',
  ...THEME_PRESET_PALETTES.catppuccin,
  terminal: {
    foreground: '#4c4f69',
    cursor: '#dc8a78',
    selectionBackground: '#acb0be',
    black: '#5c5f77',
    red: '#d20f39',
    green: '#40a02b',
    yellow: '#df8e1d',
    blue: '#1e66f5',
    magenta: '#ea76cb',
    cyan: '#179299',
    white: '#acb0be',
    brightBlack: '#6c6f85',
    brightRed: '#de293e',
    brightGreen: '#49af3d',
    brightYellow: '#eea02d',
    brightBlue: '#456eff',
    brightMagenta: '#fe85d8',
    brightCyan: '#2d9fa8',
    brightWhite: '#bcc0cc'
  },
  darkTerminal: {
    foreground: '#cdd6f4',
    cursor: '#f5e0dc',
    selectionBackground: '#585b70',
    black: '#45475a',
    red: '#f38ba8',
    green: '#a6e3a1',
    yellow: '#f9e2af',
    blue: '#89b4fa',
    magenta: '#f5c2e7',
    cyan: '#94e2d5',
    white: '#a6adc8',
    brightBlack: '#585b70',
    brightRed: '#f37799',
    brightGreen: '#89d88b',
    brightYellow: '#ebd391',
    brightBlue: '#74a8fc',
    brightMagenta: '#f2aede',
    brightCyan: '#6bd7ca',
    brightWhite: '#bac2de'
  }
}

/** Everforest — warm, low-contrast forest greens (sainnhe.everforest). */
export const everforestTheme: DesktopTheme = {
  name: 'everforest',
  label: 'Everforest',
  description: 'Warm, low-contrast forest greens',
  ...THEME_PRESET_PALETTES.everforest,
  terminal: {
    foreground: '#5c6a72',
    cursor: '#5c6a72',
    black: '#5c6a72',
    red: '#f85552',
    green: '#8da101',
    yellow: '#dfa000',
    blue: '#3a94c5',
    magenta: '#df69ba',
    cyan: '#35a77c',
    white: '#939f91',
    brightBlack: '#5c6a72',
    brightRed: '#f85552',
    brightGreen: '#8da101',
    brightYellow: '#dfa000',
    brightBlue: '#3a94c5',
    brightMagenta: '#df69ba',
    brightCyan: '#35a77c',
    brightWhite: '#f4f0d9'
  },
  darkTerminal: {
    foreground: '#d3c6aa',
    cursor: '#d3c6aa',
    black: '#343f44',
    red: '#e67e80',
    green: '#a7c080',
    yellow: '#dbbc7f',
    blue: '#7fbbb3',
    magenta: '#d699b6',
    cyan: '#83c092',
    white: '#d3c6aa',
    brightBlack: '#859289',
    brightRed: '#e67e80',
    brightGreen: '#a7c080',
    brightYellow: '#dbbc7f',
    brightBlue: '#7fbbb3',
    brightMagenta: '#d699b6',
    brightCyan: '#83c092',
    brightWhite: '#d3c6aa'
  }
}

/** Solarized — Ethan Schoonover's fixed-contrast pair (ryanolsonx.solarized). */
export const solarizedTheme: DesktopTheme = {
  name: 'solarized',
  label: 'Solarized',
  description: 'Fixed-contrast light and dark',
  ...THEME_PRESET_PALETTES.solarized,
  terminal: {
    foreground: '#657b83',
    black: '#657b83',
    red: '#dc322f',
    green: '#859900',
    yellow: '#b58900',
    blue: '#268bd2',
    magenta: '#d33682',
    cyan: '#2aa198',
    white: '#eee8d5',
    brightBlack: '#657b83',
    brightRed: '#cb4b16',
    brightGreen: '#859900',
    brightYellow: '#657b83',
    brightBlue: '#839496',
    brightMagenta: '#6c71c4',
    brightCyan: '#93a1a1',
    brightWhite: '#eee8d5'
  },
  darkTerminal: {
    foreground: '#839496',
    cursor: '#ffffff',
    selectionBackground: '#ffffff40',
    black: '#14181d',
    red: '#dc322f',
    green: '#859900',
    yellow: '#b58900',
    blue: '#268bd2',
    magenta: '#d33682',
    cyan: '#2aa198',
    white: '#e5e5e5',
    brightBlack: '#676767',
    brightRed: '#dc322f',
    brightGreen: '#859900',
    brightYellow: '#b58900',
    brightBlue: '#268bd2',
    brightMagenta: '#d33682',
    brightCyan: '#2aa198',
    brightWhite: '#e5e5e5'
  }
}

/** Native adapter of docs/operational/design.md §2–3. Palette values live here;
 * styles.css maps the adapter into the desktop's existing surface roles. */
export const semantierTheme: DesktopTheme = {
  name: 'semantier',
  label: 'Semantier',
  description: 'Asimov — neutral instruments, yellow actions, green intelligence',
  colors: {
    background: '#f3f4f5',
    foreground: '#1a1b1e',
    card: '#ffffff',
    cardForeground: '#1a1b1e',
    muted: '#ebebec',
    mutedForeground: '#43474e',
    popover: '#ffffff',
    popoverForeground: '#1a1b1e',
    primary: ACTION_YELLOW,
    primaryForeground: ACTION_INK,
    secondary: '#dde0e3',
    secondaryForeground: '#1a1b1e',
    accent: '#ebebec',
    accentForeground: '#1a1b1e',
    border: '#c2c7ce',
    input: '#72787e',
    ring: '#665300',
    midground: '#665300',
    midgroundForeground: '#ffffff',
    composerRing: '#665300',
    destructive: '#c43418',
    destructiveForeground: '#ffffff',
    sidebarBackground: '#f8f9fa',
    sidebarBorder: '#c2c7ce',
    userBubble: '#ffffff',
    userBubbleBorder: '#c2c7ce'
  },
  darkColors: {
    background: '#1a1b1e',
    foreground: '#ffffff',
    card: '#191c1d',
    cardForeground: '#ffffff',
    muted: '#2e3132',
    mutedForeground: '#c2c7ce',
    popover: '#191c1d',
    popoverForeground: '#ffffff',
    primary: ACTION_YELLOW,
    primaryForeground: ACTION_INK,
    secondary: '#3a3e40',
    secondaryForeground: '#ffffff',
    accent: '#2e3132',
    accentForeground: '#ffffff',
    border: '#43474e',
    input: '#72787e',
    ring: ACTION_YELLOW,
    midground: ACTION_YELLOW,
    midgroundForeground: ACTION_INK,
    composerRing: ACTION_YELLOW,
    destructive: '#e05a3f',
    destructiveForeground: '#1a1b1e',
    sidebarBackground: '#191c1d',
    sidebarBorder: '#43474e',
    userBubble: '#191c1d',
    userBubbleBorder: '#43474e'
  },
  typography: {
    fontSans: '"Manrope", "Noto Sans SC", "Microsoft YaHei", "PingFang SC", ' + SYSTEM_SANS,
    fontMono: SYSTEM_MONO
  },
  terminal: {
    foreground: '#1a1b1e',
    cursor: '#665300',
    selectionBackground: '#e8c92d55',
    black: '#1a1b1e',
    red: '#c43418',
    green: '#26776d',
    yellow: '#665300',
    blue: '#2b6c92',
    magenta: '#765292',
    cyan: '#267180',
    white: '#5e646a',
    brightBlack: '#43474e',
    brightRed: '#b12b14',
    brightGreen: '#20675e',
    brightYellow: '#7d5000',
    brightBlue: '#245d7e',
    brightMagenta: '#65447f',
    brightCyan: '#205f6b',
    brightWhite: '#72787e'
  },
  darkTerminal: {
    foreground: '#ffffff',
    cursor: ACTION_YELLOW,
    selectionBackground: '#e8c92d44',
    black: '#72787e',
    red: '#e05a3f',
    green: '#53b8a7',
    yellow: '#e8a020',
    blue: '#6fb5e0',
    magenta: '#ba9ada',
    cyan: '#74c2cd',
    white: '#c2c7ce',
    brightBlack: '#a3a9b0',
    brightRed: '#f08067',
    brightGreen: '#70d0be',
    brightYellow: ACTION_YELLOW,
    brightBlue: '#92c9ea',
    brightMagenta: '#d2b5eb',
    brightCyan: '#94d6dd',
    brightWhite: '#ffffff'
  }
}

// The converter's input: the keys it reads from `hermes_cli/skin_engine.py`'s
// `default` skin. Light is that skin's `light_colors` overlay (its goldenrod
// ladder for white backgrounds; `banner_border` is not overridden there) with
// one change: the ink. The CLI's `#5C4718` is a single-weight terminal color,
// but Desktop sets secondary and tertiary text at 74% / 54% of it, which left
// them at 4.2:1 / 2.7:1. A deeper ink of the same brown gives 6.5:1 / 3.5:1,
// the levels Nous Light has.
const CLASSIC_DARK_SKIN_COLORS = {
  status_bar_bg: '#1a1a2e',
  banner_text: '#FFF8DC',
  ui_accent: '#FFBF00',
  banner_border: '#CD7F32',
  banner_dim: '#B8860B',
  ui_error: '#ef5350',
  completion_menu_bg: '#1a1a2e'
}

const CLASSIC_LIGHT_SKIN_COLORS = {
  status_bar_bg: '#F5F5F5',
  banner_text: '#2B2109',
  ui_accent: '#D89B04',
  banner_border: '#CD7F32',
  banner_dim: '#B8860B',
  ui_error: '#C62828',
  completion_menu_bg: '#F5F5F5'
}

const classicPalette = (colors: Record<string, string>) => skinToDesktopTheme({ name: 'classic', colors })!.colors

/**
 * Classic Hermes — the CLI's stock `default` skin (gold on navy), offered as a
 * Desktop pick under its OWN id. Never `default`: stock config ships
 * `display.skin: default`, and Desktop reads that as "no pick" (→ semantier), so a
 * user only ever gets Classic by choosing it (#76579). Both palettes go through
 * the converter a backend skin takes, so each mode paints what the CLI paints
 * in a dark or light terminal.
 */
export const classicTheme: DesktopTheme = {
  name: 'classic',
  label: 'Classic Hermes',
  description: "Gold on navy, the CLI's original look",
  colors: classicPalette(CLASSIC_LIGHT_SKIN_COLORS),
  darkColors: classicPalette(CLASSIC_DARK_SKIN_COLORS)
}

/**
 * Midnight — deep blue-violet, near-monotone. Dark only: it has no light
 * palette because the whole idea is the dark end of the spectrum.
 */
export const midnightTheme: DesktopTheme = {
  name: 'midnight',
  label: 'Midnight',
  description: 'Deep blue-violet with cool accents',
  ...THEME_PRESET_PALETTES.midnight,
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`,
    fontUrl: 'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap'
  }
}

export const emberTheme: DesktopTheme = {
  name: 'ember',
  label: 'Ember',
  description: 'Warm crimson and bronze — forge vibes',
  ...THEME_PRESET_PALETTES.ember,
  typography: {
    fontMono: `"IBM Plex Mono", ${SYSTEM_MONO}`,
    fontUrl: 'https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap'
  }
}

/** Clean grayscale. Matches the CLI mono skin and dashboard mono theme. */
export const monoTheme: DesktopTheme = {
  name: 'mono',
  label: 'Mono',
  description: 'Clean grayscale — minimal and focused',
  ...THEME_PRESET_PALETTES.mono
}

/** Neon green on black. Matches the CLI cyberpunk skin and dashboard theme. */
export const cyberpunkTheme: DesktopTheme = {
  name: 'cyberpunk',
  label: 'Cyberpunk',
  description: 'Neon green on black — matrix terminal',
  ...THEME_PRESET_PALETTES.cyberpunk,
  typography: {
    fontMono: `"Courier New", Courier, ${SYSTEM_MONO}`,
    fontSans: `"Courier New", Courier, ${SYSTEM_MONO}`
  }
}

/** Cool slate blue for developers. Matches the CLI slate skin. */
export const slateTheme: DesktopTheme = {
  name: 'slate',
  label: 'Slate',
  description: 'Cool slate blue — focused developer theme',
  ...THEME_PRESET_PALETTES.slate,
  typography: {
    fontMono: `"JetBrains Mono", ${SYSTEM_MONO}`
  }
}

export const BUILTIN_THEMES: Record<string, DesktopTheme> = {
  semantier: semantierTheme,
  nous: nousTheme,
  github: githubTheme,
  catppuccin: catppuccinTheme,
  everforest: everforestTheme,
  solarized: solarizedTheme,
  classic: classicTheme,
  midnight: midnightTheme,
  ember: emberTheme,
  mono: monoTheme,
  slate: slateTheme,
  cyberpunk: cyberpunkTheme
}

export const BUILTIN_THEME_LIST = Object.values(BUILTIN_THEMES)

/** Skin used when nothing is persisted or the persisted name is retired. */
export const DEFAULT_SKIN_NAME = 'semantier'

/** Names that no longer resolve to a skin of their own. A stored pick of one
 *  falls back to DEFAULT_SKIN_NAME, and a cached backend theme under one is
 *  dropped (the reverted #130015 build cached the CLI `default` skin as a
 *  second "Classic Hermes"). */
export const RETIRED_SKINS = new Set(['nous-alt', 'nous-light', 'default', 'gold'])
