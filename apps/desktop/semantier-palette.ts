/** Pure native theme data, shared by the renderer and asset-independent recovery. */
// Color-emoji fonts to append to every stack as a last resort. None of the UI
// text/mono fonts carry emoji glyphs, so without this emoji render as tofu
// boxes on platforms whose default text font lacks them (e.g. Linux/#40364).
// Covers macOS, Windows, Linux, plus the `emoji` generic for anything else.
export const EMOJI_FALLBACK = '"Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji", emoji'

export const SYSTEM_SANS =
  '"Segoe WPC", "Segoe UI", -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", system-ui, sans-serif, ' +
  EMOJI_FALLBACK

// Keep the bundled JetBrains Mono first in every mono fallback chain. Several
// theme display fonts (Courier Prime, IBM Plex Mono) do not cover Vietnamese /
// Latin Extended glyphs; falling directly to platform UI fonts makes code
// blocks visibly non-monospace. JetBrains Mono is bundled with the app and
// covers those glyphs, while the Linux fallbacks catch distros without it.
export const SYSTEM_MONO =
  '"JetBrains Mono", "Cascadia Code", "Cascadia Mono", "DejaVu Sans Mono", "Liberation Mono", "Noto Sans Mono", ' +
  '"Noto Mono", "SF Mono", ui-monospace, Menlo, Monaco, Consolas, monospace, ' +
  EMOJI_FALLBACK

export const ACTION_YELLOW = '#e8c92d'
export const ACTION_INK = '#1a2028'

export const SEMANTIER_FONT_SANS = '"Manrope", "Noto Sans SC", "Microsoft YaHei", "PingFang SC", ' + SYSTEM_SANS

export const SEMANTIER_LIGHT_COLORS = {
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
  }

export const SEMANTIER_DARK_COLORS = {
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
  }
