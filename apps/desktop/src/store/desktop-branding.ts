import { atom } from 'nanostores'

import { DEFAULT_DESKTOP_BRANDING, type DesktopBranding } from '../../electron/desktop-branding-types'

const bridge = window.hermesDesktop?.branding
export const $desktopBranding = atom<DesktopBranding>(bridge?.initial ?? DEFAULT_DESKTOP_BRANDING)
export const $brandingLoadFailed = atom(false)
let revision = 0
bridge?.onChanged(value => {
  revision++
  $desktopBranding.set(value)
  $brandingLoadFailed.set(false)
})

export async function loadDesktopBranding() {
  if (!bridge) {
    return
  }

  const generation = ++revision

  try {
    const value = await bridge.get()

    if (generation === revision) {
      $desktopBranding.set(value)
      $brandingLoadFailed.set(false)
    }
  } catch {
    if (generation === revision) {
      $brandingLoadFailed.set(true)
    }
  }
}

export async function saveDesktopBranding(value: DesktopBranding) {
  if (!bridge) {
    throw new Error('Native branding unavailable')
  }

  const generation = ++revision
  const saved = await bridge.set(value)

  if (generation === revision) {
    $desktopBranding.set(saved)
  }

  $brandingLoadFailed.set(false)

  return $desktopBranding.get()
}

void loadDesktopBranding()
