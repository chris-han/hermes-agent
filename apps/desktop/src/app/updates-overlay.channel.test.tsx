import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import type { DesktopUpdateStatus } from '@/global'
import { I18nProvider } from '@/i18n/context'
import {
  $desktopVersion,
  $updateApply,
  $updateOverlayOpen,
  $updateOverlayTarget,
  $updateStatus,
  resetUpdateApplyState
} from '@/store/updates'

import { UpdatesOverlay } from './updates-overlay'

afterEach((): void => {
  cleanup()
  resetUpdateApplyState()
  $updateOverlayOpen.set(false)
  $updateStatus.set(null)
  $desktopVersion.set(null)
})

async function openWith(status: Partial<DesktopUpdateStatus>, locale: 'en' | 'zh' = 'en'): Promise<void> {
  $desktopVersion.set({
    appVersion: '0.21.6',
    electronVersion: '37',
    hermesRoot: '/h',
    nodeVersion: '22',
    platform: 'linux'
  })
  $updateStatus.set({ behind: 0, mechanism: 'posix-handoff', ...status, supported: true })
  $updateOverlayTarget.set('client')
  $updateOverlayOpen.set(true)
  await act(async (): Promise<void> => {
    render(
      <I18nProvider configClient={null} initialLocale={locale}>
        <UpdatesOverlay />
      </I18nProvider>
    )
  })
}

it('offers Change only where Settings can switch; a custom branch or older runtime gets the label alone', async () => {
  await openWith({ channel: 'stable', channelSelectable: true })
  expect(screen.getByText('Channel')).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Change' })).toBeTruthy()
  cleanup()

  await openWith({ branch: 'feature/gui', channelSelectable: true })
  expect(screen.getByText(/Branch: feature\/gui/)).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Change' })).toBeNull()
  cleanup()

  // An older runtime cannot save a channel, so Settings shows no selector to land on.
  await openWith({ channel: 'stable' })
  expect(screen.getByText('Stable releases')).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Change' })).toBeNull()
})

it.each([
  ['en', 'Complete the pending Windows installation'],
  ['zh', '请先完成待处理的 Windows 安装']
] as const)(
  'explains a real native installation hold in %s without exposing the machine code',
  async (locale, copy): Promise<void> => {
    $updateApply.set({
      ...$updateApply.get(),
      applying: false,
      stage: 'error',
      message: "Error invoking remote method 'hermes:updates:apply': Error: pending_update_installation"
    })
    await openWith({ mechanism: 'app-installer', updateAvailable: true }, locale)
    expect(screen.getByText(new RegExp(copy))).toBeTruthy()
    expect(screen.queryByText(/pending_update_installation/)).toBeNull()
  }
)
