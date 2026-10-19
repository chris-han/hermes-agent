import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { I18nProvider, useI18n } from '@/i18n'
import { en } from '@/i18n/en'
import { zh } from '@/i18n/zh'

import { desktopBrandingName } from '../../../electron/desktop-branding-types'

import { CustomizeSettings } from './customize-settings'
const native = vi.hoisted(() => ({ save: vi.fn(), decode: vi.fn() }))
vi.mock('@/store/desktop-branding', async () => {
  const { atom } = await import('nanostores')

  return {
    $desktopBranding: atom({ version: 1, displayName: '', logo: null }),
    $brandingLoadFailed: atom(false),
    loadDesktopBranding: vi.fn(),
    saveDesktopBranding: native.save
  }
})
vi.mock('@/lib/branding-image', () => ({ decodeBrandingImage: native.decode }))
import { $desktopBranding } from '@/store/desktop-branding'
const original = window.hermesDesktop

function LocaleSwitch() {
  const { setLocale } = useI18n()
  return <button onClick={() => void setLocale('en')}>Switch locale</button>
}

function mount(locale = 'en') {
  window.hermesDesktop = {
    ...original,
    branding: {
      initial: { version: 1, displayName: '', logo: null },
      get: vi.fn(),
      set: vi.fn(),
      onChanged: () => () => {}
    }
  }

  return render(
    <I18nProvider configClient={null} initialLocale={locale}>
      <CustomizeSettings />
      <LocaleSwitch />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  window.hermesDesktop = original
  $desktopBranding.set({ version: 1, displayName: '', logo: null })
  vi.clearAllMocks()
})
test('eight previews leave committed state alone; failed Apply retains editable draft; successful Apply commits', async () => {
  mount()
  fireEvent.change(screen.getByLabelText(en.branding.nameEn), { target: { value: 'Acme' } })
  expect($desktopBranding.get().displayName).toBe('')
  const nav = screen.getByRole('navigation', { name: en.branding.placements })
  expect(nav.querySelectorAll('button')).toHaveLength(8)

  for (const button of nav.querySelectorAll('button')) {
    fireEvent.click(button)
    expect(screen.getByRole('region', { name: en.branding.preview }).textContent).toContain('Acme')
  }

  native.save.mockRejectedValueOnce(new Error('disk full'))
  fireEvent.click(screen.getByRole('button', { name: en.branding.apply }))
  await screen.findByText(en.branding.saveFailed)
  expect((screen.getByLabelText(en.branding.nameEn) as HTMLInputElement).value).toBe('Acme')
  native.save.mockImplementation(async value => {
    $desktopBranding.set(value)

    return value
  })
  fireEvent.click(screen.getByRole('button', { name: en.branding.apply }))
  await screen.findByText(en.branding.saved)
  expect($desktopBranding.get().displayName).toBe('Acme')
})
test('older upload completion cannot overwrite a newer upload, removal or reset; invalid file retains prior logo', async () => {
  mount()
  const finish: ((logo: string) => void)[] = []
  native.decode.mockImplementation(() => new Promise(resolve => finish.push(resolve)))
  const input = screen.getByLabelText(en.branding.logo)
  const choose = () => fireEvent.change(input, { target: { files: [new File(['raster'], 'logo.png')] } })
  choose()
  choose()
  await act(async () => finish[1]('data:image/png;base64,new'))
  await act(async () => finish[0]('data:image/png;base64,old'))
  expect(document.querySelector('img[src$="new"]')).toBeTruthy()
  expect(document.querySelector('img[src$="old"]')).toBeNull()
  native.decode.mockRejectedValueOnce(new Error('invalid'))
  choose()
  await screen.findByText(en.branding.invalidImage)
  expect(document.querySelector('img[src$="new"]')).toBeTruthy()
  choose()
  fireEvent.click(screen.getByRole('button', { name: en.branding.remove }))
  await act(async () => finish[2]('data:image/png;base64,stale'))
  expect(document.querySelector('img[src$="stale"]')).toBeNull()
  choose()
  fireEvent.click(screen.getByRole('button', { name: en.branding.reset }))
  await act(async () => finish[3]('data:image/png;base64,stale'))
  expect(document.querySelector('img[src$="stale"]')).toBeNull()
})
test('Chinese uses one locale, nonblocking unavailable organization and localized attribution', async () => {
  mount('zh')
  expect(screen.queryByText(en.branding.title)).toBeNull()
  expect(screen.getAllByText(zh.branding.attribution).length).toBeGreaterThan(0)
  expect(screen.getByText(zh.branding.unbound)).toBeTruthy()
  expect((screen.getByRole('button', { name: zh.branding.apply }) as HTMLButtonElement).disabled).toBe(false)
  fireEvent.change(screen.getByLabelText(zh.branding.nameEn), { target: { value: '用户名称' } })
  act(() => $desktopBranding.set({ version: 1, displayName: 'Peer', logo: null }))
  await waitFor(() => expect((screen.getByLabelText(zh.branding.nameEn) as HTMLInputElement).value).toBe('用户名称'))
})

test('separate Chinese and English fields select one preview name per locale without losing either draft', async () => {
  mount('zh')
  fireEvent.change(screen.getByLabelText('组织显示名称（中文）'), { target: { value: '示例集团' } })
  fireEvent.change(screen.getByLabelText('组织显示名称（英文）'), { target: { value: 'Acme Global' } })
  const preview = () => screen.getByRole('region', { name: zh.branding.preview })
  expect(preview().textContent).toContain('示例集团')
  expect(preview().textContent).not.toContain('Acme Global')
  fireEvent.click(screen.getByRole('button', { name: 'Switch locale' }))
  await waitFor(() => expect(screen.getByRole('region', { name: en.branding.preview }).textContent).toContain('Acme Global'))
  expect((screen.getByLabelText('Organization display name (Chinese)') as HTMLInputElement).value).toBe('示例集团')
  expect((screen.getByLabelText('Organization display name (English)') as HTMLInputElement).value).toBe('Acme Global')
  native.save.mockImplementation(async value => { $desktopBranding.set(value); return value })
  fireEvent.click(screen.getByRole('button', { name: en.branding.apply }))
  await screen.findByText(en.branding.saved)
  expect($desktopBranding.get()).toMatchObject({ displayName: 'Acme Global', displayNameZh: '示例集团' })
})

 test('initial and reset previews use localized Semantier fallback', async () => {
  mount('zh')
  const preview = screen.getByRole('region', { name: zh.branding.preview })
  expect(preview.querySelector('[data-organization-brand]')?.textContent).toContain('司序科技')
  expect(preview.querySelector('img')?.getAttribute('src')).toContain('semantier-logo.png')
  fireEvent.click(screen.getByRole('button', { name: 'Switch locale' }))
  await waitFor(() => expect(screen.getByRole('region', { name: en.branding.preview }).querySelector('[data-organization-brand]')?.textContent).toContain('Semantier'))
  fireEvent.change(screen.getByLabelText('Organization display name (English)'), { target: { value: 'Acme' } })
  fireEvent.click(screen.getByRole('button', { name: en.branding.reset }))
  expect(screen.getByRole('region', { name: en.branding.preview }).querySelector('[data-organization-brand]')?.textContent).toContain('Semantier')
})

test.each([
  ['zh', '', '', '司序科技'], ['zh-CN', 'Acme', '示例', '示例'],
  ['en', 'Acme', '示例', 'Acme'], ['fr', '', '示例', '示例'],
  ['zh-TW', 'Legacy', '', 'Legacy'], ['en', ' ', ' ', 'Semantier']
])('name fallback for %s / %s / %s', (locale, displayName, displayNameZh, expected) => {
  expect(desktopBrandingName({version: 1, displayName, displayNameZh, logo: null}, locale)).toBe(expected)
})
