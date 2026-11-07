import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import type { HermesConfigRecord } from '@/hermes'
import { I18nProvider, TRANSLATIONS, useI18n } from '@/i18n'
import { browseBackward, isBrowsingHistory, resetBrowseState } from '@/store/composer-input-history'

import { useComposerPlaceholder } from './use-composer-placeholder'

afterEach(cleanup)

it('reconciles a resting starter when the saved Chinese locale loads', async () => {
  let resolveConfig!: (config: HermesConfigRecord) => void

  const configClient = {
    getConfig: () =>
      new Promise<HermesConfigRecord>(resolve => {
        resolveConfig = resolve
      }),
    saveConfig: async () => ({ ok: true })
  }

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <I18nProvider configClient={configClient} initialLocale="en">
      {children}
    </I18nProvider>
  )

  const { result } = renderHook(
    () => ({
      locale: useI18n().locale,
      placeholder: useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null })
    }),
    { wrapper }
  )

  expect(TRANSLATIONS.en.composer.newSessionPlaceholders).toContain(result.current.placeholder)

  await act(async () => {
    resolveConfig({ display: { language: 'zh' } } as HermesConfigRecord)
  })
  await waitFor(() => expect(result.current.locale).toBe('zh'))
  await waitFor(() => expect(TRANSLATIONS.zh.composer.newSessionPlaceholders).toContain(result.current.placeholder))
})

it('keeps the starter through session persistence and changes it for an explicit language choice', async () => {
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <I18nProvider configClient={null} initialLocale="en">
      {children}
    </I18nProvider>
  )

  const { result, rerender } = renderHook(
    ({ sessionId }: { sessionId: string | null }) => ({
      i18n: useI18n(),
      placeholder: useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId })
    }),
    { initialProps: { sessionId: null as string | null }, wrapper }
  )

  const starter = result.current.placeholder
  rerender({ sessionId: 'persisted-session' })
  expect(result.current.placeholder).toBe(starter)

  await act(async () => {
    await result.current.i18n.setLocale('zh')
  })
  expect(TRANSLATIONS.zh.composer.followUpPlaceholders).toContain(result.current.placeholder)
})

it('clears the previous history cursor when locale and conversation change together', async () => {
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <I18nProvider configClient={null} initialLocale="en">
      {children}
    </I18nProvider>
  )

  const { result, rerender } = renderHook(
    ({ sessionId }: { sessionId: string }) => ({
      i18n: useI18n(),
      placeholder: useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId })
    }),
    { initialProps: { sessionId: 'previous-session' }, wrapper }
  )

  browseBackward('previous-session', 'draft', ['older message'])
  expect(isBrowsingHistory('previous-session')).toBe(true)

  await act(async () => {
    rerender({ sessionId: 'next-session' })
    await result.current.i18n.setLocale('zh')
  })

  expect(isBrowsingHistory('previous-session')).toBe(false)
  expect(TRANSLATIONS.zh.composer.followUpPlaceholders).toContain(result.current.placeholder)
  resetBrowseState('next-session')
})
