import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesConfigRecord } from '@/hermes'

import { TRANSLATIONS } from './catalog'
import { type I18nConfigClient, I18nProvider, useI18n, withConfigDisplayLanguage } from './context'
import { registerAppLocale } from './registry'
import { $requestedLocale } from './runtime'
import type { Locale } from './types'

function LanguageProbe({ target = 'zh' }: { target?: Locale }) {
  const { isLoadingConfig, isSavingLocale, locale, saveError, setLocale, t } = useI18n()

  return (
    <div>
      <p data-testid="locale">{locale}</p>
      <p data-testid="label">{t.language.label}</p>
      <p data-testid="save">{t.common.save}</p>
      <p data-testid="loading">{String(isLoadingConfig)}</p>
      <p data-testid="saving">{String(isSavingLocale)}</p>
      <p data-testid="save-error">{saveError?.message ?? ''}</p>
      <button onClick={() => void setLocale(target).catch(() => undefined)} type="button">
        switch
      </button>
    </div>
  )
}

describe('I18nProvider', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('defaults to English without a config client', () => {
    render(
      <I18nProvider configClient={null}>
        <LanguageProbe />
      </I18nProvider>
    )

    expect(screen.getByTestId('locale').textContent).toBe('en')
    expect(screen.getByTestId('label').textContent).toBe('Language')
  })

  it.each([undefined, 'invalid'])(
    'defaults Windows unset/invalid language %s to Chinese without saving',
    async saved => {
      const original = window.hermesDesktop
      Object.defineProperty(window, 'hermesDesktop', {
        configurable: true,
        value: {
          getMachineProfile: vi.fn().mockResolvedValue({ platform: 'win32', locale: 'en-US' })
        }
      })

      const configClient: I18nConfigClient = {
        getConfig: vi.fn().mockResolvedValue({ display: { language: saved } }),
        saveConfig: vi.fn()
      }

      try {
        render(
          <I18nProvider configClient={configClient}>
            <LanguageProbe />
          </I18nProvider>
        )
        await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('zh'))
        expect(configClient.saveConfig).not.toHaveBeenCalled()
      } finally {
        Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: original })
      }
    }
  )

  it('normalizes an initial locale alias and switches translations', async () => {
    render(
      <I18nProvider configClient={null} initialLocale="zh-CN">
        <LanguageProbe target="en" />
      </I18nProvider>
    )

    expect(screen.getByTestId('locale').textContent).toBe('zh')
    expect(screen.getByTestId('label').textContent).toBe('语言')

    fireEvent.click(screen.getByRole('button', { name: 'switch' }))

    await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('en'))
    expect(screen.getByTestId('label').textContent).toBe('Language')
  })

  it('loads the initial locale from display.language config', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue({ display: { language: 'zh-Hans' } }),
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))

    expect(screen.getByTestId('locale').textContent).toBe('zh')
    expect(screen.getByTestId('label').textContent).toBe('语言')
    expect(configClient.saveConfig).not.toHaveBeenCalled()
  })

  it.each([
    ['fr', 'fr'],
    ['de-DE', 'de'],
    ['es', 'es']
  ] as const)('loads display.language=%s and renders the %s catalog', async (configured, locale) => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue({ display: { language: configured } }),
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))

    expect(screen.getByTestId('locale').textContent).toBe(locale)
    expect(screen.getByTestId('label').textContent).toBe(TRANSLATIONS[locale].language.label)
    expect(screen.getByTestId('label').textContent).not.toBe(TRANSLATIONS.en.language.label)
    expect(screen.getByTestId('save').textContent).not.toBe(TRANSLATIONS.en.common.save)
    expect(configClient.saveConfig).not.toHaveBeenCalled()
  })

  it('keeps English usable when config loading fails', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockRejectedValue(new Error('config unavailable')),
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient} initialLocale="zh">
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))

    expect(screen.getByTestId('locale').textContent).toBe('en')
    expect(screen.getByTestId('label').textContent).toBe('Language')
    expect(configClient.saveConfig).not.toHaveBeenCalled()
  })

  it('does not overwrite unsupported configured languages', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue({ display: { language: 'it' } }),
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient} initialLocale="zh">
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))

    expect(screen.getByTestId('locale').textContent).toBe('en')
    expect(screen.getByTestId('label').textContent).toBe('Language')
    expect(configClient.saveConfig).not.toHaveBeenCalled()
    // …but remembers the ask so a backend pack for it can be fetched.
    expect($requestedLocale.get()).toBe('it')
  })

  it('promotes a saved pack-only language once its pack registers, then drops back when it is removed', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue({ display: { language: 'pl' } }),
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))
    expect(screen.getByTestId('locale').textContent).toBe('en')

    const dispose = registerAppLocale(
      'pl',
      { endonym: 'Polski', translations: { language: { label: 'Język' } } },
      'backend'
    )

    try {
      await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('pl'))
      expect(screen.getByTestId('label').textContent).toBe('Język')
      // Unregistered keys fall back to English, never to the raw key.
      expect(screen.getByTestId('save').textContent).toBe(TRANSLATIONS.en.common.save)
      // The provider writes <html lang> in a passive effect after the render
      // commits, so it can trail the rendered locale under load.
      await waitFor(() => expect(document.documentElement.lang).toBe('pl'))
      expect(configClient.saveConfig).not.toHaveBeenCalled()
    } finally {
      dispose()
    }

    // The pack is gone (profile switch): strings fall back to English while the
    // chosen id stays put for the next sync.
    await waitFor(() => expect(screen.getByTestId('label').textContent).toBe('Language'))
    expect(screen.getByTestId('locale').textContent).toBe('pl')
  })

  it('mirrors a registered language’s direction onto the document', async () => {
    const dispose = registerAppLocale('he', { endonym: 'עברית', rtl: true }, 'plugin:hermes-lang-he')

    try {
      render(
        <I18nProvider configClient={null} initialLocale="he">
          <LanguageProbe />
        </I18nProvider>
      )

      expect(screen.getByTestId('locale').textContent).toBe('he')
      expect(document.documentElement.dir).toBe('rtl')
    } finally {
      dispose()
    }
  })

  it('reads latest config before saving language and preserves unrelated values', async () => {
    const saveConfig = vi.fn().mockResolvedValue({ ok: true })

    const latestConfig: HermesConfigRecord = {
      display: { language: 'en', skin: 'slate' },
      terminal: { cwd: '/new' }
    }

    const configClient: I18nConfigClient = {
      getConfig: vi
        .fn()
        .mockResolvedValueOnce({ display: { language: 'en', skin: 'mono' }, terminal: { cwd: '/old' } })
        .mockResolvedValueOnce(latestConfig),
      saveConfig
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))
    fireEvent.click(screen.getByRole('button', { name: 'switch' }))

    await waitFor(() => expect(saveConfig).toHaveBeenCalledTimes(1))
    expect(saveConfig).toHaveBeenCalledWith({
      display: { language: 'zh', skin: 'slate' },
      terminal: { cwd: '/new' },
      stt: { language: 'zh', local: { language: 'zh' } },
      tts: { edge: { voice: 'zh-CN-XiaoxiaoNeural' } }
    })
  })

  it('applies RTL direction for Arabic and restores LTR on switch back', async () => {
    render(
      <I18nProvider configClient={null} initialLocale="ar">
        <LanguageProbe target="en" />
      </I18nProvider>
    )

    expect(screen.getByTestId('locale').textContent).toBe('ar')
    expect(document.documentElement.dir).toBe('rtl')
    expect(document.documentElement.lang).toBe('ar')

    fireEvent.click(screen.getByRole('button', { name: 'switch' }))

    await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('en'))
    expect(document.documentElement.dir).toBe('ltr')
    expect(document.documentElement.lang).toBe('en')
  })

  it('rolls back the visible locale when saving fails', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue({ display: { language: 'en' } }),
      saveConfig: vi.fn().mockRejectedValue(new Error('save failed'))
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))
    fireEvent.click(screen.getByRole('button', { name: 'switch' }))

    await waitFor(() => expect(screen.getByTestId('save-error').textContent).toBe('save failed'))

    expect(screen.getByTestId('locale').textContent).toBe('en')
    expect(screen.getByTestId('label').textContent).toBe('Language')
  })

  it('retries a transient config failure and applies the persisted locale', async () => {
    const getConfig = vi
      .fn()
      .mockRejectedValueOnce(new Error('backend not ready yet'))
      .mockResolvedValueOnce({ display: { language: 'zh-Hans' } })

    const configClient: I18nConfigClient = {
      getConfig,
      saveConfig: vi.fn()
    }

    vi.useFakeTimers()

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe />
      </I18nProvider>
    )

    // First attempt fails → settles on English (permanent-failure contract).
    await act(async () => {})
    expect(screen.getByTestId('loading').textContent).toBe('false')
    expect(screen.getByTestId('locale').textContent).toBe('en')

    // The bounded retry succeeds and applies the persisted language.
    await act(async () => {
      vi.advanceTimersByTime(3_000)
    })
    expect(screen.getByTestId('locale').textContent).toBe('zh')
    expect(getConfig).toHaveBeenCalledTimes(2)

    vi.useRealTimers()
  })

  it('stops retrying after the bounded retry budget is exhausted', async () => {
    vi.useFakeTimers()
    const getConfig = vi.fn().mockRejectedValue(new Error('backend unavailable'))

    const configClient: I18nConfigClient = {
      getConfig,
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient} initialLocale="zh">
        <LanguageProbe />
      </I18nProvider>
    )

    // Flush the initial attempt: it fails and settles on English.
    await act(async () => {})
    expect(screen.getByTestId('locale').textContent).toBe('en')
    expect(getConfig).toHaveBeenCalledTimes(1)

    // Budget is 10 retries at 3s each; run the whole budget to completion.
    for (let i = 0; i < 10; i++) {
      await act(async () => {
        vi.advanceTimersByTime(3_000)
      })
    }

    expect(getConfig).toHaveBeenCalledTimes(11)

    // No timer is left after the budget is spent — nothing fires later.
    await act(async () => {
      vi.advanceTimersByTime(30_000)
    })
    expect(getConfig).toHaveBeenCalledTimes(11)

    vi.useRealTimers()
  })

  it('stops retrying once the provider unmounts mid-retry', async () => {
    vi.useFakeTimers()
    const getConfig = vi.fn().mockRejectedValue(new Error('backend not ready yet'))

    const view = render(
      <I18nProvider configClient={{ getConfig, saveConfig: vi.fn() }}>
        <LanguageProbe />
      </I18nProvider>
    )

    await act(async () => {})
    expect(getConfig).toHaveBeenCalledTimes(1)

    // A retry is now scheduled; unmounting must cancel it, not keep polling a
    // backend nobody is listening for.
    view.unmount()

    await act(async () => {
      vi.advanceTimersByTime(60_000)
    })
    expect(getConfig).toHaveBeenCalledTimes(1)

    vi.useRealTimers()
  })

  it('a late startup read never overrides a language the user picked mid-retry', async () => {
    vi.useFakeTimers()

    const getConfig = vi
      .fn()
      .mockRejectedValueOnce(new Error('backend not ready yet'))
      .mockResolvedValue({ display: { language: 'en' } })

    const configClient: I18nConfigClient = {
      getConfig,
      saveConfig: vi.fn().mockResolvedValue({ ok: true })
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe target="ja" />
      </I18nProvider>
    )

    await act(async () => {})
    expect(screen.getByTestId('locale').textContent).toBe('en')

    // User picks Japanese while the startup retry is still pending.
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'switch' }))
    })
    expect(screen.getByTestId('locale').textContent).toBe('ja')

    // The retry resolves with the stale on-disk value; the explicit pick wins.
    await act(async () => {
      vi.advanceTimersByTime(3_000)
    })
    expect(screen.getByTestId('locale').textContent).toBe('ja')

    vi.useRealTimers()
  })
})

describe('language-linked voice defaults', () => {
  it.each([
    ['en', 'en', 'en-US-AriaNeural'],
    ['zh', 'zh', 'zh-CN-XiaoxiaoNeural'],
    ['zh-hant', 'zh', 'zh-TW-HsiaoChenNeural'],
    ['ja', 'ja', 'ja-JP-NanamiNeural'],
    ['ar', 'ar', 'ar-SA-ZariyahNeural'],
    ['ru', 'ru', 'ru-RU-SvetlanaNeural'],
    ['fr', 'fr', 'fr-FR-DeniseNeural'],
    ['de', 'de', 'de-DE-KatjaNeural'],
    ['es', 'es', 'es-ES-ElviraNeural']
  ] as const)('maps bundled locale %s to provider-native defaults', (locale, language, voice) => {
    const next = withConfigDisplayLanguage({}, locale)
    expect(next.stt).toEqual({ language, local: { language } })
    expect(next.tts).toEqual({ edge: { voice } })
  })

  it('leaves speech choices unchanged for a plugin language without a known voice mapping', () => {
    expect(withConfigDisplayLanguage({ stt: { language: 'pl' } }, 'pl').stt).toEqual({ language: 'pl' })
  })

  it('sets actual speech settings on sparse config and follows EN/ZH round trips', () => {
    const chinese = withConfigDisplayLanguage({ display: { language: 'en' } }, 'zh')
    expect(chinese.stt).toEqual({ language: 'zh', local: { language: 'zh' } })
    expect(chinese.tts).toEqual({ edge: { voice: 'zh-CN-XiaoxiaoNeural' } })
    const english = withConfigDisplayLanguage(chinese, 'en')
    expect(english.stt).toEqual({ language: 'en', local: { language: 'en' } })
    expect(english.tts).toEqual({ edge: { voice: 'en-US-AriaNeural' } })
  })

  it('preserves a custom language, voice, provider and model without mutating input', () => {
    const original = {
      display: { language: 'en' },
      stt: { language: 'fr', local: { language: 'ja', model: 'small' } },
      tts: { provider: 'piper', edge: { voice: 'en-US-GuyNeural', speed: 1.2 } }
    }
    const snapshot = structuredClone(original)
    const next = withConfigDisplayLanguage(original, 'zh')
    expect(next.stt).toEqual(original.stt)
    expect(next.tts).toEqual(original.tts)
    expect(original).toEqual(snapshot)
  })

  it('maps traditional Chinese UI to a supported Whisper language and Taiwan voice', () => {
    const next = withConfigDisplayLanguage(
      {
        display: { language: 'zh' },
        stt: { local: { language: 'zh' } },
        tts: { edge: { voice: 'zh-CN-XiaoxiaoNeural' } }
      },
      'zh-hant'
    )
    expect(next.stt).toEqual({ language: 'zh', local: { language: 'zh' } })
    expect(next.tts).toEqual({ edge: { voice: 'zh-TW-HsiaoChenNeural' } })
  })
})
