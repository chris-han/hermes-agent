import { isRecord } from '@hermes/shared/i18n'

import type { HermesConfigRecord } from '@/types/hermes'

// App locale → provider-native defaults. Whisper uses zh for both Chinese scripts.
export const LOCALE_VOICE_DEFAULTS: Record<string, { language: string; voice: string }> = {
  en: { language: 'en', voice: 'en-US-AriaNeural' },
  zh: { language: 'zh', voice: 'zh-CN-XiaoxiaoNeural' },
  'zh-hant': { language: 'zh', voice: 'zh-TW-HsiaoChenNeural' },
  ja: { language: 'ja', voice: 'ja-JP-NanamiNeural' },
  ar: { language: 'ar', voice: 'ar-SA-ZariyahNeural' },
  ru: { language: 'ru', voice: 'ru-RU-SvetlanaNeural' },
  fr: { language: 'fr', voice: 'fr-FR-DeniseNeural' },
  de: { language: 'de', voice: 'de-DE-KatjaNeural' },
  es: { language: 'es', voice: 'es-ES-ElviraNeural' }
}

/** Follow a language change only for unset/default speech choices; custom selections win. */
export function withLocaleVoiceDefaults(
  config: HermesConfigRecord,
  previousLocale: string,
  nextLocale: string
): HermesConfigRecord {
  const next = LOCALE_VOICE_DEFAULTS[nextLocale]
  const previous = LOCALE_VOICE_DEFAULTS[previousLocale]

  // Plugin language packs need their own voice mapping; don't assign an unrelated English voice.
  if (!next) {return config}

  const stt = isRecord(config.stt) ? config.stt : {}
  const local = isRecord(stt.local) ? stt.local : {}
  const tts = isRecord(config.tts) ? config.tts : {}
  const edge = isRecord(tts.edge) ? tts.edge : {}

  const follows = (value: unknown, oldDefault: string | undefined) =>
    value == null || value === '' || value === oldDefault

  return {
    ...config,
    stt: {
      ...stt,
      ...(follows(stt.language, previous?.language) ? { language: next.language } : {}),
      local: {
        ...local,
        ...(follows(local.language, previous?.language) ? { language: next.language } : {})
      }
    },
    tts: {
      ...tts,
      edge: { ...edge, ...(follows(edge.voice, previous?.voice) ? { voice: next.voice } : {}) }
    }
  }
}
