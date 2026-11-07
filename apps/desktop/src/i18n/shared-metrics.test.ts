import { describe, expect, it } from 'vitest'

import { TRANSLATIONS } from './catalog'

describe('shared-metrics recipient', () => {
  for (const [locale, { sharedMetrics: copy }] of Object.entries(TRANSLATIONS)) {
    it(`names Semantier in ${locale} sharing choices and explanations`, () => {
      for (const text of [copy.share, copy.stripChoices.share, copy.sendLabel, copy.sendDesc, copy.sending]) {
        expect(text).toContain('Semantier')
        expect(text).not.toContain('Nous')
      }
    })
  }
})
