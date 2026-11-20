/** Renderer-only fixture: not native IPC, live registry or installation proof. */
import '@/styles.css'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import { createRoot } from 'react-dom/client'

import { MarketplaceTab } from '@/app/capabilities/skills/marketplace-tab'
import { Button } from '@/components/ui/button'
import type { HermesApiRequest } from '@/global'
import { I18nProvider } from '@/i18n'
import { ThemeProvider } from '@/themes/context'
Object.assign(window, {
  hermesDesktop: {
    api: async (request: HermesApiRequest) => {
      const url = new URL(request.path, location.origin)

      if (url.pathname.endsWith('/sources')) {
        return { sources: [], installed: {}, featured: [] }
      }

      if (url.pathname.endsWith('/search')) {
        const source = url.searchParams.get('source')

        if (source === 'semantier') {
          return {
            results: [],
            installed: {},
            timed_out: [],
            errors: [{ source, code: 'unavailable', message: 'Fixture registry unavailable', retryable: true }]
          }
        }

        return {
          results: [
            {
              name: 'Fixture third-party skill',
              description: 'Public fixture',
              source: 'official',
              identifier: 'official/demo',
              trust_level: 'builtin',
              repo: 'NousResearch/hermes-agent',
              tags: []
            }
          ],
          installed: {},
          timed_out: []
        }
      }

      if (url.pathname.endsWith('/preview')) {
        return { name: 'Fixture third-party skill', skill_md: '# Fixture preview', files: ['SKILL.md'], source: 'url' }
      }

      throw new Error('Fixture forbids mutations')
    }
  }
})
const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

function Journey() {
  const [profile, setProfile] = useState('fixture-a')

  return (
    <main className="flex h-screen flex-col p-4">
      <h1>Marketplace renderer fixture</h1>
      <Button onClick={() => setProfile(p => (p === 'fixture-a' ? 'fixture-b' : 'fixture-a'))}>
        Switch fixture profile
      </Button>
      <p>{profile}</p>
      <MarketplaceTab profile={{ connectionId: 'fixture', profile }} query="" />
    </main>
  )
}

createRoot(document.getElementById('root')!).render(
  <QueryClientProvider client={client}>
    <I18nProvider configClient={null} initialLocale="en">
      <ThemeProvider>
        <Journey />
      </ThemeProvider>
    </I18nProvider>
  </QueryClientProvider>
)
