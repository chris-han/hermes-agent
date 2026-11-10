import '../src/styles.css'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router'

import { BrowserRealProfilePanel } from '../src/app/settings/browser-real-profile-panel'
import { ToolsetConfigPanel } from '../src/app/settings/toolset-config-panel'
import { I18nProvider } from '../src/i18n'
import { ThemeProvider } from '../src/themes/context'

const locale = new URLSearchParams(location.search).get('locale') || 'en'
const key = 'v099-browser-choice'
const selected = () => localStorage.getItem(key) || 'Local Browser'
const calls: unknown[] = []
Object.assign(window, { fixtureCalls: calls })
window.hermesDesktop = {
  api: async (request: any) => {
    calls.push(request)

    if (request.path.includes('/provider')) {
      localStorage.setItem(key, request.body.provider)

      return { ok: true, name: 'browser', provider: request.body.provider }
    }

    if (request.path.includes('/toolsets/browser/config')) {
      return {
        name: 'browser',
        has_category: true,
        active_provider: selected(),
        providers: [
          {
            name: 'Local Browser',
            badge: '',
            tag: '',
            env_vars: [],
            is_active: selected() === 'Local Browser',
            status: 'ready'
          },
          {
            name: 'Edge (system)',
            badge: '',
            tag: '',
            env_vars: [],
            is_active: selected() === 'Edge (system)',
            status: 'ready'
          }
        ]
      }
    }

    if (request.path.includes('/config/schema')) {
      return { fields: {}, category_order: [] }
    }

    if (request.path.startsWith('/api/config')) {
      return { browser: { use_real_profile: false }, display: { language: locale } }
    }

    return {}
  }
} as any

createRoot(document.getElementById('root')!).render(
  <MemoryRouter>
    <QueryClientProvider client={new QueryClient()}>
      <I18nProvider configClient={null} initialLocale={locale}>
        <ThemeProvider>
          <main className="mx-auto grid max-w-xl gap-4 p-6">
            <BrowserRealProfilePanel />
            <ToolsetConfigPanel toolset="browser" />
          </main>
        </ThemeProvider>
      </I18nProvider>
    </QueryClientProvider>
  </MemoryRouter>
)
