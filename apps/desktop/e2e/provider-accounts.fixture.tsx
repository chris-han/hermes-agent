import '../src/styles.css'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router'

import { ProvidersSettings } from '../src/app/settings/providers-settings'
import { I18nProvider } from '../src/i18n'
import { ThemeProvider } from '../src/themes/context'

const params = new URLSearchParams(location.search)
const locale = params.get('locale') || 'en'
const scenario = params.get('scenario') || 'connected'
const provider = (id: string, logged_in: boolean) => ({
  id, name: id === 'nous' ? 'Nous Portal' : id === 'semantier-cloud' ? 'Semantier Cloud' : 'MiniMax',
  flow: 'device_code', cli_command: '', docs_url: '', disconnectable: true, status: { logged_in }
})
const providers = scenario === 'nous' ? [provider('nous', false)]
  : scenario === 'alternative' ? [provider('minimax-oauth', false)]
  : scenario === 'connected-only' ? [provider('semantier-cloud', true)]
  : [provider('semantier-cloud', scenario === 'connected'), provider('nous', false)]

window.hermesDesktop = {
  api: async (request: { path: string }) => {
    if (request.path.startsWith('/api/providers/oauth')) { return { providers } }
    if (request.path.startsWith('/api/providers/custom-endpoints')) { return { endpoints: [] } }
    if (request.path.startsWith('/api/profiles')) { return { profiles: [{ name: 'default' }] } }
    if (request.path.startsWith('/api/env')) { return {} }
    if (request.path.startsWith('/api/config')) { return { display: { language: locale } } }
    throw new Error(`Unexpected Accounts fixture request: ${request.path}`)
  }
} as any

createRoot(document.getElementById('root')!).render(
  <MemoryRouter>
    <QueryClientProvider client={new QueryClient()}>
      <I18nProvider configClient={null} initialLocale={locale}>
        <ThemeProvider>
          <main className="mx-auto max-w-2xl p-6">
            <ProvidersSettings onClose={() => {}} onViewChange={() => {}} view="accounts" />
          </main>
        </ThemeProvider>
      </I18nProvider>
    </QueryClientProvider>
  </MemoryRouter>
)
