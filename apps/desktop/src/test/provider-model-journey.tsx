import '@/styles.css'

/** Acceptance entry only: real UI, native ticket bridge, canonical sessions and provider HTTP. */
import { useStore } from '@nanostores/react'
import { QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router'

import { HermesGateway, setApiRequestConnection, setApiRequestProfile } from '@/api/client'
import { getGlobalModelInfo } from '@/api/models'
import { useModelControls } from '@/app/session/hooks/use-model-controls'
import { ModelSettings } from '@/app/settings/model-settings'
import { ProvidersSettings } from '@/app/settings/providers-settings'
import { ModelMenuCloseContext, ModelMenuPanel } from '@/app/shell/model-menu-panel'
import { ConfirmHost } from '@/components/confirm-host'
import { NotificationStack } from '@/components/notifications'
import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import type { HermesApiRequest, HermesConnection } from '@/global'
import { I18nProvider } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { setPrimaryGateway, setPrimaryGatewayConnection } from '@/store/gateway'
import { $activeSessionId, $currentModel, $currentProvider, setConnection, setCurrentModel, setCurrentProvider } from '@/store/session'
import { ThemeProvider } from '@/themes/context'

const params = new URLSearchParams(location.search)
const bridge = 'http://127.0.0.1:55475'
let source = Number(params.get('source') || 0)
let gateway: HermesGateway
let connection: HermesConnection
Object.assign(window, { hermesDesktop: {
  api: async (request: HermesApiRequest) => {
    const response = await fetch(`${bridge}/api`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: request.connectionId?.startsWith('journey-') ? Number(request.connectionId.slice(8)) : source, request }) })

    const data = await response.json()

    if (!response.ok) { throw new Error(data.detail || `HTTP ${response.status}`) }

    return data
  }
} })
const requestGateway = <T,>(method: string, values?: Record<string, unknown>) => gateway.request<T>(method, values)

async function connect(next: number) {
  gateway?.close()
  source = next
  connection = await (await fetch(`${bridge}/connection?source=${source}`)).json()
  gateway = new HermesGateway()
  await gateway.connect(connection.wsUrl)
  setConnection(connection)
  setPrimaryGateway(gateway, 'default')
  setPrimaryGatewayConnection(connection)
  setApiRequestConnection(connection.connectionId ?? null)
  setApiRequestProfile('default')
  queryClient.clear()
  $activeSessionId.set(null)
  setCurrentModel('')
  setCurrentProvider('')
}

await connect(source)

function Journey() {
  const [page, setPage] = useState('providers')
  const [providerView, setProviderView] = useState<'keys' | 'accounts' | 'custom-endpoints'>('keys')
  const [menuOpen, setMenuOpen] = useState(false)
  const [generation, setGeneration] = useState(0)
  const [error, setError] = useState('')
  const model = useStore($currentModel)
  const provider = useStore($currentProvider)

  const { selectModel, applySavedMainModel } = useModelControls({ queryClient, requestGateway,
    cacheProfile: 'default', cacheOwnerConnectionId: connection.connectionId })

  const newChat = async () => {
    try {
      const selected = await getGlobalModelInfo('default')

      const result = await requestGateway<{ session_id: string }>('session.create', {
        source: 'desktop', profile: 'default', model: selected.model, provider: selected.provider,
        toolsets: [], title: 'Provider journey synthetic acceptance'
      })

      $activeSessionId.set(result.session_id)
      setCurrentModel(selected.model)
      setCurrentProvider(selected.provider)
      setPage('composer')
    } catch (cause) { setError(String(cause)) }
  }

  Object.assign(window, { __providerJourney: { connect: async (next: number) => {
    await connect(next); setGeneration(value => value + 1)
  }, request: requestGateway } })

  return <main className="mx-auto flex min-h-screen max-w-5xl flex-col gap-4 p-4" key={generation}>
    <nav className="flex flex-wrap gap-2">
      <Button onClick={() => { setProviderView('keys'); setPage('providers') }}>API Keys</Button>
      <Button onClick={() => setPage('main')}>Main Model</Button>
      <Button onClick={() => void newChat()}>Open Chat</Button>
    </nav>
    {error && <p role="alert">{error}</p>}
    {page === 'providers' && <ProvidersSettings onClose={() => {}} onConfigSaved={() => void queryClient.invalidateQueries()}
      onMainModelChanged={applySavedMainModel} onViewChange={setProviderView} view={providerView} />}
    {page === 'main' && <ModelSettings onMainModelChanged={applySavedMainModel} scopeProfile="default" subpage="main" />}
    {page === 'composer' && <section>
      <p data-testid="route">{provider} / {model}</p>
      <DropdownMenu onOpenChange={setMenuOpen} open={menuOpen}>
        <DropdownMenuTrigger asChild><Button>Choose chat model</Button></DropdownMenuTrigger>
        <DropdownMenuContent className="max-w-[calc(100vw-2rem)]">
          <ModelMenuCloseContext.Provider value={() => setMenuOpen(false)}>
          <ModelMenuPanel gateway={gateway} onSelectModel={selectModel} ownerConnectionId={connection.connectionId}
            profile="default" requestGateway={requestGateway} />
          </ModelMenuCloseContext.Provider>
        </DropdownMenuContent>
      </DropdownMenu>
    </section>}
    <ConfirmHost /><NotificationStack />
  </main>
}

createRoot(document.getElementById('root')!).render(<MemoryRouter>
  <QueryClientProvider client={queryClient}>
    <I18nProvider configClient={null} initialLocale={params.get('locale') || 'en'}>
      <ThemeProvider><Journey /></ThemeProvider>
    </I18nProvider>
  </QueryClientProvider>
</MemoryRouter>)
