import '@/styles.css'

/** Offline acceptance bootstrap; the screen, API, socket and backend are production owners. */
import { QueryClientProvider } from '@tanstack/react-query'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router'

import { HermesGateway } from '@/api/client'
import { BillingSettings } from '@/app/settings/billing'
import type { HermesConnection } from '@/global'
import { I18nProvider } from '@/i18n'
import { queryClient } from '@/lib/query-client'
import { setPrimaryGateway } from '@/store/gateway'
import { setConnection, setGatewayState } from '@/store/session'
import { ThemeProvider } from '@/themes/context'

const params = new URLSearchParams(location.search)
const bridge = params.get('bridge') || 'http://127.0.0.1:55465'
const bridgeOrigin = new URL(bridge)

if (bridgeOrigin.hostname !== '127.0.0.1' || bridgeOrigin.protocol !== 'http:') {
  throw new Error('Offline acceptance requires a loopback native bridge.')
}

let active: HermesGateway | undefined

async function connect(source: number) {
  document.body.dataset.bootstrapState = 'connecting'
  delete document.body.dataset.bootstrapError
  setPrimaryGateway(null, 'default')
  setConnection(null)
  active?.close()
  const response = await fetch(`${bridge}/connection?source=${source}`, { cache: 'no-store' })

  if (!response.ok) {throw new Error('Native bootstrap failed')}
  const connection = await response.json() as HermesConnection
  const gateway = new HermesGateway()
  gateway.onState(state => { if (active === gateway) {setGatewayState(state)} })
  active = gateway
  await gateway.connect(connection.wsUrl)
  setConnection(connection)
  setPrimaryGateway(gateway, 'default')
  document.body.dataset.bootstrapState = 'ready'
}

Object.assign(window, { __semantierBillingWitness: { connect, disconnect: () => active?.close() } })
createRoot(document.getElementById('root')!).render(
  <MemoryRouter initialEntries={['/settings?tab=billing']}>
    <QueryClientProvider client={queryClient}>
      <I18nProvider configClient={null} initialLocale={params.get('locale') || 'en'}>
        <ThemeProvider><main className="flex h-screen min-h-0 flex-col"><BillingSettings /></main></ThemeProvider>
      </I18nProvider>
    </QueryClientProvider>
  </MemoryRouter>
)
void connect(Number(params.get('source') || 0)).catch(error => {
  document.body.dataset.bootstrapState = 'failed'
  document.body.dataset.bootstrapError = String(error)
})
