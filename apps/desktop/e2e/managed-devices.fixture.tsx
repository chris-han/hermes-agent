import '../src/styles.css'

import { createRoot } from 'react-dom/client'

import { SemantierManagedDevices } from '../src/app/settings/semantier-managed-devices'
import { I18nProvider } from '../src/i18n'
import { ThemeProvider } from '../src/themes/context'

const params = new URLSearchParams(location.search)
const locale = params.get('locale') || 'en'
const events: string[] = []
const scope = { userId: 'synthetic-user', workspaceId: 'synthetic-workspace', organizationId: 'synthetic-org' }

let devices = [
  {
    device_id: 'synthetic-pc-a-' + 'a'.repeat(80),
    status: 'active',
    assurance: 'UNKNOWN',
    created_at: '2026-11-20T00:00:00Z',
    last_seen_at: '2026-11-20T00:00:00Z',
    revoked_at: null
  }
]

Object.assign(window, { managedDeviceFixtureEvents: events })
window.hermesDesktop = {
  workspace: {
    managedDevices: async () => {
      if (params.get('scenario') === 'unavailable') {
        throw new Error('Synthetic unavailable service')
      }

      return { included_windows_seats: 2, devices, additional_seat_checkout: 'unavailable' }
    },
    managedDeviceRevoke: async (_origin: string, _scope: unknown, id: string) => {
      events.push(id)
      devices = devices.map(row => (row.device_id === id ? { ...row, status: 'revoked' } : row))

      return { device_id: id, status: 'revoked' }
    }
  }
} as unknown as typeof window.hermesDesktop

createRoot(document.getElementById('root')!).render(
  <I18nProvider configClient={null} initialLocale={locale}>
    <ThemeProvider>
      <main className="mx-auto grid max-w-2xl gap-4 p-6">
        <p className="text-xs text-muted-foreground">
          Synthetic device-management fixture — no actual enrollment or hardware verification
        </p>
        <SemantierManagedDevices origin="https://fixture.invalid" scope={scope} />
      </main>
    </ThemeProvider>
  </I18nProvider>
)
