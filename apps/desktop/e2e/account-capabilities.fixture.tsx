import '../src/styles.css'

import { createRoot } from 'react-dom/client'

import { FileSyncSettings } from '../src/app/settings/file-sync-settings'
import { SemantierWinmanCapability } from '../src/app/settings/semantier-winman-capability'
import { I18nProvider } from '../src/i18n'
import { $connectionsRegistry } from '../src/store/connection-registry-state'
import { ThemeProvider } from '../src/themes/context'

const locale = new URLSearchParams(location.search).get('locale') || 'en'
const origin = 'https://fixture.invalid'
const scope = { userId: 'fixture-user', workspaceId: 'fixture-workspace', organizationId: 'fixture-org' }
const events: string[] = []
let file = { state: 'Paused', folder: '' }
let enabled = false
let revision = 0
const preference = () => ({ desired: { enabled, revision, version: 1 }, execution: 'unavailable' })
Object.assign(window, { capabilityFixtureEvents: events })
window.hermesDesktop = {
  fileSync: {
    status: async () => file,
    choose: async () => {events.push('file:choose'); file = { state: 'Synced', folder: 'fixture-folder' }; return file},
    pause: async () => {events.push('file:pause'); file = { ...file, state: 'Paused' }; return file},
    resume: async () => {events.push('file:resume'); file = { ...file, state: 'Synced' }; return file}
  },
  workspace: {
    winmanCapability: async () => preference(),
    setWinmanCapability: async (_origin: string, _scope: unknown, value: boolean) => {
      events.push('winman:' + value); enabled = value; revision++; return preference()
    }
  }
} as unknown as typeof window.hermesDesktop
$connectionsRegistry.set({ connections: [], accountContext: { origin, scope } } as never)
createRoot(document.getElementById('root')!).render(
  <I18nProvider configClient={null} initialLocale={locale}>
    <ThemeProvider><main className="mx-auto grid max-w-2xl gap-6 p-6">
      <FileSyncSettings />
      <SemantierWinmanCapability origin={origin} scope={scope} />
    </main></ThemeProvider>
  </I18nProvider>
)
