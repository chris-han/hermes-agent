import '../src/styles.css'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router'

import { PreviewPane } from '../src/app/chat/right-rail/preview-pane'
import { handleServerRequest } from '../src/app/session/hooks/use-message-stream/gateway-event/server-requests'
import { I18nProvider } from '../src/i18n'
import { createClientSessionState } from '../src/lib/chat-runtime'
import { openPreview } from '../src/store/preview'
import { setActiveSessionId, setSelectedStoredSessionId } from '../src/store/session'
import { publishSessionState } from '../src/store/session-states'
import { ThemeProvider } from '../src/themes/context'

const root = createRoot(document.getElementById('host')!)
const query = new QueryClient()
let guest: any
const runtimeId = 'preview-fixture-runtime'
const storedId = 'preview-fixture-session'
let requestId = 0
const sessionState = createClientSessionState(storedId)
const deps = {
  activeSessionIdRef: { current: runtimeId },
  sessionStateByRuntimeIdRef: { current: new Map([[runtimeId, sessionState]]) },
  sessionInterrupted: () => false
}
async function request(method: string, params: Record<string, unknown>) {
  return new Promise<any>((resolve, reject) => {
    const handled = handleServerRequest({
      id: `frame-fixture-${++requestId}`, method, profile: 'default', params: { ...params, session_id: runtimeId },
      respond: (value: { value: string }) => resolve(JSON.parse(value.value)), decline: () => reject(new Error('request declined')), fail: reject
    } as any, deps as any, runtimeId)
    if (!handled) {reject(new Error('request not handled'))}
  })
}
async function binding() {
  const result = await request('preview.context', {})
  if (!result.success) {throw new Error(result.code || 'PANE_UNAVAILABLE')}
  return result.binding
}

Object.assign(window, {
  fixture: {
    async open(url: string) {
      setActiveSessionId(runtimeId)
      setSelectedStoredSessionId(storedId)
      publishSessionState(runtimeId, sessionState)
      const target = { kind: 'url' as const, label: 'Fixture', source: url, url }
      const tabId = openPreview(target)!
      root.render(
        <MemoryRouter><QueryClientProvider client={query}><I18nProvider configClient={null} initialLocale="en"><ThemeProvider>
          <div style={{ width: 1000, height: 750 }}><PreviewPane tabId={tabId} target={target} /></div>
        </ThemeProvider></I18nProvider></QueryClientProvider></MemoryRouter>
      )
      for (let i = 0; i < 200; i++) {
        guest = document.querySelector('webview') as any
        try { if (guest && guest.getURL() && !guest.isLoading()) {return guest.getWebContentsId()} } catch { /* guest is attaching */ }
        await new Promise(resolve => setTimeout(resolve, 50))
      }
      throw new Error('fixture preview did not load')
    },
    read: async () => {
      const result = await request('preview.read', { require_live: true, expected_binding: await binding() })
      if (!result.success) {throw new Error(result.code || result.error)}
      return result
    },
    act: async (action: any) => {
      const { kind, ...params } = action
      return request('preview.act', { ...params, action: kind, expected_binding: await binding() })
    },
    guestId: () => guest.getWebContentsId(),
    guestScript: (code: string) => guest.executeJavaScript(code),
    guestLoad: (url: string) => guest.loadURL(url),
    zoom: (factor: number) => guest.setZoomFactor(factor),
    otherPartition: async (url: string) => {
      const other = document.createElement('webview') as any
      other.style.cssText = 'width:100px;height:100px'
      other.setAttribute('partition', 'persist:unrelated-fixture')
      other.setAttribute('src', url)
      const ready = new Promise(resolve => other.addEventListener('dom-ready', resolve, { once: true }))
      document.body.appendChild(other)
      await ready
      return other.getWebContentsId()
    },
    dispose: () => window.hermesDesktop.previewFrames!({ webContentsId: guest.getWebContentsId(), operation: 'dispose' })
  }
})
