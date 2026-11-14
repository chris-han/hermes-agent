import { getHermesConfigRecord } from '@/api/config'
import { withTimeout } from '@/lib/with-timeout'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { captureOnboardingScope, type OnboardingScope } from '@/store/onboarding-scope'
import { openPreview, previewProfileScope } from '@/store/preview'
import { $activeSessionId } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'

/** Account navigation is available before device authorization or model setup. */
export async function openCloudSignIn(scope: OnboardingScope): Promise<void> {
  const owner = { profile: previewProfileScope(), session: $focusedStoredSessionId.get(), runtime: $activeSessionId.get() }
  const source = $connectionsRegistry.get()?.connections.find(connection => connection.id === scope.connectionId)
  const workspace = source?.transport === 'semantier-workspace-http'
  let portal = workspace ? source.url : undefined

  if (!workspace) {
    // An unavailable native backend must not prevent ordinary account browsing.
    const config = await withTimeout(getHermesConfigRecord(scope), 2000, 'Cloud portal settings unavailable')
      .catch(() => null)

    const providers = config?.providers as Record<string, { portal_origin?: string; base_url?: string }> | undefined
    const billing = config?.billing as { portal_origin?: string } | undefined
    const provider = providers?.['semantier-cloud']
    portal = provider?.portal_origin || billing?.portal_origin || provider?.base_url
  }

  const origin = new URL(portal || 'https://cloud.semantier.cn')

  if (origin.username || origin.password || (origin.protocol !== 'https:' &&
    !(origin.protocol === 'http:' && ['localhost', '127.0.0.1', '[::1]'].includes(origin.hostname)))) {
    throw new Error('Use a secure Semantier Cloud origin.')
  }

  if (JSON.stringify(captureOnboardingScope()) !== JSON.stringify(scope) || owner.profile !== previewProfileScope() ||
    owner.session !== $focusedStoredSessionId.get() || owner.runtime !== $activeSessionId.get()) { return }

  const url = new URL('/billing?entry=windows', origin.origin).href
  openPreview({ kind: 'url', label: 'Semantier Cloud', source: url, url }, owner.session, owner.runtime, owner.profile, undefined, 'matching-url')
}

/** Open a server-supplied Semantier Billing task using the existing pane owner. */
export function openCloudPortal(value: string, label = 'Semantier Billing'): boolean {
  try {
    const target = new URL(value)

    if (target.username || target.password ||
      (target.protocol !== 'https:' && !(target.protocol === 'http:' &&
        ['localhost', '127.0.0.1', '[::1]'].includes(target.hostname))) ||
      !['/billing', '/pricing'].some(path => target.pathname === path || target.pathname.startsWith(path + '/'))) { return false }

    target.searchParams.set('entry', 'windows')
    const url = target.href
    openPreview({ kind: 'url', label, source: url, url }, $focusedStoredSessionId.get(),
      $activeSessionId.get(), previewProfileScope(), undefined, 'matching-url')

    return true
  } catch {
    return false
  }
}
