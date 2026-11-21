import { useEffect, useId, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { DesktopRegistryConnection } from '@/global'
import { useI18n } from '@/i18n'
import { semantierWorkspaceCopy } from '@/i18n/semantier-workspace'
import { refreshConnectionsRegistry } from '@/store/connections'

import type { WorkspaceIdentity, WorkspaceScope } from '../../../electron/semantier-workspace-client'

import { ListRow } from './primitives'
import { SemantierManagedDevices } from './semantier-managed-devices'
import { SemantierPersonalSync } from './semantier-personal-sync'
import { SemantierWinmanCapability } from './semantier-winman-capability'

export function SemantierWorkspaceSettings({
  accountOnly = false,
  accountOrigin,
  accountScope,
  connections,
  onConnect
}: {
  accountOnly?: boolean
  accountOrigin?: string
  accountScope?: WorkspaceScope
  connections: DesktopRegistryConnection[]
  onConnect: (id: string) => Promise<void> | void
}) {
  const { locale } = useI18n()
  const copy = semantierWorkspaceCopy(locale)

  const [origin, setOrigin] = useState(
    () => accountOrigin || connections.find(c => c.transport === 'semantier-workspace-http')?.url || 'https://cloud.semantier.cn'
  )

  const [identity, setIdentity] = useState<WorkspaceIdentity | null>(null)
  const [checking, setChecking] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [accountRevision, setAccountRevision] = useState(0)
  const [modelState, setModelState] = useState<'preparing' | 'ready' | 'denied' | 'unavailable'>('unavailable')
  const [modelIssue, setModelIssue] = useState<string | undefined>()
  const generation = useRef(0)
  const accountAction = useRef(false)
  const addressId = useId()

  useEffect(() => {
    if (accountOnly && accountOrigin) {setOrigin(accountOrigin)}
  }, [accountOnly, accountOrigin])

  useEffect(() => window.hermesDesktop.workspace?.onAccountChanged?.(() => {
    void refreshConnectionsRegistry()
    if (!accountAction.current) {setAccountRevision(value => value + 1)}
  }), [])

  useEffect(() => {
    if (!accountOnly) {return}
    const api = window.hermesDesktop.workspace
    let live = true
    let updates = 0
    const update = (value: { state: typeof modelState; issue?: string }) => {setModelState(value.state); setModelIssue(value.issue)}
    const unsubscribe = api?.onModelStatus?.(value => {updates++; if (live) {update(value)}})
    void api?.modelStatus?.().then(value => {if (live && updates === 0) {update(value)}}).catch(() => {})
    return () => {live = false; unsubscribe?.()}
  }, [accountOnly])

  useEffect(() => {
    const current = ++generation.current
    setChecking(true)
    setIdentity(null)
    setError('')
    void window.hermesDesktop.workspace
      ?.identity(origin)
      .then(value => {
        if (generation.current === current) {
          setIdentity(value)
        }
      })
      .catch(reason => {
        if (generation.current === current) {
          setError(String(reason).includes('Semantier workspace request failed (403).') ? copy.forbidden : copy.unavailable)
        }
      })
      .finally(() => {
        if (generation.current === current) {
          setChecking(false)
        }
      })

    return () => {
      generation.current++
    }
  }, [origin, copy.forbidden, copy.unavailable, accountRevision])

  const connect = async () => {
    const current = generation.current
    setBusy(true)
    accountAction.current = true
    setError('')
    let signingIn = false

    try {
      const api = window.hermesDesktop.workspace

      if (!api) {
        throw new Error(copy.failed)
      }

      // Re-read before deciding whether to open hosted login; UI state is not authority.
      let actual = await api.identity(origin)

      if (generation.current !== current) { return }

      if (actual.authenticated === false) {
        signingIn = true
        actual = await api.login(origin)
      }

      if (generation.current !== current) {
        return
      }

      if (!actual.authenticated || !actual.user?.user_id || !actual.workspace?.id || !actual.organization_id) {
        throw new Error(copy.identityRequired)
      }

      if (accountOnly) {
        await api.selectAccount(origin, { userId: actual.user.user_id, workspaceId: actual.workspace.id, organizationId: actual.organization_id })
        if (generation.current !== current) {return}
        setIdentity(actual)
        await refreshConnectionsRegistry()
        return
      }
      setIdentity(actual)
      const normalized = new URL(origin).origin

      const saved = connections.find(
        c => c.transport === 'semantier-workspace-http' && new URL(c.url!).origin === normalized
      )

      const connection =
        saved ||
        (
          await window.hermesDesktop.connections.save({
            kind: 'cloud',
            label: connections.some(c => c.label === copy.title)
              ? `${copy.title} (${new URL(normalized).host})`
              : copy.title,
            url: normalized,
            transport: 'semantier-workspace-http'
          })
        ).connection

      if (generation.current !== current) {
        return
      }

      await refreshConnectionsRegistry()
      await onConnect(connection.id)
    } catch (reason) {
      if (generation.current === current) {
        setError(signingIn ? copy.failed : String(reason).includes('Semantier workspace request failed (403).')
          ? copy.forbidden : copy.unavailable)
      }
    } finally {
      accountAction.current = false
      setBusy(false)
    }
  }

  const signOut = async () => {
    setBusy(true)
    accountAction.current = true
    setError('')

    try {
      await window.hermesDesktop.workspace!.logout(origin)
      setIdentity(null)
    } catch {
      setError(copy.failed)
    } finally {
      accountAction.current = false
      setBusy(false)
    }
  }

  return (
    <form
      className="mb-5 grid min-w-0 gap-3"
      onSubmit={event => {
        event.preventDefault()

        if (!checking && !busy) {
          void connect()
        }
      }}
    >
      <ListRow description={accountOnly ? copy.accountDescription : copy.description} title={accountOnly ? copy.accountTitle : copy.title} />
      <div className="grid gap-1.5">
        <label className="text-xs text-muted-foreground" htmlFor={addressId}>
          {copy.address}
        </label>
        <Input
          disabled={busy}
          id={addressId}
          onChange={event => setOrigin(event.target.value)}
          required
          type="url"
          value={origin}
        />
      </div>
      <p className="break-words text-xs text-muted-foreground" role="status">
        {checking ? copy.checking : error ? '' : identity?.authenticated ? identity.user?.name || copy.signedIn : accountOnly ? copy.accountSignedOut : copy.signedOut}
      </p>
      {error ? (
        <p className="text-xs text-destructive" role="alert">
          {error}
        </p>
      ) : null}
      <div className="flex flex-wrap gap-2">
        <Button disabled={checking || busy} type="submit">
          {busy ? copy.waiting : error ? copy.retry : accountOnly ? identity?.authenticated ? copy.accountRetry : copy.accountSignIn : identity?.authenticated ? copy.open : copy.signIn}
        </Button>
        {identity?.authenticated ? (
          <Button disabled={busy} onClick={() => void signOut()} type="button" variant="outline">
            {copy.signOut}
          </Button>
        ) : null}
      </div>
      {accountOnly && accountOrigin === origin && identity?.authenticated && accountScope &&
        identity.user?.user_id === accountScope.userId && identity.workspace?.id === accountScope.workspaceId &&
        identity.organization_id === accountScope.organizationId ? (
          <ListRow title={copy.modelService} description={modelState === 'preparing' ? copy.modelPreparing :
            modelState === 'ready' ? copy.modelReady : modelState === 'denied' ?
              modelIssue?.startsWith('billing_account_') ? copy.modelBillingDenied : copy.modelDenied : copy.modelUnavailable}
            action={modelState === 'unavailable' || modelState === 'denied' ? (
              <Button disabled={busy} onClick={() => void window.hermesDesktop.workspace?.prepareModel?.().catch(() => setModelState('unavailable'))} type="button" variant="outline">
                {copy.modelRetry}
              </Button>
            ) : undefined} />
        ) : null}
      {accountOnly && accountOrigin === origin && identity?.authenticated && accountScope &&
        identity.user?.user_id === accountScope.userId && identity.workspace?.id === accountScope.workspaceId &&
        identity.organization_id === accountScope.organizationId ? (
          <SemantierWinmanCapability key={JSON.stringify([origin, accountScope])} origin={origin} scope={accountScope} />
        ) : null}
      {identity?.authenticated && identity.user?.user_id && identity.workspace?.id && identity.organization_id ? (
        <>
        <SemantierPersonalSync key={JSON.stringify([origin,identity.user.user_id,identity.workspace.id,identity.organization_id])} origin={origin} scope={{userId:identity.user.user_id,workspaceId:identity.workspace.id,organizationId:identity.organization_id}} />
        <SemantierManagedDevices
          origin={origin}
          scope={{
            userId: identity.user.user_id,
            workspaceId: identity.workspace.id,
            organizationId: identity.organization_id
          }}
        />
        </>
      ) : null}
    </form>
  )
}
