import { useEffect, useId, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { DesktopRegistryConnection } from '@/global'
import { useI18n } from '@/i18n'
import { semantierWorkspaceCopy } from '@/i18n/semantier-workspace'
import { refreshConnectionsRegistry } from '@/store/connections'

import type { WorkspaceIdentity } from '../../../electron/semantier-workspace-client'

import { ListRow } from './primitives'

export function SemantierWorkspaceSettings({
  connections,
  onConnect
}: {
  connections: DesktopRegistryConnection[]
  onConnect: (id: string) => Promise<void> | void
}) {
  const { locale } = useI18n()
  const copy = semantierWorkspaceCopy(locale)

  const [origin, setOrigin] = useState(
    () => connections.find(c => c.transport === 'semantier-workspace-http')?.url || 'https://cloud.semantier.cn'
  )

  const [identity, setIdentity] = useState<WorkspaceIdentity | null>(null)
  const [checking, setChecking] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const generation = useRef(0)
  const addressId = useId()

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
      .catch(() => {
        if (generation.current === current) {
          setError(copy.failed)
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
  }, [origin, copy.failed])

  const connect = async () => {
    const current = generation.current
    setBusy(true)
    setError('')

    try {
      const api = window.hermesDesktop.workspace

      if (!api) {
        throw new Error(copy.failed)
      }

      // Re-read before deciding whether to open hosted login; UI state is not authority.
      let actual = await api.identity(origin)

      if (!actual.authenticated) {
        actual = await api.login(origin)
      }

      if (generation.current !== current) {
        return
      }

      if (!actual.authenticated || !actual.user?.user_id || !actual.workspace?.id || !actual.organization_id) {
        throw new Error(copy.identityRequired)
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
    } catch {
      if (generation.current === current) {
        setError(copy.failed)
      }
    } finally {
      setBusy(false)
    }
  }

  const signOut = async () => {
    setBusy(true)
    setError('')

    try {
      await window.hermesDesktop.workspace!.logout(origin)
      setIdentity(null)
    } catch {
      setError(copy.failed)
    } finally {
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
      <ListRow description={copy.description} title={copy.title} />
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
        {checking ? copy.checking : identity?.authenticated ? identity.user?.name || copy.signedIn : copy.signedOut}
      </p>
      {error ? (
        <p className="text-xs text-destructive" role="alert">
          {error}
        </p>
      ) : null}
      <div className="flex flex-wrap gap-2">
        <Button disabled={checking || busy} type="submit">
          {busy ? copy.waiting : identity?.authenticated ? copy.open : copy.signIn}
        </Button>
        {identity?.authenticated ? (
          <Button disabled={busy} onClick={() => void signOut()} type="button" variant="outline">
            {copy.signOut}
          </Button>
        ) : null}
      </div>
    </form>
  )
}
