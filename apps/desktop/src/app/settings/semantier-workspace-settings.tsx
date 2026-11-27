import { useEffect, useId, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { DesktopRegistryConnection } from '@/global'
import { useI18n } from '@/i18n'
import { semantierWorkspaceCopy } from '@/i18n/semantier-workspace'
import {
  authenticateWorkspace,
  DEFAULT_SEMANTIER_WORKSPACE_ORIGIN,
  ensureWorkspaceConnection
} from '@/lib/semantier-workspace-connection'

import type { WorkspaceIdentity } from '../../../electron/semantier-workspace-client'

import { ListRow } from './primitives'
import { SemantierDesktopProjects } from './semantier-desktop-projects'
import { SemantierManagedDevices } from './semantier-managed-devices'
import { SemantierPersonalSync } from './semantier-personal-sync'

export function SemantierWorkspaceSettings({
  connections,
  onConnect,
  projectsOnly = false
}: {
  connections: DesktopRegistryConnection[]
  onConnect?: (id: string) => Promise<void> | void
  projectsOnly?: boolean
}) {
  const { locale } = useI18n()
  const copy = semantierWorkspaceCopy(locale)

  const [origin, setOrigin] = useState(
    () => connections.find(c => c.transport === 'semantier-workspace-http')?.url || DEFAULT_SEMANTIER_WORKSPACE_ORIGIN
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
      const isCurrent = () => generation.current === current
      const actual = await authenticateWorkspace(origin, isCurrent)

      if (!actual) {
        return
      }

      setIdentity(actual)

      if (projectsOnly) {
        return
      }
      const connection = await ensureWorkspaceConnection(origin, connections, copy.title)

      if (isCurrent()) {
        await onConnect?.(connection.id)
      }
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

        if (!checking && !busy && (!projectsOnly || !identity?.authenticated)) {
          void connect()
        }
      }}
    >
      <ListRow
        description={
          projectsOnly
            ? locale === 'zh'
              ? '通过工作区账户继续项目。'
              : 'Use your Workspace account to continue a project.'
            : copy.description
        }
        title={projectsOnly ? (locale === 'zh' ? '项目' : 'Projects') : copy.title}
      />
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
        {(!projectsOnly || !identity?.authenticated) && (
          <Button disabled={checking || busy} type="submit">
            {busy
              ? copy.waiting
              : projectsOnly
                ? locale === 'zh'
                  ? '登录工作区'
                  : 'Sign in to Workspace'
                : identity?.authenticated
                  ? copy.open
                  : copy.signIn}
          </Button>
        )}
        {identity?.authenticated ? (
          <Button disabled={busy} onClick={() => void signOut()} type="button" variant="outline">
            {copy.signOut}
          </Button>
        ) : null}
      </div>
      {identity?.authenticated && identity.user?.user_id && identity.workspace?.id && identity.organization_id ? (
        projectsOnly ? (
          <SemantierDesktopProjects
            key={JSON.stringify([origin, identity.user.user_id, identity.workspace.id, identity.organization_id])}
            origin={origin}
            scope={{
              userId: identity.user.user_id,
              workspaceId: identity.workspace.id,
              organizationId: identity.organization_id
            }}
          />
        ) : (
          <>
            <SemantierPersonalSync
              key={JSON.stringify([origin, identity.user.user_id, identity.workspace.id, identity.organization_id])}
              origin={origin}
              scope={{
                userId: identity.user.user_id,
                workspaceId: identity.workspace.id,
                organizationId: identity.organization_id
              }}
            />
            <SemantierManagedDevices
              origin={origin}
              scope={{
                userId: identity.user.user_id,
                workspaceId: identity.workspace.id,
                organizationId: identity.organization_id
              }}
            />
          </>
        )
      ) : null}
    </form>
  )
}
