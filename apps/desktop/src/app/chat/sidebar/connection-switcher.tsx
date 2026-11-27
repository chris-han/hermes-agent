import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  dropdownMenuRow,
  DropdownMenuSearch,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import type { DesktopRegistryConnection } from '@/global'
import { useI18n } from '@/i18n'
import { semantierWorkspaceCopy } from '@/i18n/semantier-workspace'
import {
  CONNECTION_SEARCH_THRESHOLD,
  connectionMatchesQuery,
  connectionTooltip,
  sortConnectionsForDisplay
} from '@/lib/connection-display'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2 } from '@/lib/icons'
import {
  authenticateWorkspace,
  DEFAULT_SEMANTIER_WORKSPACE_ORIGIN,
  ensureWorkspaceConnection
} from '@/lib/semantier-workspace-connection'
import { cn } from '@/lib/utils'
import { $activeConnectionId, $connectionsRegistry, $pendingConnectionId, selectConnection } from '@/store/connections'
import { closeFindBar } from '@/store/find-in-page'
import { notifyError } from '@/store/notifications'

import { ConnectionGlyph } from './connection-glyph'
import { useLocalDeviceSwitch } from './local-device-switch'

export function ConnectionSwitcher({
  compact = false,
  onConnect,
  onDiagnostics,
  readiness
}: {
  compact?: boolean
  onConnect: () => void
  onDiagnostics?: () => void
  readiness?: string
}) {
  const { locale, t } = useI18n()
  const cloudCopy = semantierWorkspaceCopy(locale)
  const title = readiness ? cloudCopy.executionEnvironment : t.settings.connections.title
  const registry = useStore($connectionsRegistry)
  const activeConnectionId = useStore($activeConnectionId)
  const pendingConnectionId = useStore($pendingConnectionId)
  const [searchQuery, setSearchQuery] = useState('')
  const [signingIn, setSigningIn] = useState(false)
  const signingInRef = useRef(false)
  const intent = useRef(0)
  useEffect(
    () => () => {
      intent.current++
    },
    []
  )
  const [menuOpen, setMenuOpen] = useState(false)
  const connectionListRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)
  const { dialog: localDeviceDialog, request: requestLocalDevice } = useLocalDeviceSwitch()

  const connections = useMemo(() => sortConnectionsForDisplay(registry?.connections ?? []), [registry?.connections])

  const activeConnection = connections.find(connection => connection.id === activeConnectionId)
  const searchable = connections.length >= CONNECTION_SEARCH_THRESHOLD

  const kindLabels: Record<DesktopRegistryConnection['kind'], string> = {
    cloud: t.settings.connections.kindCloud,
    local: t.settings.connections.kindLocal,
    remote: t.settings.connections.kindRemote,
    ssh: t.settings.connections.kindSsh
  }

  const displayedConnections = searchable
    ? connections.filter(connection => connectionMatchesQuery(connection, searchQuery, [kindLabels[connection.kind]]))
    : connections

  useEffect(() => {
    if (!menuOpen || !searchable || searchQuery) {
      return
    }

    connectionListRef.current?.querySelector('[aria-checked="true"]')?.scrollIntoView({ block: 'nearest' })
  }, [activeConnectionId, menuOpen, searchQuery, searchable])

  useEffect(() => {
    if (!menuOpen) {
      return
    }

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') {
        return
      }

      event.preventDefault()
      event.stopPropagation()
      setMenuOpen(false)
      setSearchQuery('')
    }

    window.addEventListener('keydown', closeOnEscape, { capture: true })

    return () => window.removeEventListener('keydown', closeOnEscape, { capture: true })
  }, [menuOpen])

  if (!readiness && connections.length <= 1) {
    return null
  }

  const chooseCloud = (connection?: DesktopRegistryConnection) => {
    if (signingInRef.current) {
      return
    }

    const revision = ++intent.current
    const initialConnection = $activeConnectionId.get()
    const isCurrent = () => intent.current === revision && $activeConnectionId.get() === initialConnection
    const origin = connection?.url ?? DEFAULT_SEMANTIER_WORKSPACE_ORIGIN
    signingInRef.current = true
    setSigningIn(true)

    void (async () => {
      try {
        if (!(await authenticateWorkspace(origin, isCurrent)) || !isCurrent()) {
          return
        }

        const target =
          connection ??
          (await ensureWorkspaceConnection(origin, $connectionsRegistry.get()?.connections ?? [], cloudCopy.title))

        if (isCurrent()) {
          await selectConnection(target.id)
        }
      } catch (error) {
        if (isCurrent()) {
          notifyError(error, t.profiles.switchConnectionFailed(cloudCopy.title))
        }
      } finally {
        signingInRef.current = false
        setSigningIn(false)
      }
    })()
  }

  const choose = (connectionId: string) => {
    triggerHaptic('selection')
    const connection = connections.find(candidate => candidate.id === connectionId)

    if (readiness && connection?.transport === 'semantier-workspace-http') {
      chooseCloud(connection)

      return
    }

    const revision = ++intent.current

    void (async () => {
      if (connection?.kind === 'local' && connectionId !== activeConnectionId) {
        const accepted = await requestLocalDevice({
          connectionId,
          label: connection.label,
          replaceCenter: true
        })

        if (!accepted || intent.current !== revision) {
          return
        }
      }

      await selectConnection(connectionId).catch(error =>
        notifyError(error, t.profiles.switchConnectionFailed(connection?.label ?? connectionId))
      )
    })()
  }

  return (
    <div
      aria-busy={signingIn || pendingConnectionId !== null}
      aria-label={title}
      className={cn(
        'min-w-20 shrink overflow-hidden',
        compact ? (readiness ? 'h-full max-w-64' : 'h-full max-w-40') : 'max-w-72'
      )}
      data-slot="connection-switcher"
      role="group"
    >
      <DropdownMenu
        onOpenChange={open => {
          setMenuOpen(open)

          if (!open) {
            setSearchQuery('')
          }
        }}
        open={menuOpen}
      >
        <DropdownMenuTrigger asChild>
          <ConnectionSwitcherTrigger
            activeConnection={activeConnection}
            compact={compact}
            localLabel={readiness ? cloudCopy.thisDevice : undefined}
            pending={signingIn || pendingConnectionId !== null}
            readiness={readiness}
            title={title}
          />
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="start"
          className={cn('min-w-52 max-w-72', searchable && 'w-72 overflow-hidden p-0')}
          collisionPadding={8}
          onKeyDownCapture={event => {
            if (searchable && (event.metaKey || event.ctrlKey) && event.key.toLocaleLowerCase() === 'f') {
              event.preventDefault()
              event.stopPropagation()
              // The app-level keybind sees the chord at window capture before
              // this portal and may open Find in page. This menu owns the chord
              // while it is open, so close that surface before focusing here.
              closeFindBar()
              searchInputRef.current?.focus()
              searchInputRef.current?.select()
            }
          }}
          side="top"
        >
          {searchable && (
            <>
              <DropdownMenuSearch
                className="[font-family:inherit] text-xs font-normal leading-4"
                onKeyDown={event => {
                  if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') {
                    return
                  }

                  const results = connectionListRef.current?.querySelectorAll<HTMLElement>(
                    '[role="menuitemradio"]:not([data-disabled])'
                  )

                  const target =
                    event.key === 'ArrowDown' ? results?.item(0) : results?.item((results?.length ?? 1) - 1)

                  if (target) {
                    event.preventDefault()
                    event.stopPropagation()
                    target.focus()
                  }
                }}
                onValueChange={setSearchQuery}
                placeholder={t.settings.connections.searchPlaceholder}
                ref={searchInputRef}
                value={searchQuery}
              />
              <DropdownMenuSeparator className="m-0" />
            </>
          )}
          <DropdownMenuRadioGroup
            className={
              searchable
                ? 'dt-portal-scrollbar h-48 max-h-[calc(var(--radix-dropdown-menu-content-available-height)-4.5rem)] overflow-y-auto p-1'
                : undefined
            }
            onValueChange={choose}
            ref={connectionListRef}
            value={activeConnectionId ?? ''}
          >
            {displayedConnections.length === 0 ? (
              <div
                className="flex h-full items-center justify-center px-4 text-center text-xs text-(--ui-text-tertiary)"
                role="status"
              >
                {t.settings.connections.noSearchResults}
              </div>
            ) : (
              displayedConnections.map(connection => (
                <DropdownMenuRadioItem
                  className={cn('min-w-0', searchable && dropdownMenuRow)}
                  disabled={Boolean(readiness && signingIn && connection.transport === 'semantier-workspace-http')}
                  key={connection.id}
                  onSelect={readiness && connection.id === activeConnectionId ? () => choose(connection.id) : undefined}
                  value={connection.id}
                >
                  <ConnectionLabel connection={connection} localLabel={readiness ? cloudCopy.thisDevice : undefined} />
                </DropdownMenuRadioItem>
              ))
            )}
          </DropdownMenuRadioGroup>
          {readiness && !connections.some(connection => connection.transport === 'semantier-workspace-http') && (
            <DropdownMenuItem disabled={signingIn} onSelect={() => chooseCloud()}>
              {cloudCopy.title}
            </DropdownMenuItem>
          )}
          <DropdownMenuSeparator className={searchable ? 'm-0' : undefined} />
          <DropdownMenuItem className={searchable ? dropdownMenuRow : undefined} onSelect={onConnect}>
            <ManageGatewaysLabel label={t.profiles.connectGateway} />
          </DropdownMenuItem>
          {onDiagnostics && <DropdownMenuItem onSelect={onDiagnostics}>{cloudCopy.diagnostics}</DropdownMenuItem>}
        </DropdownMenuContent>
      </DropdownMenu>
      {localDeviceDialog}
    </div>
  )
}

interface ConnectionMenuProps {
  activeConnection?: DesktopRegistryConnection
  compact: boolean
  localLabel?: string
  pending: boolean
  readiness?: string
  title: string
}

function ConnectionSwitcherTrigger({
  activeConnection,
  compact,
  localLabel,
  pending,
  readiness,
  title,
  ...triggerProps
}: ConnectionMenuProps & React.ComponentProps<'button'>) {
  return (
    <Button
      {...triggerProps}
      aria-label={
        activeConnection
          ? `${title}: ${activeConnection.kind === 'local' ? (localLabel ?? activeConnection.label) : activeConnection.label}${readiness ? ` · ${readiness}` : ''}`
          : title
      }
      className={cn(
        'w-full min-w-0 justify-between overflow-hidden px-1 text-(--ui-text-secondary) data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground',
        compact && 'h-full min-h-0 rounded-none px-1.5 text-[0.6875rem] font-normal',
        triggerProps.className
      )}
      size="xs"
      type="button"
      variant="ghost"
    >
      <span className="flex min-w-0 flex-1 items-center gap-1 overflow-hidden">
        {pending && <Loader2 aria-hidden="true" className="size-3 shrink-0 animate-spin" />}
        {activeConnection ? (
          <ConnectionLabel connection={activeConnection} localLabel={localLabel} />
        ) : (
          <span className="truncate">{title}</span>
        )}
      </span>
      {readiness && <span className="shrink-0 px-1 text-muted-foreground">{readiness}</span>}
      <Codicon aria-hidden="true" className="shrink-0 opacity-60" name="chevron-down" size="0.875rem" />
    </Button>
  )
}

function ManageGatewaysLabel({ label }: { label: string }) {
  return (
    <span className="flex min-w-0 items-center gap-1.5 text-(--ui-text-secondary)">
      <Codicon aria-hidden="true" name="settings-gear" size="0.875rem" />
      <span className="truncate">{label}</span>
    </span>
  )
}

function ConnectionLabel({ connection, localLabel }: { connection: DesktopRegistryConnection; localLabel?: string }) {
  return (
    <span className="flex min-w-0 flex-1 items-center gap-1 overflow-hidden" title={connectionTooltip(connection)}>
      <ConnectionGlyph connection={connection} />
      <span className="truncate">
        {connection.kind === 'local' ? (localLabel ?? connection.label) : connection.label}
      </span>
    </span>
  )
}
