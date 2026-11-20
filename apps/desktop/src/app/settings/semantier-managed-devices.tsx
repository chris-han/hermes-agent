import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { managedDeviceCopy } from '@/i18n/semantier-managed-devices'

import type { ManagedDeviceInventory, WorkspaceScope } from '../../../electron/semantier-workspace-client'

export function SemantierManagedDevices({ origin, scope }: { origin: string; scope: WorkspaceScope }) {
  const { locale } = useI18n()
  const copy = managedDeviceCopy(locale)
  const [snapshot, setSnapshot] = useState<{ scopeKey: string; value: ManagedDeviceInventory } | null>(null)
  const [busy, setBusy] = useState(false)
  const [failed, setFailed] = useState(false)
  const [pending, setPending] = useState<string | null>(null)
  const generation = useRef(0)
  const scopeKey = JSON.stringify([origin, scope.userId, scope.workspaceId, scope.organizationId])
  const inventory = snapshot?.scopeKey === scopeKey ? snapshot.value : null
  // eslint-disable-next-line no-restricted-syntax -- Request epoch invalidation, not an atom mirrored into a ref.
  useEffect(() => {
    const current = ++generation.current
    setSnapshot(null)
    setPending(null)
    setBusy(false)
    setFailed(false)

    return () => {
      generation.current = current + 1
    }
  }, [scopeKey])

  async function load() {
    const current = generation.current
    setBusy(true)
    setFailed(false)

    try {
      const api = window.hermesDesktop.workspace

      if (!api?.managedDevices) {
        throw new Error('Device management unavailable')
      }

      const value = await api.managedDevices(origin, scope)

      if (generation.current === current) {
        setSnapshot({ scopeKey, value })
      }
    } catch {
      if (generation.current === current) {
        setFailed(true)
      }
    } finally {
      if (generation.current === current) {
        setBusy(false)
      }
    }
  }

  async function revoke(deviceId: string) {
    const current = generation.current
    setBusy(true)
    setFailed(false)

    try {
      const api = window.hermesDesktop.workspace

      if (!api?.managedDeviceRevoke) {
        throw new Error('Device management unavailable')
      }

      const receipt = await api.managedDeviceRevoke(origin, scope, deviceId)

      if (generation.current !== current) {
        return
      }

      if (receipt.device_id !== deviceId || receipt.status !== 'revoked') {
        throw new Error('Device receipt mismatch')
      }

      setSnapshot(old =>
        old?.scopeKey === scopeKey
          ? {
              scopeKey,
              value: {
                ...old.value,
                devices: old.value.devices.map(row =>
                  row.device_id === deviceId ? { ...row, status: 'revoked' } : row
                )
              }
            }
          : old
      )
      setPending(null)
      await load()
    } catch {
      if (generation.current === current) {
        setFailed(true)
      }
    } finally {
      if (generation.current === current) {
        setBusy(false)
      }
    }
  }

  return (
    <section aria-label={copy.title} className="grid min-w-0 gap-3">
      <h4 className="text-sm font-medium">{copy.title}</h4>
      <p className="text-xs text-muted-foreground">{copy.included}</p>
      <div>
        <Button disabled={busy} onClick={() => void load()} type="button" variant="outline">
          {inventory ? copy.refresh : copy.view}
        </Button>
      </div>
      {busy ? (
        <p className="text-xs text-muted-foreground" role="status">
          {copy.loading}
        </p>
      ) : null}
      {failed ? (
        <p className="text-xs text-destructive" role="alert">
          {copy.failed}
        </p>
      ) : null}
      {inventory ? (
        <>
          {inventory.seat_summary ? (
            <div className="grid gap-1 text-xs text-muted-foreground">
              <p>
                {copy.summary
                  .replace('{active}', String(inventory.seat_summary.active_windows_devices))
                  .replace('{licensed}', String(inventory.seat_summary.licensed_windows_seats))}
              </p>
              <p>
                {copy.purchased
                  .replace('{included}', String(inventory.seat_summary.included_windows_seats))
                  .replace('{purchased}', String(inventory.seat_summary.purchased_windows_seats))}
              </p>
              {inventory.seat_summary.development_windows_seats > 0 ? (
                <p>{copy.development.replace('{count}', String(inventory.seat_summary.development_windows_seats))}</p>
              ) : null}
              <p>{copy.ceiling.replace('{count}', String(inventory.seat_summary.new_enrollment_limit))}</p>
              {inventory.seat_summary.retained_above_capacity > 0 ? <p>{copy.retained}</p> : null}
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">{copy.coverageUnknown}</p>
          )}
          {inventory.devices_has_more ? <p className="text-xs text-muted-foreground">{copy.more}</p> : null}
          {!inventory.devices.length ? <p className="text-xs text-muted-foreground">{copy.empty}</p> : null}
          <ul className="grid min-w-0 gap-4">
            {inventory.devices.map(row => (
              <li className="grid min-w-0 gap-2" key={row.device_id}>
                <span className="break-all text-xs font-medium">{row.device_id}</span>
                <p className="text-xs text-muted-foreground">{row.status === 'active' ? copy.active : copy.revoked}</p>
                <p className="text-xs text-muted-foreground">
                  {row.assurance === 'TPM_ATTESTED' ? copy.verified : copy.unknown}
                </p>
                <p className="break-words text-xs text-muted-foreground">
                  {copy.seen}: <time dateTime={row.last_seen_at}>{row.last_seen_at}</time>
                </p>
                {row.status === 'active' ? (
                  pending === row.device_id ? (
                    <div className="grid gap-2">
                      <p className="text-xs">{copy.question}</p>
                      <div className="flex flex-wrap gap-2">
                        <Button
                          disabled={busy}
                          onClick={() => void revoke(row.device_id)}
                          type="button"
                          variant="outline"
                        >
                          {copy.confirm}
                        </Button>
                        <Button disabled={busy} onClick={() => setPending(null)} type="button" variant="ghost">
                          {copy.cancel}
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div>
                      <Button disabled={busy} onClick={() => setPending(row.device_id)} type="button" variant="outline">
                        {copy.remove}
                      </Button>
                    </div>
                  )
                ) : null}
              </li>
            ))}
          </ul>
          {inventory.additional_seat_checkout === 'unavailable' ? (
            <p className="text-xs text-muted-foreground">{copy.checkout}</p>
          ) : null}
        </>
      ) : null}
      <p className="text-xs text-muted-foreground">{copy.enrollment}</p>
      <p className="text-xs text-muted-foreground">{copy.local}</p>
    </section>
  )
}
