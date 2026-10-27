import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { deliverySourceCopy } from '@/i18n/delivery-source'
import { $updateApply, $updateChecking, $updateDeliverySource, $updateSourceChecking, $updateSourceLoading, $updateStatus, ingestDeliverySource } from '@/store/updates'

import type { DeliverySourceId } from '../../../electron/updater/delivery-source'

/** Native installation setting, independent of gateway connection and release track. */
export function UpdateDeliverySourceRow() {
  const { locale } = useI18n()
  const copy = deliverySourceCopy[locale] ?? deliverySourceCopy.en
  const selection = useStore($updateDeliverySource)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const generation = useRef(0)
  const pendingTarget = useRef<string | null>(null)
  const checking = useStore($updateChecking)
  const sourceChecking = useStore($updateSourceChecking)
  const sourceLoading = useStore($updateSourceLoading)
  const applying = useStore($updateApply)
  // eslint-disable-next-line no-restricted-syntax -- These refs track request epochs and pending intent, not reactive value mirrors.
  useEffect(() => {
    const bridge = window.hermesDesktop?.updates
    let live = true
    const readGeneration = generation.current
    const read = bridge?.getDeliverySources

    if (read) {
      $updateSourceLoading.set(true)
      void read().then(value => {
        if (live && generation.current === readGeneration) {

          if (value) { ingestDeliverySource(value) }

          if (!value) { $updateDeliverySource.set(null) }
          setError('')
        }
      }).catch(() => { if (live) { setError(copy.failed) } })
        .finally(() => { if (live) { $updateSourceLoading.set(false) } })
    }

    const unsubscribe = bridge?.onDeliverySourceChanged?.(value => {
      if (!live) { return }

      if (value.selected !== pendingTarget.current) { generation.current += 1 }
      setError('')
      ingestDeliverySource(value, true)
    })

    return () => {
      live = false
      generation.current += 1
      pendingTarget.current = null
      $updateSourceLoading.set(false)
      unsubscribe?.()
    }
  }, [copy.failed, copy.savedCheckFailed])

  if (!window.hermesDesktop?.updates?.getDeliverySources) { return null }

  if (!selection) { return error ? <p role="alert">{error}</p> : sourceLoading ? <p role="status">{copy.loading}</p> : null }

  const change = async (value: string) => {
    const bridge = window.hermesDesktop?.updates

    if (!bridge?.setDeliverySource || pending || checking || sourceChecking || applying.applying) {return}
    const requestGeneration = ++generation.current
    pendingTarget.current = value
    setPending(true)
    setError('')
    $updateStatus.set(null)
    $updateSourceChecking.set(true)

    try {
      const result = await bridge.setDeliverySource(value as DeliverySourceId)

      if (generation.current !== requestGeneration) { return }
      ingestDeliverySource(result.selection)
      setError('')
      $updateStatus.set(result.status)
    } catch {
      if (generation.current !== requestGeneration) { return }
      setError(copy.failed)
      const saved = await bridge.getDeliverySources?.().catch(() => null)

      if (saved && generation.current === requestGeneration) { ingestDeliverySource(saved) }
    } finally {
      pendingTarget.current = null

      if (generation.current === requestGeneration) { $updateSourceChecking.set($updateDeliverySource.get()?.checkStatus === 'checking') }
      setPending(false)
    }
  }

  const message = error || (selection.checkStatus === 'failed' ? copy.savedCheckFailed : '')
  const endpoint = selection.options.find(row => row.id === selection.selected)?.publicBase

  return (
    <div className="grid min-w-0 gap-2">
      <label className="text-sm" htmlFor="update-delivery-source">{copy.title}</label>
      <Select disabled={pending || checking || sourceChecking || applying.applying} onValueChange={value => void change(value)} value={selection.selected}>
        <SelectTrigger aria-label={copy.title} className="w-full min-w-0" id="update-delivery-source">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {selection.options.map(option => <SelectItem key={option.id} value={option.id}>{copy[option.id]}</SelectItem>)}
        </SelectContent>
      </Select>
      <p className="break-all text-xs text-muted-foreground">{endpoint || copy.missing}</p>
      {selection.checkStatus === 'checking' && <p role="status">{copy.savedChecking}</p>}
      {message && <p className="text-sm text-destructive" role="alert">{message}</p>}
    </div>
  )
}
