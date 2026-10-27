import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { deliverySourceCopy } from '@/i18n/delivery-source'
import { $updateApply, $updateChecking, $updateStatus } from '@/store/updates'

import type { DeliverySourceId, DeliverySourceStatus } from '../../../electron/updater/delivery-source'

/** Native installation setting, independent of gateway connection and release track. */
export function UpdateDeliverySourceRow() {
  const { locale } = useI18n()
  const copy = deliverySourceCopy[locale] ?? deliverySourceCopy.en
  const [selection, setSelection] = useState<DeliverySourceStatus | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const checking = useStore($updateChecking)
  const applying = useStore($updateApply)
  useEffect(() => {
    const bridge = window.hermesDesktop?.updates
    let live = true
    void bridge?.getDeliverySources?.().then(value => { if (live) {setSelection(value)} })
      .catch(() => { if (live) {setError(copy.failed)} })

    const unsubscribe = bridge?.onDeliverySourceChanged?.(value => {
      setSelection(value)
      $updateStatus.set(null)
    })

    return () => { live = false; unsubscribe?.() }
  }, [copy.failed])

  if (!selection) {return error ? <p role="alert">{error}</p> : null}

  const change = async (value: string) => {
    const bridge = window.hermesDesktop?.updates

    if (!bridge?.setDeliverySource || pending || checking || applying.applying) {return}
    setPending(true)
    setError('')
    $updateStatus.set(null)
    $updateChecking.set(true)

    try {
      const result = await bridge.setDeliverySource(value as DeliverySourceId)
      setSelection(result.selection)
      $updateStatus.set(result.status)
    } catch {
      setError(copy.failed)
      const saved = await bridge.getDeliverySources?.().catch(() => null)

      if (saved) {setSelection(saved)}
    } finally {
      $updateChecking.set(false)
      setPending(false)
    }
  }

  const endpoint = selection.options.find(row => row.id === selection.selected)?.publicBase

  return (
    <div className="grid min-w-0 gap-2">
      <label className="text-sm" htmlFor="update-delivery-source">{copy.title}</label>
      <Select disabled={pending || checking || applying.applying} onValueChange={value => void change(value)} value={selection.selected}>
        <SelectTrigger aria-label={copy.title} className="w-full min-w-0" id="update-delivery-source">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {selection.options.map(option => <SelectItem key={option.id} value={option.id}>{copy[option.id]}</SelectItem>)}
        </SelectContent>
      </Select>
      <p className="break-all text-xs text-muted-foreground">{endpoint || copy.missing}</p>
      {error && <p className="text-sm text-destructive" role="alert">{error}</p>}
    </div>
  )
}
