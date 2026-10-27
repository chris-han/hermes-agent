import type { IpcMain, IpcMainInvokeEvent } from 'electron'

import type { DeliverySourceId, DeliverySources, DeliverySourceStatus } from './delivery-source'
import type { UpdateOperation } from './operation'

import type { UpdaterStatusWire } from './index'

interface Deps {
  assertViewer: (event: IpcMainInvokeEvent) => void
  eligible: () => boolean
  busy: () => boolean
  sources: DeliverySources
  operation: Pick<UpdateOperation, 'reset' | 'check' | 'checkStatus'>
  changed: (selection: DeliverySourceStatus) => void
  check: () => Promise<UpdaterStatusWire>
}

/** Source mutation is privileged native composition; the renderer supplies only an enum. */
export function registerDeliverySourceIpc(ipc: Pick<IpcMain, 'handle'>, deps: Deps): void {
  let transitioning = false
  const selection = (): DeliverySourceStatus => ({ ...deps.sources.status(), checkStatus: deps.operation.checkStatus })

  ipc.handle('hermes:updates:delivery:get', event => {
    deps.assertViewer(event)

    return deps.eligible() ? selection() : null
  })
  ipc.handle('hermes:updates:delivery:set', async (event, source: unknown) => {
    deps.assertViewer(event)

    if (!deps.eligible()) { throw new Error('This installation cannot select an update delivery source') }

    if (transitioning || deps.busy()) { throw new Error('An update is already in progress.') }
    transitioning = true

    try {
      // Validate endpoint projection before commit; after save, use captured
      // non-secret options so a later config read cannot masquerade as a save error.
      const options = deps.sources.status().options
      deps.operation.reset()
      deps.sources.select(source)

      const savedSelection = (checkStatus: DeliverySourceStatus['checkStatus']): DeliverySourceStatus => ({
        selected: source as DeliverySourceId, options, checkStatus
      })

      deps.changed(savedSelection('checking'))
      let status: UpdaterStatusWire

      try {
        status = await deps.operation.check(deps.check)
      } catch {
        status = { supported: true, error: 'check-failed', message: 'The saved update source could not be checked.' }
      }

      const checkStatus = status.error ? 'failed' : 'completed'
      const saved = savedSelection(checkStatus)
      deps.changed(saved)

      return { saved: true, selection: saved, check_status: checkStatus, status }
    } finally {
      transitioning = false
    }
  })
}
