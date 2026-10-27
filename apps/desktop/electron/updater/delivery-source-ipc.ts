import type { IpcMain, IpcMainInvokeEvent } from 'electron'

import type { DeliverySources, DeliverySourceStatus } from './delivery-source'
import type { UpdateOperation } from './operation'

import type { UpdaterStatusWire } from './index'

interface Deps {
  assertViewer: (event: IpcMainInvokeEvent) => void
  eligible: () => boolean
  busy: () => boolean
  sources: DeliverySources
  operation: Pick<UpdateOperation, 'reset'>
  changed: (selection: DeliverySourceStatus) => void
  check: () => Promise<UpdaterStatusWire>
}

/** Source mutation is privileged native composition; the renderer supplies only an enum. */
export function registerDeliverySourceIpc(ipc: Pick<IpcMain, 'handle'>, deps: Deps): void {
  ipc.handle('hermes:updates:delivery:get', event => {
    deps.assertViewer(event)

    return deps.eligible() ? deps.sources.status() : null
  })
  ipc.handle('hermes:updates:delivery:set', async (event, source: unknown) => {
    deps.assertViewer(event)

    if (!deps.eligible()) { throw new Error('This installation cannot select an update delivery source') }

    if (deps.busy()) { throw new Error('An update is already in progress.') }
    deps.operation.reset()
    deps.sources.select(source)
    const selection = deps.sources.status()
    deps.changed(selection)

    return { selection, status: await deps.check() }
  })
}
