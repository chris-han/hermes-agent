import type { IpcMain, IpcMainInvokeEvent } from 'electron'

import type { SemantierObservations } from './semantier-observations'

/** Main-frame application viewers only; reporting stays optional and profile scoped. */
export function registerSemantierFeedbackIpc(
  ipc: Pick<IpcMain, 'handle'>,
  observations: SemantierObservations,
  assertViewer: (event: IpcMainInvokeEvent) => void
): void {
  ipc.handle(
    'semantier:feedback',
    (event, command: string, value: { enabled?: boolean; profile?: string; ids?: string[] }) => {
      assertViewer(event)

      if (command === 'status') {
        return observations.status()
      }

      if (command === 'journey-consent' && typeof value?.enabled === 'boolean') {
        observations.journeyConsent(value.enabled)

        return observations.status()
      }

      const profile = value?.profile

      if (typeof profile !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(profile)) {
        throw new Error('Original local profile required')
      }

      if (command === 'consent' && typeof value.enabled === 'boolean') {
        observations.consent(value.enabled, profile)

        return observations.status()
      }

      if (command === 'drain') {
        return observations.drain(profile)
      }

      if (
        command === 'ack' &&
        Array.isArray(value.ids) &&
        value.ids.length <= 100 &&
        value.ids.every(id => typeof id === 'string' && id.length <= 128)
      ) {
        observations.ack(profile, value.ids)

        return observations.status()
      }

      throw new Error('Unsupported feedback operation')
    }
  )
}
