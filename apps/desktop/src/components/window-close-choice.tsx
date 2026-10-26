import { useEffect, useRef, useState } from 'react'

import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { useI18n } from '@/i18n'
import { windowCloseCopy } from '@/i18n/window-close'

import type { WindowCloseRequest } from '../../electron/minimize-to-tray'

function WindowCloseDialog({ request, onDismiss }: { request: WindowCloseRequest; onDismiss: (id: string) => void }) {
  const { locale } = useI18n()
  const copy = windowCloseCopy[locale] ?? windowCloseCopy.en
  const answered = useRef(false)
  const answer = window.hermesDesktop?.minimizeToTray?.answerClose

  return <ConfirmDialog
    confirmLabel={copy.keep}
    description={`${copy.question}\n\n${copy.detail}`}
    onClose={() => {
      if (!answered.current) {
        answered.current = true
        void answer?.(request.id, 'cancel').catch(() => undefined)
      }

      onDismiss(request.id)
    }}
    onConfirm={async () => {
      const result = await answer?.(request.id, 'background')

      if (!result?.ok) {
        throw new Error(result?.reason === 'background-unavailable' ? copy.backgroundUnavailable : copy.unavailable)
      }

      answered.current = true
    }}
    open
    secondaryAction={{ label: copy.quit, onClick: () => {
      answered.current = true
      void answer?.(request.id, 'quit').catch(() => undefined)
    } }}
    title={copy.title}
  />
}

/** Main owns the lifecycle request. Subscribe before reading its snapshot so a
 * close arriving before React mounts is recovered without replacing a newer one. */
export function WindowCloseChoiceHost() {
  const [request, setRequest] = useState<WindowCloseRequest | null>(null)

  useEffect(() => {
    const api = window.hermesDesktop?.minimizeToTray

    if (!api?.onCloseRequested || !api.getCloseRequest || !api.answerClose) { return }
    let live = true
    let revision = 0

    const stop = api.onCloseRequested(next => {
      revision += 1

      if (live) { setRequest(next) }
    })

    const readRevision = revision
    void api.getCloseRequest().then(next => {
      if (live && revision === readRevision) { setRequest(next) }
    }).catch(() => undefined)

    return () => { live = false; stop() }
  }, [])

  return request ? <WindowCloseDialog key={request.id} onDismiss={id => setRequest(current => current?.id === id ? null : current)} request={request} /> : null
}
