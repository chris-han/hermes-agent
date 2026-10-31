import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { capabilityScoped } from '@/api/client'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { confirm } from '@/store/confirm'
import { $defaultProfileRoute, refreshDefaultProfile } from '@/store/default-profile'
import { $desktopVersion, $updateStatus } from '@/store/updates'

interface Preview {
  payload: { submission_key: string; note: string; sections: Record<string, string>; app_version: string }
  bytes: number
  redactions: number
  retention_days: number
  destination: string
}
interface Receipt {
  diagnostic_id: string
  received_at: string
  expires_at: string
  deleted_at: string | null
}

function feedback<T>(profile: string, operation: string, body: object = {}) {
  return window.hermesDesktop.api<T>({
    ...capabilityScoped({ connectionId: LOCAL_CONNECTION_ID, profile }),
    path: `/api/client-feedback/${operation}`,
    method: 'POST',
    body
  })
}

// The native API bridge serializes HTTP errors as "status: JSON", sometimes
// wrapped by Electron IPC. Do not classify numbers inside arbitrary log text.
function feedbackFailure(error: unknown) {
  const message = error instanceof Error ? error.message : String(error)
  const match = message.match(/(?:^\s*|error:\s*)([45]\d\d):\s*(.*)$/i)
  const status = Number(match?.[1] ?? 0)
  let retryAfter = 30

  if (status === 429) {
    try {
      const value = Number(JSON.parse(match?.[2] ?? '{}')?.detail?.retry_after_seconds)

      if (Number.isFinite(value) && value > 0) {retryAfter = Math.max(1, Math.min(Math.ceil(value), 900))}
    } catch {
      /* Missing server guidance uses the bounded local retry delay. */
    }
  }

  return { status, retryAfter }
}

interface FeedbackState {
  enabled: boolean
  diagnostic_signed_in: boolean
  authorization_pending?: string
}
interface AuthorizationState {
  purpose: 'diagnostic' | 'reporting'
  authorization_url: string
  expires_at: string
}

/** Dedicated feedback authorization never calls the model-provider owner. */
function useFeedbackAuthorization(profile: string) {
  const [pending, setPending] = useState(false)
  const [boundState, setState] = useState<{ profile: string; value: FeedbackState } | null>(null)
  const state = boundState?.profile === profile ? boundState.value : null
  const [error, setError] = useState(false)

  const refresh = useCallback(async () => {
    if (!profile) {
      return
    }

    const current = await feedback<FeedbackState>(profile, 'status')
    setState({ profile, value: current })
    setPending(!!current.authorization_pending)
  }, [profile])

  useEffect(() => {
    void refresh().catch(() => setError(true))
  }, [refresh])
  useEffect(() => {
    if (!profile || !pending) {
      return
    }

    let live = true
    let running = false

    const poll = async () => {
      if (running) {
        return
      }

      running = true

      try {
        const reply = await feedback<{ status: string; purpose?: string }>(profile, 'authorization_poll')

        if (!live) {
          return
        }

        if (reply.status === 'approved') {
          if (reply.purpose === 'reporting') {
            await window.hermesDesktop.semantierFeedback?.consent(true, profile)
          }

          await refresh()
          setError(false)
        } else if (reply.status !== 'pending') {
          setPending(false)
          setError(true)
        }
      } catch {
        if (live) {
          setError(true)
        }
      } finally {
        running = false
      }
    }

    void poll()
    const timer = setInterval(() => void poll(), 5000)

    return () => {
      live = false
      clearInterval(timer)
    }
  }, [profile, pending, refresh])

  async function begin(purpose: 'diagnostic' | 'reporting') {
    setError(false)

    try {
      const reply = await feedback<AuthorizationState>(profile, 'authorization_start', { purpose })
      setPending(true)
      await window.hermesDesktop.openExternal(reply.authorization_url)
    } catch {
      setError(true)
    }
  }

  async function cancel() {
    setPending(false)

    try {
      await feedback(profile, 'authorization_cancel')
      setError(false)
    } catch {
      setError(true)
    }
  }

  return { state, pending, error, begin, cancel, refresh }
}

export const openSemantierDiagnostics = () => window.dispatchEvent(new Event('semantier:send-diagnostics'))

export function SemantierFeedbackSettings() {
  const { t: translations } = useI18n()
  const copy = translations.semantierFeedback
  const route = useStore($defaultProfileRoute)

  const authorization = useFeedbackAuthorization(
    !route?.connectionId || route.connectionId === LOCAL_CONNECTION_ID ? (route?.profile ?? '') : ''
  )

  const enabled = authorization.state?.enabled ?? false
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [journeyEnabled,setJourneyEnabled] = useState(false)
  useEffect(() => {let live=true;void window.hermesDesktop.semantierFeedback?.status().then(value=>{if(live)setJourneyEnabled(value.journeyEnabled===true)}).catch(()=>undefined);return()=>{live=false}},[])

  async function consent() {
    if (!route?.profile || (route.connectionId && route.connectionId !== LOCAL_CONNECTION_ID)) {
      return
    }

    setBusy(true)
    setError('')

    try {
      if (enabled) {
        await window.hermesDesktop.semantierFeedback?.consent(false, route.profile)
        await feedback(route.profile, 'consent', { enabled: false })
        await authorization.refresh()
      } else {
        await authorization.begin('reporting')
      }
    } catch {
      setError(
        copy.reportingCouldNotChangeSignInOr
      )
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="mt-4 grid gap-3">
      <h3>{copy.semantierDiagnosticsPrivacy}</h3>
      <p>{copy.optionalUpdateResultReportingToSemantierCloud}</p>
      <label className="flex items-start gap-2">
        <input type="checkbox" checked={journeyEnabled} disabled={!window.hermesDesktop.semantierFeedback?.journeyConsent}
          onChange={event=>{const enabled=event.target.checked;void window.hermesDesktop.semantierFeedback?.journeyConsent?.(enabled).then(value=>setJourneyEnabled(value.journeyEnabled)).catch(()=>setError(copy.reportingCouldNotChangeSignInOr))}} />
        <span>{copy.journeyObservationConsent}<small className="block">{copy.journeyObservationLimit}</small></span>
      </label>
      <p>
        {copy.reporting}: {enabled ? copy.on : copy.off}
      </p>
      <Button
        disabled={
          busy ||
          authorization.pending ||
          !route?.profile ||
          (!!route.connectionId && route.connectionId !== LOCAL_CONNECTION_ID)
        }
        onClick={() => void consent()}
        variant="secondary"
      >
        {enabled ? copy.stopReportingAndClearUnsentEvents : copy.enableUpdateResultReporting}
      </Button>
      <Button data-semantier-diagnostics-trigger onClick={openSemantierDiagnostics} variant="secondary">
        {copy.sendDiagnostics}
      </Button>
      {authorization.pending && (
        <p role="status">
          {copy.waitingForAuthorizationInYourBrowser}{' '}
          <Button onClick={() => void authorization.cancel()} variant="secondary">
            {copy.cancel}
          </Button>
        </p>
      )}
      {(error || authorization.error) && <p role="alert">{error || copy.authorizationNotConfirmedRetryOrCancelThis}</p>}
    </section>
  )
}

/** Consent-bound outbox drains only to its original local profile. No inference. */
export function SemantierFeedbackHost() {
  const { locale, t: translations } = useI18n()
  const copy = translations.semantierFeedback
  const [profile, setProfile] = useState('')
  const returnFocus = useRef<HTMLElement | null>(null)

  const rememberReturnFocus = useCallback(() => {
    returnFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
  }, [])

  const authorization = useFeedbackAuthorization(profile)
  const [preview, setPreview] = useState<Preview | null>(null)
  const [receipt, setReceipt] = useState<Receipt | null>(null)
  const [note, setNote] = useState('')
  const [include, setInclude] = useState(false)
  const [error, setError] = useState('')
  const [sending, setSending] = useState(false)
  const [draftKey, setDraftKey] = useState('')
  const [submitted, setSubmitted] = useState(false)
  const [retryAfter, setRetryAfter] = useState(0)
  const version = useStore($desktopVersion)
  const update = useStore($updateStatus)
  useEffect(() => {
    if (!retryAfter) {return}
    const timer = setTimeout(() => setRetryAfter(value => Math.max(0, value - 1)), 1000)

    return () => clearTimeout(timer)
  }, [retryAfter])
  useEffect(() => {
    let live = true

    const open = async () => {
      const route = await refreshDefaultProfile()

      if (!live) {
        return
      }

      if (!route?.profile || (route.connectionId && route.connectionId !== LOCAL_CONNECTION_ID)) {
        setError(copy.selectAnIntendedLocalProfileToPrepare)

        return
      }

      setDraftKey(crypto.randomUUID())
      setSubmitted(false)
      setRetryAfter(0)
      setInclude(false)
      setProfile(route.profile)
      setPreview(null)
      setReceipt(null)
      setNote('')
      setError('')
    }

    const listener = () => {
      rememberReturnFocus()
      void open()
    }

    window.addEventListener('semantier:send-diagnostics', listener)
    const stop = window.hermesDesktop.semantierFeedback?.onOpen?.(listener)

    return () => {
      live = false
      window.removeEventListener('semantier:send-diagnostics', listener)
      stop?.()
    }
  }, [locale, copy, rememberReturnFocus])
  useEffect(() => {
    let live = true
    let running = false

    async function drain() {
      if (running || !live) {
        return
      }

      running = true

      try {
        const api = window.hermesDesktop.semantierFeedback
        const state = await api?.status()

        if (!state?.enabled || !state.profile) {
          return
        }

        const events = await api?.drain(state.profile)

        if (events?.length) {
          await feedback(state.profile, 'record', { events })
          await api?.ack(
            state.profile,
            events.map(e => e.event_id)
          )
        } else {
          await feedback(state.profile, 'flush')
        }

        const owner = await feedback<FeedbackState>(state.profile, 'status')

        if (!owner.enabled) {
          await api?.consent(false, state.profile)
        }
      } catch {
        /* Bounded owner queue retains offline events. */
      } finally {
        running = false
      }
    }

    void drain()
    const timer = setInterval(() => void drain(), 30000)

    return () => {
      live = false
      clearInterval(timer)
    }
  }, [])
  useEffect(() => {
    if (!profile || receipt) {
      return
    }

    let live = true
    const app = version?.appVersion ?? ''
    const appVersion = /^\d+\.\d+\.\d+$/.test(app) ? app + '.0' : app
    void feedback<Preview>(profile, 'preview', {
      submission_key: draftKey,
      app_version: appVersion,
      release_id: update?.installedReleaseId ?? null,
      note,
      include_update: include,
      context: JSON.stringify({
        code: update?.error ?? null,
        mechanism: update?.mechanism ?? null,
        channel: update?.channel ?? null
      })
    })
      .then(p => {
        if (live) {
          setPreview(p)
        }
      })
      .catch(() => {
        if (live) {
          setError(
            copy.couldNotPrepareASanitizedPreviewRetry
          )
        }
      })

    return () => {
      live = false
    }
  }, [
    profile,
    note,
    include,
    version?.appVersion,
    receipt,
    update?.installedReleaseId,
    update?.channel,
    update?.error,
    update?.mechanism,
    draftKey,
    copy
  ])

  const close = () => {
    if (authorization.pending) {
      void authorization.cancel()
    }

    setProfile('')
    setPreview(null)
    setReceipt(null)
    setNote('')
    setError('')
    setDraftKey('')
    setSubmitted(false)
    setRetryAfter(0)
  }

  async function send() {
    if (receipt) {
      close()

      return
    }

    if (!preview || sending || retryAfter) {
      return
    }

    setSubmitted(true)
    setSending(true)
    setError('')

    try {
      const reply = await feedback<{ receipt: Receipt }>(profile, 'submit', preview.payload)
      setReceipt(reply.receipt)
    } catch (failure) {
      const recovery = feedbackFailure(failure)

      const messages: Record<number, string> = {
        401: copy.reportSignInAgain,
        403: copy.reportPermissionDenied,
        413: copy.reportTooLarge,
        422: copy.reportContentRejected,
        429: copy.reportRateLimited,
        503: copy.reportServiceUnavailable
      }

      setError(messages[recovery.status] ?? copy.reportNotConfirmedSignInOrRetry)

      if (recovery.status === 413 || recovery.status === 422) {
        // Definite rejection permits a new editable draft. Ambiguous replies
        // retain the exact original payload/key for receipt recovery.
        setSubmitted(false)
        setDraftKey(crypto.randomUUID())
      }

      if (recovery.status === 429) {setRetryAfter(recovery.retryAfter)}

      if (recovery.status === 401) {void authorization.refresh().catch(() => undefined)}
    } finally {
      setSending(false)
    }
  }

  return (
    <Dialog
      onOpenChange={open => {
        if (!open && !sending) {
          close()
        }
      }}
      open={!!profile}
    >
      <DialogContent
        onCloseAutoFocus={event => {
          event.preventDefault()
          const origin = returnFocus.current

          const target = origin?.isConnected && origin !== document.body
            ? origin
            : document.querySelector<HTMLElement>('[data-semantier-diagnostics-trigger]')

          target?.focus()
          returnFocus.current = null
        }}
      >
        <DialogHeader>
          <DialogTitle>{receipt ? copy.diagnosticReceipt : copy.sendDiagnostics}</DialogTitle>
          <DialogDescription>{copy.reviewTheSanitizedContentBeforeSendingTo}</DialogDescription>
        </DialogHeader>
        {receipt ? (
          <div className="grid gap-3">
            <p role="status">{copy.reportSent}</p>
            <p>{receipt.diagnostic_id}</p>
            <p>
              {receipt.received_at} · {copy.expires}: {receipt.expires_at}
            </p>
            <Button onClick={() => void navigator.clipboard.writeText(receipt.diagnostic_id)} variant="secondary">
              {copy.copyReportID}
            </Button>
            <Button
              disabled={!!receipt.deleted_at}
              onClick={() => {
                void confirm({
                  title: copy.removeUploadedLog,
                  description: receipt.diagnostic_id + ' · ' + copy.minimalReceiptAndAuditMetadataRemain,
                  onConfirm: async () => {
                    const r = await feedback<{ receipt: Receipt }>(profile, 'erase', {
                      diagnostic_id: receipt.diagnostic_id
                    })

                    setReceipt(r.receipt)
                  }
                })
              }}
              variant="secondary"
            >
              {receipt.deleted_at ? copy.uploadedLogRemoved : copy.removeUploadedLog}
            </Button>
          </div>
        ) : (
          <div className="grid gap-3">
            <p>
              {copy.originalLocalProfile}: {profile}
            </p>
            <label>
              {copy.optionalProblemDescription}
              <Textarea
                disabled={submitted}
                maxLength={4000}
                onChange={e => {
                  setPreview(null)
                  setNote(e.target.value)
                }}
                value={note}
              />
            </label>
            <label>
              <input
                checked={include}
                disabled={submitted}
                onChange={e => {
                  setPreview(null)
                  setInclude(e.target.checked)
                }}
                type="checkbox"
              />{' '}
              {copy.includeStructuredUpdateError}
            </label>
            <p>
              {preview?.bytes ?? '—'} {copy.uTF8Bytes} · {preview?.redactions ?? '—'} {copy.redactions} ·{' '}
              {copy.payloadRetained30Days}
            </p>
            <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-all">
              {preview
                ? JSON.stringify({ note: preview.payload.note, sections: preview.payload.sections }, null, 2)
                : copy.preparingPreview}
            </pre>
            <p>{copy.excludesConversationsDocumentsArbitraryFilesAndCrash}</p>
            <Button
              disabled={authorization.pending}
              onClick={() => void authorization.begin('diagnostic')}
              variant="secondary"
            >
              {authorization.state?.diagnostic_signed_in ? copy.renewDiagnosticAuthorization : copy.signInToSend}
            </Button>
            {authorization.pending && (
              <p role="status">
                {copy.waitingForBrowserAuthorizationYourPreviewStays}{' '}
                <Button onClick={() => void authorization.cancel()} variant="secondary">
                  {copy.cancel}
                </Button>
              </p>
            )}
            {authorization.error && <p role="alert">{copy.authorizationNotConfirmedRetryOrCancel}</p>}
          </div>
        )}
        {error && (
          <p role="alert">
            {error === copy.reportRateLimited
              ? retryAfter > 0
                ? copy.reportRateLimited.replace('{seconds}', String(retryAfter))
                : copy.reportRetryReady
              : error}
          </p>
        )}
        <DialogFooter>
          <Button disabled={sending} onClick={close} variant="secondary">
            {copy.close}
          </Button>
          <Button
            disabled={
              sending || retryAfter > 0 || (!receipt && (!preview || !authorization.state?.diagnostic_signed_in))
            }
            onClick={() => void send()}
          >
            {sending ? copy.sending : receipt ? copy.done : copy.sendDiagnosticReport}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
