import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { openSemantierDiagnostics, SemantierFeedbackHost, SemantierFeedbackSettings } from './semantier-feedback'

vi.mock('@/store/default-profile', async () => {
  const { atom } = await import('nanostores')
  const route = { profile: 'semantier-local', connectionId: 'local' }

  return { $defaultProfileRoute: atom(route), refreshDefaultProfile: async () => route }
})
vi.mock('@/store/updates', async () => {
  const { atom } = await import('nanostores')

  return { $desktopVersion: atom({ appVersion: '0.0.39' }), $updateStatus: atom(null) }
})
const submit = vi.fn()

const api = vi.fn(async ({ path, body }: { path: string; body: Record<string, unknown> }) => {
  if (path.endsWith('/status')) {return { enabled: false, diagnostic_signed_in: true }}

  if (path.endsWith('/preview'))
    {return {
      payload: { ...body, note: body.note, sections: {} },
      bytes: 100,
      redactions: 0,
      retention_days: 30,
      destination: 'Semantier Cloud'
    }}

  if (path.endsWith('/submit')) {return submit(body)}
  throw new Error('Unexpected native request: ' + path)
})

beforeEach(() => {
  submit.mockReset()
  api.mockClear()
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { api, semantierFeedback: { status: async () => ({ enabled: false }) } }
  })
})
afterEach(cleanup)

async function open() {
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <SemantierFeedbackHost />
    </I18nProvider>
  )
  await act(async () => openSemantierDiagnostics())
  await screen.findByRole('dialog')
  await waitFor(() =>
    expect((screen.getByRole('button', { name: 'Send diagnostic report' }) as HTMLButtonElement).disabled).toBe(false)
  )
}

it.each([
  [401, 'Sign in again to send this report. Your preview is retained.'],
  [403, 'This account cannot upload diagnostics. Your preview is retained.'],
  [413, 'The report is too large. Remove the update excerpt or shorten the description.'],
  [422, 'The report content was rejected. Review the description and selected excerpt.'],
  [429, 'Too many requests. Retry is available in 2 seconds.'],
  [503, 'The feedback service is unavailable. Retry keeps the same report ID.']
])('shows the recovery for HTTP %s without losing the draft', async (status, message) => {
  submit.mockRejectedValue(
    new Error(`Error invoking remote method 'hermes:api': Error: ${status}: {"detail":{"retry_after_seconds":2}}`)
  )
  await open()
  const note = screen.getByRole('textbox', { name: 'Optional problem description' }) as HTMLTextAreaElement
  fireEvent.change(note, { target: { value: 'Retain this explanation' } })
  await waitFor(() =>
    expect((screen.getByRole('button', { name: 'Send diagnostic report' }) as HTMLButtonElement).disabled).toBe(false)
  )
  fireEvent.click(screen.getByRole('button', { name: 'Send diagnostic report' }))
  expect(await screen.findByRole('alert')).toHaveProperty('textContent', message)
  expect(note.value).toBe('Retain this explanation')
  expect(note.disabled).toBe(status !== 413 && status !== 422)

  if (status === 429)
    {expect((screen.getByRole('button', { name: 'Send diagnostic report' }) as HTMLButtonElement).disabled).toBe(true)}
})
it('retries an ambiguous reply with the exact same submitted payload', async () => {
  submit
    .mockRejectedValueOnce(new Error('503: unavailable'))
    .mockResolvedValueOnce({
      receipt: {
        diagnostic_id: 'retained-report',
        received_at: '2026-10-26T00:00:00Z',
        expires_at: '2026-11-25T00:00:00Z',
        deleted_at: null
      }
    })
  await open()
  fireEvent.click(screen.getByRole('button', { name: 'Send diagnostic report' }))
  await screen.findByRole('alert')
  fireEvent.click(screen.getByRole('button', { name: 'Send diagnostic report' }))
  await screen.findByText('Report sent')
  expect(submit).toHaveBeenCalledTimes(2)
  expect(submit.mock.calls[1][0]).toEqual(submit.mock.calls[0][0])
})

it('allows a manual retry after the bounded rate limit without sending automatically', async () => {
  submit.mockRejectedValue(new Error('429: {"detail":{"retry_after_seconds":2}}'))
  await open()
  vi.useFakeTimers()

  try {
    const button = screen.getByRole('button', { name: 'Send diagnostic report' }) as HTMLButtonElement
    await act(async () => fireEvent.click(button))
    expect(button.disabled).toBe(true)
    await act(async () => vi.advanceTimersByTimeAsync(1000))
    await act(async () => vi.advanceTimersByTimeAsync(1000))
    expect(button.disabled).toBe(false)
    expect(screen.getByRole('alert').textContent).toBe('You can retry now. The same report ID is retained.')
    expect(submit).toHaveBeenCalledTimes(1)
  } finally {
    vi.useRealTimers()
  }
})

it('allows a corrected new draft after a definite size rejection', async () => {
  submit.mockRejectedValueOnce(new Error('413: rejected')).mockResolvedValueOnce({ receipt: { diagnostic_id: 'corrected-report', received_at: '2026-10-26T00:00:00Z', expires_at: '2026-11-25T00:00:00Z', deleted_at: null } })
  await open()
  fireEvent.click(screen.getByRole('button', { name: 'Send diagnostic report' }))
  await screen.findByRole('alert')
  fireEvent.change(screen.getByRole('textbox', { name: 'Optional problem description' }), { target: { value: 'Shorter description' } })
  await waitFor(() => expect((screen.getByRole('button', { name: 'Send diagnostic report' }) as HTMLButtonElement).disabled).toBe(false))
  fireEvent.click(screen.getByRole('button', { name: 'Send diagnostic report' }))
  await screen.findByText('Report sent')
  expect(submit.mock.calls[1][0].submission_key).not.toBe(submit.mock.calls[0][0].submission_key)
  expect(submit.mock.calls[1][0].note).toBe('Shorter description')
})


it.each([false, true])('returns focus after Escape when the diagnostic action is replaced: %s', async replaceTrigger => {
  const content = (key: number) => (
    <I18nProvider configClient={null} initialLocale="en">
      <SemantierFeedbackSettings key={key} />
      <SemantierFeedbackHost />
    </I18nProvider>
  )

  const view = render(content(0))
  const trigger = screen.getByRole('button', { name: 'Send diagnostics' })
  trigger.focus()
  fireEvent.click(trigger)
  await screen.findByRole('dialog')

  if (replaceTrigger) {view.rerender(content(1))}

  fireEvent.keyDown(window.document, { key: 'Escape' })
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  await waitFor(() => expect(window.document.activeElement).toBe(screen.getByRole('button', { name: 'Send diagnostics' })))
})
