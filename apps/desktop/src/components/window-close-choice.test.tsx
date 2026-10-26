import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { WindowCloseChoiceHost } from './window-close-choice'

const answer = vi.fn()
let receive: (request: { id: string }) => void
const previous = Object.getOwnPropertyDescriptor(window, 'hermesDesktop')
beforeEach(() => {
  answer.mockReset().mockResolvedValue({ ok: true })
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { minimizeToTray: {
    getCloseRequest: async () => ({ id: 'request-1' }),
    answerClose: answer,
    onCloseRequested: (callback: typeof receive) => { receive = callback;

 return vi.fn() }
  } } })
})
afterEach(() => {
  cleanup()

  if (previous) { Object.defineProperty(window, 'hermesDesktop', previous) }
  else { Reflect.deleteProperty(window, 'hermesDesktop') }
})

it('recovers an early native request, focuses Keep running, and cancels only this request', async () => {
  render(<WindowCloseChoiceHost />)
  const dialog = await screen.findByRole('dialog', { name: 'Close Semantier Agent' })
  expect(dialog.textContent).toContain('separately managed gateway')
  const keep = screen.getByRole('button', { name: 'Keep running in background' })
  await waitFor(() => expect(keep.ownerDocument.activeElement).toBe(keep))
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  expect(answer).toHaveBeenCalledWith('request-1', 'cancel')
  expect(screen.queryByRole('dialog')).toBeNull()
})

it('shows tray failure inline, allows retry, and never reports background success on failure', async () => {
  answer.mockResolvedValueOnce({ ok: false, reason: 'background-unavailable' })
  render(<WindowCloseChoiceHost />)
  fireEvent.click(await screen.findByRole('button', { name: 'Keep running in background' }))
  await screen.findByText('The system tray is unavailable. The window will stay open. Try again or choose Quit.')
  expect(screen.getByRole('dialog')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Keep running in background' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull(), { timeout: 2000 })
  expect(answer.mock.calls).toEqual([['request-1', 'background'], ['request-1', 'background']])
})

it('localizes the three choices and sends Quit through the native lifecycle owner', async () => {
  render(<I18nProvider configClient={null} initialLocale="zh"><WindowCloseChoiceHost /></I18nProvider>)
  await screen.findByRole('dialog', { name: '关闭 Semantier Agent' })
  expect(screen.getByRole('button', { name: '在后台运行' })).toBeTruthy()
  expect(screen.getByRole('button', { name: '取消' })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: '退出 Semantier Agent' }))
  expect(answer.mock.calls).toEqual([['request-1', 'quit']])
})

it('does not let a late initial snapshot replace a newer pushed request', async () => {
  let finish!: (value: { id: string }) => void
  window.hermesDesktop.minimizeToTray!.getCloseRequest = () => new Promise(resolve => { finish = resolve })
  render(<WindowCloseChoiceHost />)
  act(() => receive({ id: 'newer-request' }))
  await screen.findByRole('dialog')
  await act(async () => finish({ id: 'old-request' }))
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  expect(answer.mock.calls).toEqual([['newer-request', 'cancel']])
})
