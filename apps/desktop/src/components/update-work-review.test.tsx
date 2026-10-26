import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $updateWorkReview, applyUpdates } from '@/store/updates'

import { UpdateWorkReviewDialog } from './update-work-review'

vi.mock('@/store/updates', async () => {
  const { atom } = await import('nanostores')

  return { $updateWorkReview: atom(undefined), applyUpdates: vi.fn() }
})
afterEach(() => { cleanup(); vi.clearAllMocks() })
const review = { token: 'bound-review', currentVersion: '0.0.39.0', targetVersion: '0.0.40.0', work: { count: 1, titles: ['Draft'] }, stopsWork: true }

test('Later does not apply, and closing an old review cannot dismiss a newer request', async () => {
  $updateWorkReview.set(review)
  render(<UpdateWorkReviewDialog review={review} />)
  expect((await screen.findByRole('dialog')).textContent).toContain('0.0.39.0 → 0.0.40.0')
  expect(screen.getByText('Draft')).toBeTruthy()
  $updateWorkReview.set({ ...review, token: 'newer-review' })
  fireEvent.click(screen.getByRole('button', { name: 'Later' }))
  expect(applyUpdates).not.toHaveBeenCalled()
  expect($updateWorkReview.get()?.token).toBe('newer-review')
  $updateWorkReview.set(review)
  fireEvent.click(screen.getByRole('button', { name: 'Later' }))
  expect($updateWorkReview.get()).toBeUndefined()
  expect(applyUpdates).not.toHaveBeenCalled()
})

test('Chinese confirmation sends only the bound token and keeps a failed update usable', async () => {
  vi.mocked(applyUpdates).mockResolvedValue({ ok: false, message: 'Verification failed' })
  $updateWorkReview.set(review)
  render(<I18nProvider configClient={null} initialLocale="zh"><UpdateWorkReviewDialog review={{ ...review, stopsWork: false }} /></I18nProvider>)
  expect((await screen.findByRole('dialog')).textContent).toContain('独立网关')
  fireEvent.click(screen.getByRole('button', { name: '更新并重启' }))
  await waitFor(() => expect(applyUpdates).toHaveBeenCalledWith({ workReviewToken: 'bound-review' }))
  expect(await screen.findByText('Verification failed')).toBeTruthy()
  expect(screen.getByRole('button', { name: '稍后' })).toBeTruthy()
})
