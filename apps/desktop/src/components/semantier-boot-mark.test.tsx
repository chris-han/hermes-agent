import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { SemantierBootMark } from './semantier-boot-mark'

beforeEach(() => {
  vi.useFakeTimers()
  window.matchMedia = vi.fn().mockReturnValue({
    addEventListener: vi.fn(),
    matches: false,
    removeEventListener: vi.fn()
  })
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

it('keeps each prefix legible, resolves the words in order, and stops at SEMANTIER', () => {
  const { container } = render(<SemantierBootMark active />)
  const seen: string[] = []

  for (let time = 0; time < 4500; time += 45) {
    const word = container.querySelector('[data-boot-word]')?.textContent ?? ''
    expect(word).toMatch(/^(SĒMA|SÉMA|SEMA)/)

    if (['SĒMANTIKÓS', 'SÉMANTIQUE', 'SEMANTIER'].includes(word) && seen.at(-1) !== word) {
      seen.push(word)
    }

    act(() => vi.advanceTimersByTime(45))
  }

  expect(seen).toEqual(['SĒMANTIKÓS', 'SÉMANTIQUE', 'SEMANTIER'])
  expect(container.querySelector('[data-boot-word]')?.textContent).toBe('SEMANTIER')
  expect(vi.getTimerCount()).toBe(0)
})

it('settles immediately on readiness or reduced motion', () => {
  const { container, rerender } = render(<SemantierBootMark active />)
  act(() => vi.advanceTimersByTime(90))
  rerender(<SemantierBootMark active={false} />)
  expect(container.querySelector('[data-boot-word]')?.textContent).toBe('SEMANTIER')
  expect(vi.getTimerCount()).toBe(0)

  window.matchMedia = vi.fn().mockReturnValue({
    addEventListener: vi.fn(),
    matches: true,
    removeEventListener: vi.fn()
  })
  rerender(<SemantierBootMark active />)
  expect(container.querySelector('[data-boot-word]')?.textContent).toBe('SEMANTIER')
  expect(vi.getTimerCount()).toBe(0)
})
