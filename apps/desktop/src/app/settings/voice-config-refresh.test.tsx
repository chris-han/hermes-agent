import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { queryClient } from '@/lib/query-client'

import { VoiceProviderFields } from './voice-provider-fields'

const previousBridge = window.hermesDesktop

afterEach(() => {
  cleanup()
  queryClient.clear()
  window.hermesDesktop = previousBridge
})

describe('voice config refresh after a language save', () => {
  it.each([false, true])('refreshes clean controls and preserves pending custom edits (dirty=%s)', async dirty => {
    let record = { tts: { edge: { voice: 'en-US-AriaNeural' } } }
    const api = vi.fn(async request => (request.path.includes('schema') ? { fields: {} } : structuredClone(record)))
    window.hermesDesktop = { ...previousBridge, api } as typeof window.hermesDesktop
    render(
      <QueryClientProvider client={queryClient}>
        <VoiceProviderFields providerKey="edge" section="tts" />
      </QueryClientProvider>
    )
    const input = await screen.findByDisplayValue('en-US-AriaNeural')

    if (dirty) {
      fireEvent.change(input, { target: { value: 'en-US-GuyNeural' } })
    }

    record = { tts: { edge: { voice: 'zh-CN-XiaoxiaoNeural' } } }
    await act(async () => {
      await queryClient.invalidateQueries({ queryKey: ['hermes-config-record'] })
    })
    await waitFor(() =>
      expect((input as HTMLInputElement).value).toBe(dirty ? 'en-US-GuyNeural' : 'zh-CN-XiaoxiaoNeural')
    )
    expect(api.mock.calls.every(([request]) => request.method !== 'PUT')).toBe(true)
  })
})
