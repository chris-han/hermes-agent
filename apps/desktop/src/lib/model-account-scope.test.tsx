import { act, renderHook, waitFor } from '@testing-library/react'
import { expect, it, vi } from 'vitest'

import { useModelAccountScope, visibleAccountModelOptions } from './model-account-scope'

it('shares one account generation across simultaneous model consumers', async () => {
  const original = Object.getOwnPropertyDescriptor(window, 'hermesDesktop')

  const account = (userId: string) => ({ connections: [], accountContext: { origin: 'https://cloud.example',
    scope: { userId, workspaceId: `w-${userId}`, organizationId: 'o' } } })

  let notify!: () => void
  const list = vi.fn().mockResolvedValueOnce(account('a')).mockResolvedValue(account('b'))
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {
    connections: { list }, workspace: { onAccountChanged: (callback: () => void) => {notify = callback;

 return () => {}} }
  } })
  const connection = { mode: 'local' as const, baseUrl: 'http://127.0.0.1', connectionId: 'local' }
  const first = renderHook(() => useModelAccountScope(connection, 'local'))
  const second = renderHook(() => useModelAccountScope(connection, 'local'))

  try {
    await waitFor(() => expect(first.result.current.allowCloud).toBe(true))
    expect(first.result.current.key).toEqual(second.result.current.key)
    expect(list).toHaveBeenCalledTimes(1)
    act(() => notify())
    expect(first.result.current.allowCloud).toBe(false)
    expect(second.result.current.allowCloud).toBe(false)
    await waitFor(() => expect(first.result.current.allowCloud).toBe(true))
    expect(first.result.current.key).toEqual(second.result.current.key)
    expect(JSON.stringify(first.result.current.key)).toContain('w-b')
  } finally {
    first.unmount(); second.unmount()

    if (original) {Object.defineProperty(window, 'hermesDesktop', original)} else {delete (window as { hermesDesktop?: unknown }).hermesDesktop}
  }
})

it('suppresses only Cloud choices when the account is unresolved', () => {
  const data = { provider: 'semantier-cloud', model: 'old-model', providers: [
    { slug: 'semantier-cloud', name: 'Cloud', models: ['old-model'] },
    { slug: 'local', name: 'Local', models: ['local-model'] }
  ] }

  expect(visibleAccountModelOptions(data, false)).toMatchObject({ provider: '', model: '', providers: [{ slug: 'local' }] })
  expect(visibleAccountModelOptions(data, true)).toBe(data)
})

it('ignores an earlier registry response after account change', async () => {
  const original = Object.getOwnPropertyDescriptor(window, 'hermesDesktop')

  const registry = (userId: string) => ({ connections: [], accountContext: { origin: 'https://cloud.example',
    scope: { userId, workspaceId: `w-${userId}`, organizationId: 'o' } } })

  let oldReply!: (value: unknown) => void
  let notify!: () => void
  const list = vi.fn().mockImplementationOnce(() => new Promise(resolve => {oldReply = resolve})).mockResolvedValue(registry('b'))
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {
    connections: { list }, workspace: { onAccountChanged: (callback: () => void) => {notify = callback;

 return () => {}} }
  } })
  const connection = { mode: 'local' as const, baseUrl: 'http://127.0.0.1', connectionId: 'local' }
  const hook = renderHook(() => useModelAccountScope(connection, 'local'))

  try {
    await waitFor(() => expect(oldReply).toBeDefined())
    act(() => notify())
    await waitFor(() => expect(hook.result.current.allowCloud).toBe(true))
    await act(async () => {oldReply(registry('a'))})
    expect(JSON.stringify(hook.result.current.key)).toContain('w-b')
  } finally {
    hook.unmount()

    if (original) {Object.defineProperty(window, 'hermesDesktop', original)} else {delete (window as { hermesDesktop?: unknown }).hermesDesktop}
  }
})
