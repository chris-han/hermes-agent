import { expect, it, vi } from 'vitest'

vi.mock('electron', () => ({
  BrowserWindow: vi.fn(),
  ipcMain: { handle: vi.fn() },
  session: { fromPartition: vi.fn() }
}))

import { ipcMain, session } from 'electron'

import { assertWorkspaceScope, registerSemantierWorkspaceIpc } from './semantier-workspace-ipc'

const scope = { userId: 'user-a', workspaceId: 'workspace-a', organizationId: 'org-a' }
const identity = {
  authenticated: true,
  user: { user_id: 'user-a', workspace_id: 'workspace-a' },
  workspace: { id: 'workspace-a' },
  organization_id: 'org-a'
}

it('accepts only the current authenticated server scope', () => {
  expect(() => assertWorkspaceScope(identity, scope)).not.toThrow()

  for (const expected of [
    { ...scope, userId: 'user-b' },
    { ...scope, workspaceId: 'workspace-b' },
    { ...scope, organizationId: 'org-b' }
  ]) {
    expect(() => assertWorkspaceScope(identity, expected)).toThrow('changed')
  }

  expect(() => assertWorkspaceScope({ ...identity, authenticated: false }, scope)).toThrow('changed')
})

it('signals account invalidation only after clearing the origin-owned cookie jar', async () => {
  const order: string[] = []

  const owned = {
    fetch: vi.fn(),
    clearStorageData: vi.fn(async () => {
      order.push('cookies')
    }),
    clearCache: vi.fn(async () => {
      order.push('cache')
    })
  }

  vi.mocked(session.fromPartition).mockReturnValue(owned as never)
  vi.mocked(ipcMain.handle).mockClear()
  const changed = vi.fn(() => {
    order.push('changed')
  })
  registerSemantierWorkspaceIpc(changed)
  const handler = vi.mocked(ipcMain.handle).mock.calls.find(([name]) => name === 'hermes:workspace:logout')![1]
  await handler({} as never, 'https://cloud.example')
  expect(changed).toHaveBeenCalledWith('https://cloud.example', false)
  expect(order).toEqual(['cookies', 'cache', 'changed'])
})

it('forwards incremental reads through the same before-and-after account guard', async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(Response.json(identity))
    .mockResolvedValueOnce(Response.json({ sessionId: 'original', messages: [], nextCursor: '218' }))
    .mockResolvedValueOnce(Response.json({ ...identity, user: { ...identity.user, user_id: 'user-b' } }))

  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler = vi.mocked(ipcMain.handle).mock.calls.find(([name]) => name === 'hermes:workspace:history')![1]
  await expect(handler({} as never, 'https://cloud.example', scope, 'original', '218')).rejects.toThrow('changed')
  expect(fetch.mock.calls[1][0]).toBe('https://cloud.example/api/history?sessionKey=original&afterId=218')
  expect(fetch).toHaveBeenCalledTimes(3)
})

it('checks the authenticated scope before canonical Chat creation', async () => {
  const fetch = vi.fn().mockResolvedValue(Response.json({ ...identity, user: { ...identity.user, user_id: 'user-b' } }))
  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler = vi
    .mocked(ipcMain.handle)
    .mock.calls.find(([name]) => name === 'hermes:workspace:ensure-winman-chat')![1]
  await expect(handler({ sender: { id: 99 } } as never, 'https://cloud.example', scope)).rejects.toThrow('changed')
  expect(fetch).toHaveBeenCalledTimes(1)
  expect(fetch.mock.calls[0][0]).toBe('https://cloud.example/auth/context')
})

it.each(['native-context', 'bind-native-context', 'send-native'])(
  'rejects a changed account before %s can access its captured Native owner',
  async name => {
    const fetch = vi
      .fn()
      .mockResolvedValue(Response.json({ ...identity, user: { ...identity.user, user_id: 'user-b' } }))
    vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
    vi.mocked(ipcMain.handle).mockClear()
    registerSemantierWorkspaceIpc()
    const handler = vi.mocked(ipcMain.handle).mock.calls.find(([channel]) => channel === 'hermes:workspace:' + name)![1]
    await expect(
      handler(
        { sender: { id: 99 } } as never,
        'https://cloud.example',
        scope,
        'canonical-a',
        'main-a',
        'request',
        'binding'
      )
    ).rejects.toThrow('changed')
    expect(fetch).toHaveBeenCalledTimes(1)
  }
)

it('creates only in the captured account and returns the real concurrent history snapshot', async () => {
  const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith('/auth/context')) {
      return Response.json(identity)
    }

    if (url.endsWith('/api/sessions') && init?.method === 'POST') {
      return Response.json({
        ok: true,
        sessionKey: 'session_desktop_create-001',
        entry: { key: 'session_desktop_create-001' }
      })
    }

    return Response.json({
      sessionId: 'session_desktop_create-001',
      messages: [{ messageId: '91', role: 'assistant', text: 'Server-owned' }]
    })
  })

  vi.mocked(session.fromPartition).mockReturnValue({ fetch: fetcher } as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler = vi.mocked(ipcMain.handle).mock.calls.find(([name]) => name === 'hermes:workspace:create-session')![1]
  const result = await handler({ sender: { id: 99 } } as never, 'https://cloud.example', scope, 'Main', 'create-001')
  expect(result).toMatchObject({ sessionId: 'session_desktop_create-001', messages: [{ messageId: '91' }] })
  expect(fetcher.mock.calls.filter(([, init]) => init?.method === 'POST')).toHaveLength(1)
  fetcher.mockClear()
  await expect(
    handler({} as never, 'https://cloud.example', { ...scope, userId: 'foreign' }, 'Main', 'other')
  ).rejects.toThrow('changed')
  expect(fetcher.mock.calls.some(([, init]) => init?.method === 'POST')).toBe(false)
})

it('cancels only the caller-owned in-flight Main stream without retrying or submitting another turn', async () => {
  let streamSignal: AbortSignal | undefined
  let streamController: ReadableStreamDefaultController<Uint8Array> | undefined

  const fetch = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith('/auth/context')) {
      return Response.json(identity)
    }

    if (url.endsWith('/api/send-stream')) {
      streamSignal = init?.signal as AbortSignal

      return new Response(
        new ReadableStream<Uint8Array>({
          start(controller) {
            streamController = controller
            streamSignal.addEventListener('abort', () => controller.error(new DOMException('Stopped', 'AbortError')), {
              once: true
            })
          }
        }),
        { headers: { 'Content-Type': 'text/event-stream' } }
      )
    }

    throw new Error('Unexpected request: ' + url)
  })

  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler = (name: string) => vi.mocked(ipcMain.handle).mock.calls.find(([channel]) => channel === name)?.[1]
  const stop = handler('hermes:workspace:interrupt')
  expect(stop).toBeTypeOf('function')
  const event = { sender: { id: 99 } } as never
  const pending = handler('hermes:workspace:send')!(
    event,
    'https://cloud.example',
    scope,
    'original',
    'hello',
    'retained-request'
  )
  const rejected = expect(pending).rejects.toThrow('semantier.workspace.user_stop.v1:retained-request')
  await vi.waitFor(() => expect(streamController).toBeDefined())
  await expect(
    stop!(event, 'https://cloud.example', { ...scope, userId: 'other' }, 'original', 'retained-request')
  ).rejects.toThrow('changed')
  expect(streamSignal?.aborted).toBe(false)
  await expect(
    stop!({ sender: { id: 100 } } as never, 'https://cloud.example', scope, 'original', 'retained-request')
  ).resolves.toMatchObject({ cancel_requested: false })
  expect(streamSignal?.aborted).toBe(false)
  await expect(stop!(event, 'https://cloud.example', scope, 'original', 'older-request')).resolves.toMatchObject({
    cancel_requested: false
  })
  expect(streamSignal?.aborted).toBe(false)
  await expect(stop!(event, 'https://cloud.example', scope, 'original', 'retained-request')).resolves.toMatchObject({
    cancel_requested: true,
    request_id: 'retained-request'
  })
  await rejected
  expect(streamSignal?.aborted).toBe(true)
  await expect(stop!(event, 'https://cloud.example', scope, 'original', 'retained-request')).resolves.toMatchObject({
    cancel_requested: false
  })
  expect(fetch.mock.calls.filter(([url]) => url.endsWith('/api/send-stream'))).toHaveLength(1)
})

it('withdraws a Main send stopped while its initial identity check is still waiting', async () => {
  let release: (() => void) | undefined
  let reads = 0

  const fetch = vi.fn(async (url: string) => {
    if (!url.endsWith('/auth/context')) {
      throw new Error('Stopped request must not be sent')
    }

    if (++reads === 1) {
      await new Promise<void>(resolve => {
        release = resolve
      })
    }

    return Response.json(identity)
  })

  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler = (name: string) => vi.mocked(ipcMain.handle).mock.calls.find(([channel]) => channel === name)![1]
  const event = { sender: { id: 101 } } as never
  const pending = handler('hermes:workspace:send')(
    event,
    'https://cloud.example',
    scope,
    'original',
    'hello',
    'withdrawn-request'
  )
  const rejected = expect(pending).rejects.toThrow('semantier.workspace.user_stop.v1:withdrawn-request')
  await vi.waitFor(() => expect(release).toBeTypeOf('function'))
  await expect(
    handler('hermes:workspace:interrupt')(event, 'https://cloud.example', scope, 'original', 'withdrawn-request')
  ).resolves.toMatchObject({ cancel_requested: true })
  release!()
  await rejected
  expect(fetch.mock.calls.every(([url]) => url.endsWith('/auth/context'))).toBe(true)
})

it('stops Native work by awaiting its original admission and cancelling that exact receipt once', async () => {
  let admit: ((value: Response) => void) | undefined
  let cancelled = false
  const admission = 'a'.repeat(32)

  const fetch = vi.fn(async (url: string) => {
    if (url.endsWith('/auth/context')) {
      return Response.json(identity)
    }

    if (url.endsWith('/api/winman/messages/canonical-a')) {
      return new Promise<Response>(resolve => {
        admit = resolve
      })
    }
    const receipt = {
      admission_id: admission,
      status: cancelled ? 'terminal' : 'queued',
      outcome: cancelled ? 'cancelled' : null
    }

    if (url.endsWith('/cancel')) {
      cancelled = true

      return Response.json({
        cancel_requested: true,
        receipt: { ...receipt, status: 'terminal', outcome: 'cancelled' }
      })
    }

    if (url.includes('/api/winman/history/')) {
      return Response.json({
        session_id: 'canonical-a',
        bot_id: 'bot-a',
        native_profile_id: 'profile-a',
        canonical_session: { id: 'canonical-a' },
        messages: [],
        admission_receipt: receipt
      })
    }
    throw new Error('Unexpected request ' + url)
  })

  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler = (name: string) => vi.mocked(ipcMain.handle).mock.calls.find(([channel]) => channel === name)![1]
  const event = { sender: { id: 102 } } as never
  const pending = handler('hermes:workspace:send-native')(
    event,
    'https://cloud.example',
    scope,
    'canonical-a',
    'hello',
    'native-request',
    'binding'
  )
  await vi.waitFor(() => expect(admit).toBeTypeOf('function'))
  const stopped = handler('hermes:workspace:interrupt')(
    event,
    'https://cloud.example',
    scope,
    'canonical-a',
    'native-request'
  )
  expect(cancelled).toBe(false)
  admit!(
    Response.json({
      session_id: 'canonical-a',
      bot_id: 'bot-a',
      receipt: { admission_id: admission, status: 'queued', outcome: null }
    })
  )
  await expect(stopped).resolves.toMatchObject({ cancel_requested: true, request_id: 'native-request' })
  await expect(pending).resolves.toMatchObject({
    admission_receipt: { admission_id: admission, status: 'terminal', outcome: 'cancelled' }
  })
  expect(fetch.mock.calls.filter(([url]) => url.endsWith('/api/winman/messages/canonical-a'))).toHaveLength(1)
  expect(fetch.mock.calls.filter(([url]) => url.endsWith('/cancel'))).toHaveLength(1)
  await expect(
    handler('hermes:workspace:interrupt')(event, 'https://cloud.example', scope, 'canonical-a', 'native-request')
  ).resolves.toMatchObject({ cancel_requested: false })
})

it('waits for File Sync to stop before logout can clear account cookies', async () => {
  const order: string[] = []
  const owned = {
    fetch: vi.fn(),
    clearStorageData: vi.fn(async () => {
      order.push('cookies')
    }),
    clearCache: vi.fn(async () => {
      order.push('cache')
    })
  }
  vi.mocked(session.fromPartition).mockReturnValue(owned as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc(
    () => {
      order.push('changed')
    },
    async () => {
      order.push('sync-stopped')
    }
  )
  const handler = vi.mocked(ipcMain.handle).mock.calls.find(([name]) => name === 'hermes:workspace:logout')![1]
  await handler({} as never, 'https://cloud.example')
  expect(order).toEqual(['sync-stopped', 'cookies', 'cache', 'changed'])
})
it('refuses logout before cookie mutation if the sync process cannot stop', async () => {
  const owned = { fetch: vi.fn(), clearStorageData: vi.fn(), clearCache: vi.fn() }
  vi.mocked(session.fromPartition).mockReturnValue(owned as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc(undefined, async () => {
    throw new Error('child still alive')
  })
  const handler = vi.mocked(ipcMain.handle).mock.calls.find(([name]) => name === 'hermes:workspace:logout')![1]
  await expect(handler({} as never, 'https://cloud.example')).rejects.toThrow('child still alive')
  expect(owned.clearStorageData).not.toHaveBeenCalled()
})

it.each(['managed-devices', 'managed-device-consent', 'managed-device-revoke', 'managed-device-enroll'])(
  'refuses a changed account before %s',
  async name => {
    const fetch = vi
      .fn()
      .mockResolvedValue(Response.json({ ...identity, user: { ...identity.user, user_id: 'foreign' } }))
    vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
    vi.mocked(ipcMain.handle).mockClear()
    registerSemantierWorkspaceIpc()
    const handler = vi.mocked(ipcMain.handle).mock.calls.find(([channel]) => channel === 'hermes:workspace:' + name)![1]
    await expect(handler({} as never, 'https://cloud.example', scope, 'device-a', ['skills:sync'])).rejects.toThrow(
      'changed'
    )
    expect(fetch).toHaveBeenCalledTimes(1)
  }
)

it('discards a device inventory received during an account switch', async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(Response.json(identity))
    .mockResolvedValueOnce(
      Response.json({
        included_windows_seats: 2,
        devices: [{ device_id: 'private-device' }],
        additional_seat_checkout: 'unavailable'
      })
    )
    .mockResolvedValueOnce(Response.json({ ...identity, user: { ...identity.user, user_id: 'foreign' } }))

  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler = vi
    .mocked(ipcMain.handle)
    .mock.calls.find(([channel]) => channel === 'hermes:workspace:managed-devices')![1]
  await expect(handler({} as never, 'https://cloud.example', scope)).rejects.toThrow('changed')
  expect(fetch).toHaveBeenCalledTimes(3)
})

it('withholds a Project continuation response after the Workspace account changes', async () => {
  const fetch = vi
    .fn()
    .mockResolvedValueOnce(Response.json(identity))
    .mockResolvedValueOnce(Response.json({ project: { project_ref: 'private' } }))
    .mockResolvedValueOnce(Response.json({ ...identity, organization_id: 'org-b' }))

  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler = vi
    .mocked(ipcMain.handle)
    .mock.calls.find(([name]) => name === 'hermes:workspace:continue-project')![1]

  await expect(handler({} as never, 'https://cloud.example', scope, 'project:' + 'a'.repeat(32))).rejects.toThrow(
    'changed'
  )
  expect(fetch).toHaveBeenCalledTimes(3)
})

it('shares the signed default sidebar inventory with the actual native Main IPC', async () => {
  const { readScopedWorkspaceDesktop } = await import('./semantier-workspace-ipc')
  let release!: (value: Response) => void
  const first = new Promise<Response>(resolve => {
    release = resolve
  })

  const fetch = vi
    .fn()
    .mockReturnValueOnce(first)
    .mockResolvedValueOnce(Response.json({ sessions: [{ key: 'main-a', title: 'Main' }] }))
    .mockResolvedValueOnce(Response.json(identity))

  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
  vi.mocked(ipcMain.handle).mockClear()
  registerSemantierWorkspaceIpc()
  const handler = vi.mocked(ipcMain.handle).mock.calls.find(([name]) => name === 'hermes:workspace:sessions')![1]

  const desktop = readScopedWorkspaceDesktop(
    'https://cloud.example',
    scope,
    '/api/profiles/sessions/sidebar?recents_profile=default&recents_limit=40',
    'default',
    scope
  )

  const registry = readScopedWorkspaceDesktop('https://cloud.example', scope, '/api/sessions', 'default', scope)
  const native = handler({} as never, 'https://cloud.example', scope)
  expect(fetch).toHaveBeenCalledTimes(1)
  release(Response.json(identity))
  const [a, b, c] = await Promise.all([native, desktop, registry])

  expect(c).toEqual(a)
  expect(a).toMatchObject({ sessions: [{ key: 'main-a', id: 'main-a', title: 'Main' }] })
  expect(b).toMatchObject({
    recents: { sessions: a.sessions },
    cron: { unavailable: true },
    messaging: { unavailable: true }
  })
  expect(fetch).toHaveBeenCalledTimes(3)
})

it('joins overlapping startup inventory reads and rejects a changed scope before delivery', async () => {
  const { readScopedWorkspaceDesktop } = await import('./semantier-workspace-ipc')
  let release!: (value: Response) => void
  const first = new Promise<Response>(resolve => {
    release = resolve
  })

  const fetch = vi
    .fn()
    .mockReturnValueOnce(first)
    .mockResolvedValueOnce(Response.json({ sessions: [] }))
    .mockResolvedValueOnce(Response.json({ ...identity, organization_id: 'changed' }))
    .mockResolvedValueOnce(Response.json(identity))
    .mockResolvedValueOnce(Response.json({ sessions: [] }))
    .mockResolvedValueOnce(Response.json(identity))

  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as unknown as ReturnType<typeof session.fromPartition>)
  const read = () =>
    readScopedWorkspaceDesktop('https://cloud.example', scope, '/api/profiles/sessions/sidebar', 'default')
  const requests = [read(), read()]
  expect(fetch).toHaveBeenCalledTimes(1)
  release(Response.json(identity))
  const results = await Promise.allSettled(requests)
  expect(results.every(result => result.status === 'rejected' && /changed/.test(String(result.reason)))).toBe(true)
  expect(fetch).toHaveBeenCalledTimes(3)
  await expect(read()).resolves.toMatchObject({ recents: { sessions: [] } })
  expect(fetch).toHaveBeenCalledTimes(6)
})

it('shares only the pending initial scope check across distinct startup reads', async () => {
  const { readScopedWorkspaceDesktop } = await import('./semantier-workspace-ipc')
  let release!: (value: Response) => void
  const initial = new Promise<Response>(resolve => {
    release = resolve
  })
  let identityReads = 0

  const fetch = vi.fn(async (input: string) => {
    if (new URL(input).pathname === '/auth/context') {
      identityReads += 1

      return identityReads === 1 ? initial : Response.json(identity)
    }

    return Response.json({ sessions: [] })
  })

  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)

  const origin = 'https://cloud.example'
  const sessions = () => readScopedWorkspaceDesktop(origin, scope, '/api/sessions', 'default')
  const profiles = () => readScopedWorkspaceDesktop(origin, scope, '/api/profiles/sessions', 'default')
  const first = sessions()
  const second = profiles()
  expect(identityReads).toBe(1)
  release(Response.json(identity))
  await Promise.all([first, second])
  expect(identityReads).toBe(3)

  await sessions()
  expect(identityReads).toBe(5)
})

it('checks a freshly resolved registry scope and revalidates after the protected read', async () => {
  const { readScopedWorkspaceDesktop } = await import('./semantier-workspace-ipc')

  const fetch = vi
    .fn()
    .mockResolvedValueOnce(Response.json(identity))
    .mockResolvedValueOnce(Response.json({ sessions: [] }))
    .mockResolvedValueOnce(Response.json({ ...identity, organization_id: 'changed' }))

  vi.mocked(session.fromPartition).mockReturnValue({ fetch } as never)
  const origin = 'https://cloud.example'
  const read = (resolved: typeof scope) =>
    readScopedWorkspaceDesktop(origin, scope, '/api/sessions', 'default', resolved)

  await expect(read({ ...scope, organizationId: 'other' })).rejects.toThrow('changed')
  expect(fetch).not.toHaveBeenCalled()
  await expect(read(scope)).rejects.toThrow('changed')
  expect(fetch.mock.calls.map(([url]) => new URL(url).pathname)).toEqual([
    '/auth/context',
    '/api/sessions',
    '/auth/context'
  ])
})
