import { describe, expect, it, vi } from 'vitest'

import { AccountModelCoordinator } from './account-model-coordinator'

const scope = { userId: 'user-a', workspaceId: 'workspace-a', organizationId: 'org-a' }
const account = { origin: 'https://cloud.example', scope }
const identity = { authenticated: true, user: { user_id: scope.userId, workspace_id: scope.workspaceId },
  workspace: { id: scope.workspaceId }, organization_id: scope.organizationId }

function fixture() {
  let selected: typeof account | null = account
  const identityRead = vi.fn(async () => identity)
  const approve = vi.fn(async () => ({ status: 'approved' }))
  const deviceStatus = vi.fn(async () => ({ status: 'pending' }))
  const fence = vi.fn(async (_account: Record<string, string> | null) => {})
  const native = vi.fn<(path: string, method?: string, body?: unknown) => Promise<any>>(async path => {
    if (path === '/api/providers/oauth') {return { providers: [{ id: 'semantier-cloud', status: { logged_in: false } }] }}
    if (path.includes('/start?')) {return { session_id: 'cloud-session', verification_url: 'https://cloud.example/billing/connect?intent_ref=dci_test' }}
    if (path.includes('/approval-attempt/')) {return { first: true }}
    if (path.includes('/poll/')) {return { status: 'approved', readiness: { account_provisioning: { inference_ready: true }, catalog: { status: 'available', data: [{ id: 'eligible' }] } } }}
    return { ok: true }
  })
  const owner = new AccountModelCoordinator({ native, fence, client: () => ({ identity: identityRead, approveDevice: approve, deviceStatus }), current: () => selected })
  return { owner, native, fence, approve, deviceStatus, identityRead, setAccount: (value: typeof account | null) => {selected = value} }
}

describe('account model preparation', () => {
  it('uses the selected account and the native device flow without changing the agent runtime', async () => {
    const f = fixture()
    expect(await f.owner.prepare(account)).toBe('ready')
    expect(f.approve).toHaveBeenCalledWith('dci_test')
    expect(f.native).toHaveBeenCalledWith('/api/providers/oauth/semantier-cloud/start?account_prepare=true', 'POST', {
      expected_actor: { user_id: 'user-a', workspace_id: 'workspace-a', organization_id: 'org-a' }, origin: 'https://cloud.example'
    })
    expect(f.native.mock.calls.some(([path]) => path.includes('gateway') || path.includes('/model/set'))).toBe(false)
  })

  it('does not approve an intent from another origin', async () => {
    const f = fixture()
    f.native.mockImplementation(async (path: string) => path.includes('/start?')
      ? { session_id: 'cloud-session', verification_url: 'https://foreign.example/billing/connect?intent_ref=dci_test' }
      : path === '/api/providers/oauth' ? { providers: [] } : { ok: true })
    await expect(f.owner.prepare(account)).rejects.toThrow('origin')
    expect(f.approve).not.toHaveBeenCalled()
  })

  it('does not replay an approval whose previous outcome is uncertain', async () => {
    const f = fixture()
    const native = f.native.getMockImplementation()!
    f.native.mockImplementation(async (path: string, method?: string, body?: unknown) =>
      path.includes('/approval-attempt/') ? { first: false } : native(path, method, body))
    expect(await f.owner.prepare(account)).toBe('unavailable')
    expect(f.approve).not.toHaveBeenCalled()
  })

  it('uses a fresh poll operation after a definite pending response', async () => {
    const f = fixture()
    const native = f.native.getMockImplementation()!
    let polls = 0
    f.native.mockImplementation(async (path: string, method?: string, body?: unknown) => {
      if (path.includes('/poll/') && ++polls === 1) {return { status: 'pending' }}
      return native(path, method, body)
    })
    expect(await f.owner.prepare(account)).toBe('unavailable')
    f.deviceStatus.mockResolvedValue({ status: 'approved' })
    expect(await f.owner.prepare(account)).toBe('ready')
    const calls = f.native.mock.calls.filter(([path]) => path.includes('/poll/')).map(([path]) => path)
    expect(calls).toHaveLength(2)
    expect(calls[0]).not.toBe(calls[1])
  })

  it('reuses the same poll operation after a lost response', async () => {
    const f = fixture()
    const native = f.native.getMockImplementation()!
    let polls = 0
    f.native.mockImplementation(async (path: string, method?: string, body?: unknown) => {
      if (path.includes('/poll/') && ++polls === 1) {throw new Error('lost response')}
      return native(path, method, body)
    })
    await expect(f.owner.prepare(account)).rejects.toThrow('lost response')
    f.deviceStatus.mockResolvedValue({ status: 'approved' })
    expect(await f.owner.prepare(account)).toBe('ready')
    const calls = f.native.mock.calls.filter(([path]) => path.includes('/poll/')).map(([path]) => path)
    expect(calls).toHaveLength(2)
    expect(calls[0]).toBe(calls[1])
  })

  it('coalesces concurrent preparation for the same account', async () => {
    const f = fixture()
    expect(await Promise.all([f.owner.prepare(account), f.owner.prepare(account)])).toEqual(['ready', 'ready'])
    expect(f.approve).toHaveBeenCalledTimes(1)
  })

  it('suspends the native fence before a changed account can activate', async () => {
    const f = fixture()
    let release!: () => void
    const blocked = new Promise<void>(resolve => {release = resolve})
    f.identityRead.mockImplementationOnce(async () => {await blocked; return identity})
    const pending = f.owner.prepare(account)
    await vi.waitFor(() => expect(f.identityRead).toHaveBeenCalled())
    f.setAccount(null)
    await f.owner.suspend()
    release()
    await expect(pending).rejects.toThrow('changed')
    expect(f.approve).not.toHaveBeenCalled()
    expect(f.fence.mock.calls.at(-1)).toEqual([null])
  })

  it('orders suspension after an account fence write already in progress', async () => {
    const f = fixture()
    let release!: () => void
    const blocked = new Promise<void>(resolve => {release = resolve})
    f.fence.mockImplementationOnce(async () => {await blocked})
    const pending = f.owner.prepare(account)
    await vi.waitFor(() => expect(f.fence).toHaveBeenCalledTimes(1))
    f.setAccount(null)
    const suspended = f.owner.suspend()
    release()
    await suspended
    await expect(pending).rejects.toThrow('changed')
    expect(f.fence.mock.calls.at(-1)).toEqual([null])
    expect(f.approve).not.toHaveBeenCalled()
  })
})
