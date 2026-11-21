import { randomUUID } from 'node:crypto'

import { assertWorkspaceScope, normalizeSemantierWorkspaceOrigin, type WorkspaceIdentity, type WorkspaceScope } from './semantier-workspace-client'

type Account = { origin: string; scope: WorkspaceScope }
type Client = { identity: () => Promise<WorkspaceIdentity>; approveDevice: (intent: string) => Promise<{ status: string }>;
  deviceStatus: (intent: string) => Promise<{ status: string }> }
type Options = {
  current: () => Account | null
  client: (origin: string) => Client
  fence: (account: Record<string, string> | null) => Promise<void>
  native: (path: string, method?: string, body?: unknown) => Promise<any>
}

/** Coordinates existing account and native device owners; never handles credentials. */
export class AccountModelCoordinator {
  private epoch = 0
  private fenceQueue: Promise<void> = Promise.resolve()
  private polls = new Map<string, string>()
  private running: { key: string; task: Promise<'ready' | 'unavailable'> } | null = null
  constructor(private readonly options: Options) {}

  private enqueue(action: () => Promise<void>) {
    const task = this.fenceQueue.catch(() => {}).then(action)
    this.fenceQueue = task
    return task
  }

  async suspend() {
    this.epoch++
    await this.enqueue(async () => {
      await this.options.fence(null)
    })
  }

  async prepare(account: Account): Promise<'ready' | 'unavailable'> {
    const key = JSON.stringify([this.epoch, account])
    if (this.running) {
      if (this.running.key === key) {return this.running.task}
      throw new Error('Previous account preparation is still settling.')
    }
    const task = this.perform(account)
    this.running = { key, task }
    try {return await task} finally {if (this.running?.task === task) {this.running = null}}
  }

  private async perform(account: Account): Promise<'ready' | 'unavailable'> {
    const origin = normalizeSemantierWorkspaceOrigin(account.origin)
    const expected = { user_id: account.scope.userId, workspace_id: account.scope.workspaceId, organization_id: account.scope.organizationId }
    const epoch = this.epoch
    const client = this.options.client(origin)
    const current = async () => {
      const selected = this.options.current()
      if (epoch !== this.epoch || !selected || selected.origin !== origin ||
          selected.scope.userId !== account.scope.userId || selected.scope.workspaceId !== account.scope.workspaceId ||
          selected.scope.organizationId !== account.scope.organizationId) {throw new Error('The account changed. Retry preparation.')}
      assertWorkspaceScope(await client.identity(), account.scope)
      if (epoch !== this.epoch) {throw new Error('The account changed. Retry preparation.')}
    }

    await current()
    await this.enqueue(async () => {
      if (epoch !== this.epoch) {throw new Error('The account changed. Retry preparation.')}
      await this.options.fence({ origin, ...expected })
    })
    await current()
    const providers = await this.options.native('/api/providers/oauth')
    await current()
    const status = providers?.providers?.find((item: any) => item.id === 'semantier-cloud')?.status
    if (status?.logged_in) {return this.ready(status.readiness)}

    const started = await this.options.native('/api/providers/oauth/semantier-cloud/start?account_prepare=true', 'POST', { expected_actor: expected, origin })
    await current()
    const target = new URL(started.verification_url)
    if (target.origin !== origin || target.username || target.password || target.pathname !== '/billing/connect' || target.hash ||
        [...target.searchParams.keys()].length !== 1 || !/^dci_[A-Za-z0-9_-]{1,128}$/.test(target.searchParams.get('intent_ref') || '') ||
        typeof started.session_id !== 'string' || !/^cloud-[A-Za-z0-9_-]{1,128}$/.test(started.session_id)) {
      throw new Error('Cloud device origin or intent is invalid.')
    }
    const intent = target.searchParams.get('intent_ref')!
    const existing = await client.deviceStatus(intent)
    await current()
    if (existing.status === 'pending') {
      const attempt = await this.options.native(`/api/providers/oauth/semantier-cloud/approval-attempt/${encodeURIComponent(started.session_id)}`, 'POST', {})
      await current()
      if (attempt?.first !== true) {return 'unavailable'}
      const approval = await client.approveDevice(intent)
      if (approval.status !== 'approved') {throw new Error('Cloud device authorization was not approved.')}
    } else if (!['approved', 'claimed', 'installed'].includes(existing.status)) {
      return 'unavailable'
    }
    await current()
    const operation = this.polls.get(started.session_id) || randomUUID()
    this.polls.set(started.session_id, operation)
    const reply = await this.options.native(`/api/providers/oauth/semantier-cloud/poll/${encodeURIComponent(started.session_id)}?operation_id=${operation}`)
    await current()
    if (reply.status !== 'approved') {return 'unavailable'}
    this.polls.delete(started.session_id)
    return this.ready(reply.readiness)
  }

  private ready(readiness: any): 'ready' | 'unavailable' {
    return readiness?.account_provisioning?.inference_ready === true && readiness?.catalog?.status === 'available' &&
      Array.isArray(readiness.catalog.data) && readiness.catalog.data.length > 0 ? 'ready' : 'unavailable'
  }
}
