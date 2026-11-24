import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { $activeConnectionId, $connectionsRegistry } from '@/store/connections'

import type { PersistentFollowup, WinmanCapability, WorkspaceScope } from '../../electron/semantier-workspace-client'

export function PersistentFollowupPanel() {
  const activeId = useStore($activeConnectionId)
  const registry = useStore($connectionsRegistry)
  const available = registry?.connections.filter(connection => connection.transport === 'semantier-workspace-http') ?? []
  const current = available.find(connection => connection.id === activeId) ?? (available.length === 1 ? available[0] : undefined)

  return <FollowupScope key={current?.id ?? 'local'} origin={current?.url ?? undefined} />
}

function FollowupScope({ origin }: { origin?: string }) {
  const { locale } = useI18n()
  const zh = locale.startsWith('zh')
  const title = zh ? '持续跟进' : 'Persistent follow-up'
  const [scope, setScope] = useState<WorkspaceScope>()
  const [status, setStatus] = useState<WinmanCapability>()
  const [items, setItems] = useState<PersistentFollowup[]>([])
  const [reading, setReading] = useState(Boolean(origin))
  const [saving, setSaving] = useState(false)
  const [failed, setFailed] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [reminder, setReminder] = useState<{ id: string; text: string }>()
  const api = window.hermesDesktop?.workspace
  const enabled = status?.desired.enabled ?? false
  const readyForWork = enabled && status?.execution === 'ready' && !reading && !saving

  useEffect(() => {
    if (!origin || !api?.winmanCapability) { setReading(false);

 return }

    let current = true

    const load = async () => {
      try {
        const identity = await api.identity(origin)

        if (!current) { return }

        if (!identity.authenticated || !identity.user?.user_id || !identity.workspace?.id || !identity.organization_id) {
          setScope(undefined)
          setStatus(undefined)
          setItems([])
          setReminder(undefined)

          return
        }

        const expected = { userId: identity.user.user_id, workspaceId: identity.workspace.id, organizationId: identity.organization_id }
        const state = await api.winmanCapability(origin, expected)

        if (!current) { return }
        setScope(expected)
        setStatus(state)
        setFailed(false)
      } catch {
        if (current) { setScope(undefined); setStatus(undefined); setItems([]); setReminder(undefined); setFailed(true) }
      } finally {
        if (current) { setReading(false) }
      }
    }

    void load()

    return () => { current = false }
  }, [api, origin, refresh])

  useEffect(() => {
    setItems([])
    setReminder(undefined)

    if (!readyForWork || !scope || !origin || !api) {return}
    let current = true
    void api.persistentFollowups(origin, scope).then(
      work => { if (current) {setItems(work.items)} },
      () => { if (current) {setFailed(true)} }
    )

    return () => { current = false }
  }, [api, origin, scope, status, readyForWork])

  const save = async (enabled: boolean) => {
    if (!scope || !origin || !api) { return }
    setSaving(true)
    setFailed(false)

    try {
      setStatus(await api.winmanCapability(origin, scope, enabled))
    } catch {
      setStatus(undefined)
      setFailed(true)
    } finally {
      setSaving(false)
    }
  }

  const readReminder = async (id: string) => {
    if (!readyForWork || !scope || !origin || !api) { return }

    try {
      const text = await api.persistentReminder(origin, scope, id)

      if (!text) { setFailed(true);

 return }

      setReminder({ id, text })
    } catch {
      setFailed(true)
    }
  }

  const message = failed ? (zh ? '读取或保存失败。请重试。' : 'Could not load or save. Try again.')
    : reading ? (zh ? '正在读取状态…' : 'Checking status…')
    : saving ? (zh ? '正在保存…' : 'Saving…')
    : !status ? (zh ? '登录工作区后可启用。' : 'Sign in to a Workspace to enable.')
    : !enabled ? (zh ? '已关闭' : 'Off')
    : status.execution === 'ready' ? (zh ? '已开启 · 可执行' : 'On · Ready')
    : (zh ? '已开启 · 暂不可执行' : 'On · Execution unavailable')

  return (
    <section aria-label={title} className="mx-2 my-2 grid shrink-0 gap-2 rounded-md border p-2 text-sm">
      <div className="flex flex-wrap items-center justify-between gap-1">
        <h2 className="font-medium">{title}</h2>
        {status && <Button aria-checked={enabled} aria-label={title} disabled={saving || reading} onClick={() => { void save(!enabled) }} role="switch"
          size="sm" variant="outline">
          {enabled ? (zh ? '开启' : 'On') : (zh ? '关闭' : 'Off')}
        </Button>}
      </div>
      <p className="text-xs text-muted-foreground" role="status">{message}</p>
      {enabled && status?.execution === 'ready' && <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer">{zh ? '如何安排跟进' : 'How to schedule a follow-up'}</summary>
        <p className="mt-1">{zh
          ? '在 Main 的来源会话中，用下列命令为已确认的来源安排提醒，时间须带时区。'
          : 'In the original Main conversation, schedule the confirmed source using a time with its time zone.'}</p>
        <code className="mt-1 block break-words">/followup create selected &lt;ISO 8601&gt;</code>
      </details>}
      {readyForWork && items.length > 0 && <ul className="grid max-h-40 gap-2 overflow-auto">
        {items.map(item => <li className="grid gap-1 border-t pt-2" data-work-id={item.id} key={item.id}>
          <div className="break-words font-medium">{item.title}</div>
          <time className="text-xs text-muted-foreground" dateTime={item.scheduled_at}>
            {new Date(item.scheduled_at).toLocaleString(locale)}
          </time>
          {item.receipt && <Button onClick={() => { void readReminder(item.id) }} size="sm" variant="ghost">
            {zh ? '查看提醒' : 'Read reminder'}
          </Button>}
          {reminder?.id === item.id && <p className="break-words">{reminder.text}</p>}
        </li>)}
      </ul>}
      {origin && <Button disabled={reading || saving} onClick={() => {
        setReading(true)
        setRefresh(value => value + 1)
      }} size="sm" variant="ghost">{failed ? (zh ? '重试' : 'Retry') : (zh ? '刷新' : 'Refresh')}</Button>}
    </section>
  )
}
