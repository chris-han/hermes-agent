import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { RefreshCw } from '@/lib/icons'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { refreshConnectionsRegistry } from '@/store/connections'

import { ListRow, SectionHeading } from './primitives'

type Status = { state: 'Paused' | 'Syncing' | 'Synced' | 'Conflict' | 'Error'; folder?: string; message?: string; stopState?: 'PENDING' | 'UNKNOWN' }

export function FileSyncSettings() {
  const { locale } = useI18n()
  const zh = locale === 'zh'
  const account = useStore($connectionsRegistry)?.accountContext
  const [status, setStatus] = useState<Status>({ state: 'Paused' })
  const generation = useRef(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [actionError, setActionError] = useState('')
  const api = window.hermesDesktop?.fileSync
  const connected = Boolean(account)
  const stopMessage = status.stopState && (zh
    ? '无法确认先前发送的任务已停止，任务仍可能完成。文件会保留；点击“继续”可重试停止检查，确认前不会重新开始同步。'
    : 'File Sync could not confirm that previously sent work has stopped; it may still finish. Your files are preserved. Resume will retry the stop check; syncing will not restart until the stop is confirmed.')
  useEffect(() => window.hermesDesktop?.workspace?.onAccountChanged?.(() => {
    void refreshConnectionsRegistry()
  }), [])
  useEffect(() => {
    let live = true
    generation.current++
    setBusy(false)
    setStatus({ state: 'Paused' })
    setError('')
    setActionError('')

    let request = 0
    let polling = false

    const update = () => {
      // A timer tick must not supersede a slow response from the current owner.
      if (!api || !connected || polling) {return}
      const current = ++request
      const ownerGeneration = generation.current

      polling = true
      void api
        .status()
        .then(s => {
          if (live && current === request && ownerGeneration === generation.current) {
            setStatus(s)
            setError('')
          }
        })
        .catch(() => {
          if (live && current === request && ownerGeneration === generation.current) {setError(zh ? '请检查 Semantier 账户连接。' : 'Check your Semantier account connection to see File Sync status.')}
        })
        .finally(() => {polling = false})
    }

    update()
    const timer = setInterval(update, 2000)

    return () => {
      live = false
      clearInterval(timer)
    }
  }, [api, connected, account?.origin, account?.scope.userId, account?.scope.workspaceId, account?.scope.organizationId, zh])

  async function act(action: 'choose' | 'pause' | 'resume') {
    if (!api) {return}
    const current = ++generation.current
    setBusy(true)
    setError('')
    setActionError('')

    try {
      const result = await api[action]()

      if (current === generation.current) {setStatus(result)}
    } catch (error) {
      if (current === generation.current) {setActionError(error instanceof Error ? error.message : String(error))}
    } finally {
      if (current === generation.current) {
        generation.current++
        setBusy(false)
      }
    }
  }

  return (
    <div className="space-y-4">
      <SectionHeading icon={RefreshCw} title={zh ? '文件同步' : 'File Sync'} />
      <p className="text-sm text-muted-foreground">
        {zh
          ? '选择一个个人文件夹，在桌面端与工作区的“同步文件”之间保持更新。首次设置请选择空文件夹；已有绑定可以继续使用原文件夹。'
          : 'Choose one personal folder to keep up to date with Workspace → Synced Files. For a new setup, choose an empty folder; an existing binding can keep its folder.'}
      </p>
      {!connected || !api ? (
        <p>{zh ? '请先登录 Semantier 账户。' : 'Sign in to your Semantier account first.'}</p>
      ) : (
        <>
          <ListRow
            action={
              <Button disabled={busy} onClick={() => void act('choose')}>
                {zh ? '选择并启用' : 'Choose and enable'}
              </Button>
            }
            description={status.folder || (zh ? '尚未选择文件夹' : 'No folder selected')}
            title={zh ? '文件夹' : 'Folder'}
          />
          <ListRow
            action={
              <div className="flex gap-2">
                <Button disabled={busy || !status.folder} onClick={() => void act('pause')}>
                  {zh ? '暂停' : 'Pause'}
                </Button>
                <Button disabled={busy || !status.folder} onClick={() => void act('resume')}>
                  {zh ? '继续' : 'Resume'}
                </Button>
              </div>
            }
            description={status.state}
            title={zh ? '同步状态' : 'Sync status'}
          />
          <p className="text-xs text-muted-foreground">
            {zh
              ? '状态基于文件大小、修改时间及变更事件，并非完整内容校验。冲突版本会保留。'
              : 'Status uses size, modification time and change detection, rather than a full content audit. Conflict copies are preserved.'}
          </p>
          <p className="text-xs text-muted-foreground">
            {zh
              ? '暂停或切换账户后，已发送的上传仍可能完成。'
              : 'Uploads already sent may still finish after pausing or switching accounts.'}
          </p>
        </>
      )}
      {(actionError || error || stopMessage || status.message) && (
        <p className="text-sm text-destructive" role="alert">
          {stopMessage || actionError || error || status.message}
        </p>
      )}
    </div>
  )
}
