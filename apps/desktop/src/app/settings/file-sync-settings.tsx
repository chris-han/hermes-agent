import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { RefreshCw } from '@/lib/icons'
import { $connection } from '@/store/session'

import { ListRow, SectionHeading } from './primitives'

type Status = { state: 'Paused' | 'Syncing' | 'Synced' | 'Conflict' | 'Error'; folder?: string; message?: string }

export function FileSyncSettings() {
  const { locale } = useI18n()
  const zh = locale === 'zh'
  const connection = useStore($connection)
  const [status, setStatus] = useState<Status>({ state: 'Paused' })
  const generation = useRef(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const api = window.hermesDesktop?.fileSync
  const connected = connection?.transport === 'semantier-workspace-http'
  useEffect(() => {
    let live = true
    generation.current++
    setBusy(false)
    setStatus({ state: 'Paused' })
    setError('')

    const update = () => {
      if (api && connected)
        {void api
          .status()
          .then(s => {
            if (live) {setStatus(s)}
          })
          .catch(() => {
            if (live) {setError(zh ? '请重新连接工作区。' : 'Reconnect your Workspace to see File Sync status.')}
          })}
    }

    update()
    const timer = setInterval(update, 2000)

    return () => {
      live = false
      clearInterval(timer)
    }
  }, [api, connected, connection?.connectionId, connection?.workspaceScope?.userId, zh])

  async function act(action: 'choose' | 'pause' | 'resume') {
    if (!api) {return}
    const current = generation.current
    setBusy(true)
    setError('')

    try {
      const result = await api[action]()

      if (current === generation.current) {setStatus(result)}
    } catch (error) {
      if (current === generation.current) {setError(error instanceof Error ? error.message : String(error))}
    } finally {
      if (current === generation.current) {setBusy(false)}
    }
  }

  return (
    <div className="space-y-4">
      <SectionHeading icon={RefreshCw} title={zh ? '文件同步' : 'File Sync'} />
      <p className="text-sm text-muted-foreground">
        {zh
          ? '选择一个个人文件夹，在桌面端与工作区的“同步文件”之间保持更新。'
          : 'Choose one personal folder to keep up to date with Workspace → Synced Files.'}
      </p>
      {!connected || !api ? (
        <p>{zh ? '请先将此窗口连接到 Semantier 工作区。' : 'Connect this window to your Semantier Workspace first.'}</p>
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
        </>
      )}
      {(error || status.message) && (
        <p className="text-sm text-destructive" role="alert">
          {error || status.message}
        </p>
      )}
    </div>
  )
}
