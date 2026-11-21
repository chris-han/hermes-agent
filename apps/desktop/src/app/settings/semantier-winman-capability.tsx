import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { semantierWorkspaceCopy } from '@/i18n/semantier-workspace'

import type { WinmanCapability, WorkspaceScope } from '../../../electron/semantier-workspace-client'

import { ListRow } from './primitives'

/** Acknowledged intent and actual executor availability are separate. */
export function SemantierWinmanCapability({ origin, scope }: { origin: string; scope: WorkspaceScope }) {
  const { locale } = useI18n()
  const copy = semantierWorkspaceCopy(locale)
  const [value, setValue] = useState<WinmanCapability | null>(null)
  const [busy, setBusy] = useState(false)
  const [confirm, setConfirm] = useState(false)
  const [failed, setFailed] = useState(false)
  const generation = useRef(0)
  const pendingWrite = useRef(false)
  const scopeKey = JSON.stringify([origin, scope.userId, scope.workspaceId, scope.organizationId])
  useEffect(() => {
    let live = true
    setValue(null)
    setFailed(false)
    setConfirm(false)
    const refresh = () => {
      if (pendingWrite.current) {return}
      const current = ++generation.current
      void Promise.resolve().then(() => {
        const api = window.hermesDesktop.workspace
        if (!api) {throw new Error('Winman capability is unavailable.')}
        return api.winmanCapability(origin, scope)
      }).then(result => {
        if (live && current === generation.current) {setValue(result); setFailed(false)}
      }).catch(() => {if (live && current === generation.current) {setFailed(true)}})
    }
    refresh()
    window.addEventListener('focus', refresh)
    return () => {live = false; generation.current++; window.removeEventListener('focus', refresh)}
    // Scope identity, rather than the renderer's object identity, owns the request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeKey])

  async function change(enabled: boolean) {
    pendingWrite.current = true
    generation.current++
    setBusy(true)
    setFailed(false)
    try {
      const api = window.hermesDesktop.workspace
      if (!api) {throw new Error('Winman capability is unavailable.')}
      const result = await api.setWinmanCapability(origin, scope, enabled)
      setValue(result)
      setConfirm(false)
    } catch {setFailed(true)} finally {pendingWrite.current = false; setBusy(false)}
  }

  return <div>
    <ListRow title={copy.winmanTitle} description={failed ? copy.winmanError : !value ? copy.winmanLoading :
      value.desired.enabled ? copy.winmanRequested : copy.winmanOff}
      action={value ? <Button disabled={busy} type="button" variant="outline" onClick={() => {
        if (value.desired.enabled) {void change(false)} else {setConfirm(true)}
      }}>{value.desired.enabled ? copy.winmanDisable : copy.winmanEnable}</Button> : undefined} />
    {confirm && !value?.desired.enabled ? <div role="group" aria-label={copy.winmanTitle}>
      <p>{copy.winmanConsequences}</p>
      <Button disabled={busy} type="button" onClick={() => void change(true)}>{copy.winmanConfirm}</Button>
      <Button disabled={busy} type="button" variant="outline" onClick={() => setConfirm(false)}>{copy.winmanCancel}</Button>
    </div> : null}
  </div>
}
