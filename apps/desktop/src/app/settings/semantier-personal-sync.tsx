import { useEffect, useId, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { personalSyncCopy } from '@/i18n/semantier-personal-sync'

import type { PersonalPin, PersonalSnapshot, PersonalSource } from '../../../electron/personal-capability-sync'
import type { WorkspaceScope } from '../../../electron/semantier-workspace-client'

type Selection = { receipt: PersonalPin; candidate: PersonalPin; source: Record<string, unknown> }

function skillText(source: Record<string, unknown>) {
  try {
    const skill = source.skill as { files: Array<{ relative_path: string; content_base64: string }> }
    const file = skill.files.find(f => f.relative_path === 'SKILL.md')!

    return new TextDecoder().decode(Uint8Array.from(atob(file.content_base64), c => c.charCodeAt(0)))
  } catch {
    return ''
  }
}

export function SemantierPersonalSync({ origin, scope }: { origin: string; scope: WorkspaceScope }) {
  const { locale } = useI18n()
  const copy = personalSyncCopy(locale)
  const id = useId()
  const epoch = useRef(0)
  const [name, setName] = useState('')
  const [preview, setPreview] = useState<PersonalSnapshot | null>(null)
  const [contextIndex, setContextIndex] = useState<number | null>(null)
  const [sources, setSources] = useState<PersonalSource[] | null>(null)
  const [more, setMore] = useState(false)
  const [selection, setSelection] = useState<Selection | null>(null)
  const [replace, setReplace] = useState(false)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [failed, setFailed] = useState(false)
  const scopeKey = JSON.stringify([origin, scope.userId, scope.organizationId, scope.workspaceId])
  // eslint-disable-next-line no-restricted-syntax -- Request epoch invalidation; no reactive or atom value is mirrored.
  useEffect(() => {
    const current = ++epoch.current
    setPreview(null)
    setContextIndex(null)
    setSources(null)
    setSelection(null)
    setReplace(false)
    setMessage('')
    setFailed(false)
    setBusy(false)

    return () => {
      epoch.current = current + 1
    }
  }, [scopeKey])

  async function perform(action: () => Promise<void>) {
    const current = epoch.current
    setBusy(true)
    setFailed(false)
    setMessage('')

    try {
      await action()
    } catch {
      if (current === epoch.current) {
        setFailed(true)
      }
    } finally {
      if (current === epoch.current) {
        setBusy(false)
      }
    }
  }

  const api = () => {
    const value = window.hermesDesktop.workspace

    if (!value) {
      throw new Error('Personal sync unavailable')
    }

    return value
  }

  const list = async () => {
    const current = epoch.current
    const value = await api().personalSources(origin, scope)

    if (current === epoch.current) {
      setSources(value.sources)
      setMore(value.has_more)
    }
  }

  return (
    <section aria-label={copy.title} className="grid min-w-0 gap-3 border-t pt-4">
      <h4 className="text-sm font-medium">{copy.title}</h4>
      <p className="text-xs text-muted-foreground">{copy.intro}</p>
      <label className="text-xs" htmlFor={id}>
        {copy.name}
      </label>
      <Input
        disabled={busy}
        id={id}
        onChange={e => {
          setName(e.target.value)
          setPreview(null)
          setContextIndex(null)
        }}
        value={name}
      />
      <div className="flex flex-wrap gap-2">
        <Button
          className="h-auto min-h-8 whitespace-normal"
          disabled={busy || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(name)}
          onClick={() =>
            void perform(async () => {
              const current = epoch.current
              const value = await api().personalPreview(origin, scope, name)

              if (current === epoch.current) {
                setPreview(value)
                setContextIndex(null)
              }
            })
          }
          type="button"
          variant="outline"
        >
          {copy.preview}
        </Button>
        <Button disabled={busy} onClick={() => void perform(list)} type="button" variant="outline">
          {copy.sources}
        </Button>
      </div>
      {preview ? (
        <div className="grid min-w-0 gap-2 rounded-md border p-3">
          <p className="text-xs font-medium">
            {preview.skill_id} · {copy.files}
          </p>
          {preview.files.map(file => (
            <p className="break-all text-xs text-muted-foreground" key={String(file.relative_path)}>
              {String(file.relative_path)} · {String(file.content_hash)}
            </p>
          ))}
          <fieldset className="grid min-w-0 gap-2">
            <legend className="text-xs">{copy.context}</legend>
            {preview.contexts.map((entry, index) => (
              <label className="grid min-w-0 gap-1 text-xs" key={entry.source_file + entry.content_hash}>
                <span>
                  <input
                    checked={contextIndex === index}
                    disabled={busy}
                    name={id + '-context'}
                    onChange={() => setContextIndex(index)}
                    type="radio"
                  />{' '}
                  {entry.source_file}
                </span>
                <span className="whitespace-pre-wrap break-words">{entry.content}</span>
              </label>
            ))}
            {!preview.contexts.length ? <p className="text-xs">{copy.noContext}</p> : null}
          </fieldset>
          <Button
            className="h-auto min-h-8 whitespace-normal"
            disabled={busy || contextIndex === null}
            onClick={() =>
              void perform(async () => {
                const current = epoch.current
                await api().personalShare(origin, scope, preview, preview.contexts[contextIndex!])

                if (current === epoch.current) {
                  setMessage(copy.shared)
                  await list()
                }
              })
            }
            type="button"
            variant="outline"
          >
            {copy.share}
          </Button>
        </div>
      ) : null}
      {sources?.length === 0 ? <p className="text-xs">{copy.empty}</p> : null}
      {sources?.map(row => (
        <div className="grid min-w-0 gap-2 rounded-md border p-3" key={row.receipt.artifact_ref}>
          <p className="text-xs font-medium">
            {row.skill_id} · {row.source_revision}
          </p>
          <p className="break-all text-xs text-muted-foreground">{row.source_transfer.artifact_hash}</p>
          <Button
            className="h-auto min-h-8 whitespace-normal"
            disabled={busy}
            onClick={() =>
              void perform(async () => {
                const current = epoch.current
                const value = await api().personalPrepare(origin, scope, row.receipt)

                if (current === epoch.current) {
                  setSelection(value)
                  setReplace(false)
                }
              })
            }
            type="button"
            variant="outline"
          >
            {copy.prepare}
          </Button>
        </div>
      ))}
      {more ? <p className="text-xs text-muted-foreground">{copy.more}</p> : null}
      {selection ? (
        <div className="grid min-w-0 gap-3 rounded-md border p-3">
          <p className="text-xs">{copy.review}</p>
          <p className="break-all text-xs">{String(selection.source.artifact_hash)}</p>
          <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words text-xs">
            {skillText(selection.source)}
          </pre>
          <p className="whitespace-pre-wrap break-words text-xs">
            {String((selection.source.decision_context as { content?: string })?.content || '')}
          </p>
          <Button
            className="h-auto min-h-8 whitespace-normal"
            disabled={busy}
            onClick={() =>
              void perform(async () => {
                const current = epoch.current
                await api().personalAdmit(origin, scope, selection.receipt, selection.candidate)

                if (current === epoch.current) {
                  setMessage(copy.enabled)
                }
              })
            }
            type="button"
            variant="outline"
          >
            {copy.enable}
          </Button>
          <label className="text-xs">
            <input checked={replace} disabled={busy} onChange={e => setReplace(e.target.checked)} type="checkbox" />{' '}
            {copy.replace}
          </label>
          <Button
            className="h-auto min-h-8 whitespace-normal"
            disabled={busy}
            onClick={() =>
              void perform(async () => {
                const current = epoch.current

                const value = (await api().personalInstall(
                  origin,
                  scope,
                  selection.receipt,
                  selection.candidate,
                  replace
                )) as { status?: string }

                if (!['installed', 'unchanged'].includes(value.status || '')) {
                  throw new Error('Personal package not installed')
                }

                if (current === epoch.current) {
                  setMessage(copy.installed)
                }
              })
            }
            type="button"
            variant="outline"
          >
            {copy.install}
          </Button>
        </div>
      ) : null}
      {busy ? (
        <p className="text-xs" role="status">
          {copy.pending}
        </p>
      ) : null}
      {failed ? (
        <p className="text-xs text-destructive" role="alert">
          {copy.failed}
        </p>
      ) : null}
      {message ? (
        <p className="text-xs" role="status">
          {message}
        </p>
      ) : null}
      <p className="text-xs text-muted-foreground">{copy.assurance}</p>
    </section>
  )
}
