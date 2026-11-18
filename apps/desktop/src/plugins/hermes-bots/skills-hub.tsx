/**
 * Native skill discovery and installation for the owning Bot profile.
 *
 * A leaf: the advanced profile editor and the create dialog both mount it, and
 * it reaches back into neither.
 */

import { Button, Codicon, host, Input, useI18n } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'

import { useBots } from './i18n'
import { requestForBot } from './routing'
import type { RosterRow } from './types'

/** One `skills.manage action=search` hit. */
interface HubSkillResult {
  description?: string
  name: string
}
interface HubSkillsSectionProps {
  /** Existing bot to route through; omitted for the launch profile at create time. */
  bot?: RosterRow
  onInstalled?: (name: string) => void
}

export function HubSkillsSection({ bot, onInstalled }: HubSkillsSectionProps) {
  const b = useBots()
  const { t } = useI18n()
  const h = t.skills.hub
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<HubSkillResult[] | null>(null)
  const [searching, setSearching] = useState(false)
  const [installing, setInstalling] = useState<null | string>(null)
  const [installed, setInstalled] = useState<Record<string, boolean>>({})
  const [error, setError] = useState(false)
  const generation = useRef(0)
  const searchGeneration = useRef(0)
  const installLock = useRef(false)
  // eslint-disable-next-line no-restricted-syntax -- Cancellation epoch for stale async work, not a mirror of reactive state.
  useEffect(() => {
    generation.current += 1
    setResults(null)
    setInstalled({})
    setInstalling(null)
    setSearching(false)
    installLock.current = false

    return () => {
      generation.current += 1
    }
  }, [bot?.name, bot?.connectionId, bot?.targetProfile])

  const search = async () => {
    const q = query.trim()

    if (!q || searching) {
      return
    }

    const epoch = generation.current
    const searchEpoch = ++searchGeneration.current
    setError(false)
    setSearching(true)
    setResults(null)

    try {
      const params = { action: 'search', query: q, ...(bot ? { profile: bot.name } : {}) }

      const res: { results?: HubSkillResult[] } = await (bot
        ? requestForBot(bot, 'skills.manage', params)
        : host.request('skills.manage', params))

      if (epoch !== generation.current || searchEpoch !== searchGeneration.current) {
        return
      }

      setResults(res.results || [])
    } catch {
      if (epoch === generation.current) {
        setError(true)
        setResults(null)
      }
    } finally {
      if (epoch === generation.current) {
        setSearching(false)
      }
    }
  }

  const install = async (name: string, displayName?: string) => {
    const label = displayName || name

    if (installLock.current || installed[label]) {
      return
    }

    installLock.current = true
    const epoch = generation.current
    setInstalling(label)

    try {
      // Existing bots must use their owner route; the active gateway is not
      // necessarily the gateway that owns the bot. Create-time installs stay
      // ambient because there is no bot row to route yet.
      const params = {
        action: 'install',
        query: name,
        ...(bot ? { profile: bot.name } : {})
      }

      const result = await (bot ? requestForBot(bot, 'skills.manage', params) : host.request('skills.manage', params))

      if (epoch !== generation.current) {
        return
      }

      if (!result || (result as { installed?: boolean }).installed !== true) {
        throw new Error('Install did not confirm success')
      }

      setInstalled(prev => ({
        ...prev,
        [label]: true
      }))
      host.notify({
        kind: 'success',
        message: b.tools.installed(label)
      })

      if (typeof onInstalled === 'function') {
        onInstalled(label)
      }
    } catch (err) {
      if (epoch === generation.current) {
        host.notifyError(err, b.tools.installFailed(label))
      }
    } finally {
      if (epoch === generation.current) {
        installLock.current = false
        setInstalling(null)
      }
    }
  }

  return (
    <div className="grid gap-1.5 border-t border-(--ui-stroke-secondary) pt-2">
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-[0.7rem] font-medium text-(--ui-text-secondary)">{b.tools.skillsHub}</div>
      </div>
      {error && (
        <div className="text-xs" role="alert">
          {h.actionFailed}
          <Button onClick={() => void search()} size="sm">
            {h.search}
          </Button>
        </div>
      )}
      <div className="flex gap-1.5">
        <Input
          className="h-7 flex-1 text-xs"
          onChange={event => setQuery(event.target.value)}
          onKeyDown={event => {
            // IME guard: Enter confirming a composed word must not search.
            if (event.nativeEvent?.isComposing || event.keyCode === 229) {
              return
            }

            if (event.key === 'Enter') {
              event.preventDefault()
              void search()
            }
          }}
          placeholder={b.tools.searchHub}
          value={query}
        />
        <Button disabled={searching || !query.trim()} onClick={() => void search()} size="sm" variant="secondary">
          {searching ? h.searching : h.search}
        </Button>
      </div>
      {searching ? <div className="px-1 text-[0.65rem] text-(--ui-text-quaternary)">{b.tools.searchHint}</div> : null}
      {results === null ? null : results.length === 0 ? (
        <div className="px-1 py-1.5 text-[0.7rem] text-(--ui-text-quaternary)">{h.noResults}</div>
      ) : (
        <div
          className="overflow-y-auto overscroll-contain"
          style={{
            maxHeight: 150
          }}
        >
          <div className="grid gap-1">
            {results.map(r => (
              <div className="flex items-center gap-2 text-xs" key={r.name}>
                <div className="min-w-0 flex-1">
                  <div className="truncate font-medium">{r.name}</div>
                  {r.description ? (
                    <div className="truncate text-[0.65rem] text-(--ui-text-quaternary)">{r.description}</div>
                  ) : null}
                </div>
                {installed[r.name] ? (
                  <span className="flex shrink-0 items-center gap-0.5 text-[0.65rem] text-(--ui-text-tertiary)">
                    <Codicon name="check" size="0.65rem" />
                    {h.installed}
                  </span>
                ) : (
                  <Button
                    aria-label={b.tools.installHint(r.name)}
                    className="shrink-0 px-2 font-semibold"
                    disabled={installing !== null}
                    onClick={() => void install(r.name)}
                    size="sm"
                    variant="ghost"
                  >
                    {installing === r.name ? '…' : '+'}
                  </Button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
