/** Semantier-only consent/journal. Never reads or changes Nous telemetry. */
import { randomUUID } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
export interface ObservationTarget {
  release_id: string
  target_sha256: string
  target_version: string
  package_family: string
  build_id: string
}
export interface Observation extends ObservationTarget {
  schema_version: '1.0.0'
  event_id: string
  update_id: string
  installation_ref: string
  from_version: string
  stage: 'handoff' | 'verify' | 'restart_confirmation'
  outcome: 'pending' | 'unknown' | 'failed' | 'target_confirmed'
  occurred_at: string
  observed_version?: string
  observed_family?: string
  observed_sha256?: string
  error_code?: string
}

function validObservation(value: unknown): value is Observation {
  if (!value || typeof value !== 'object') { return false }
  const e = value as Record<string, unknown>
  const text = ['event_id','update_id','installation_ref','release_id','target_version','package_family','build_id','from_version']

  return e.schema_version === '1.0.0' && text.every(k => typeof e[k] === 'string' && (e[k] as string).length > 0 && (e[k] as string).length <= 160)
    && typeof e.target_sha256 === 'string' && /^[a-f0-9]{64}$/.test(e.target_sha256)
    && typeof e.occurred_at === 'string' && Number.isFinite(Date.parse(e.occurred_at))
    && ['handoff','verify','restart_confirmation'].includes(String(e.stage))
    && ['pending','unknown','failed','target_confirmed'].includes(String(e.outcome))
}

interface Journal {
  enabled: boolean
  profile: string | null
  queue: Observation[]
  pending: Observation | null
  dropped: number
}
export class SemantierObservations {
  private state: Journal
  private file: string
  private storageUnavailable = false
  constructor(dir: string) {
    this.file = path.join(dir, 'semantier-update-observations.json')
    this.state = { enabled: false, profile: null, queue: [], pending: null, dropped: 0 }

    try {
      if (!fs.existsSync(this.file)) {
        return
      }

      const info = fs.lstatSync(this.file)

      if (info.isSymbolicLink() || info.size > 1024 * 1024 + 65536) {
        throw new Error('Invalid optional journal')
      }

      const parsed = JSON.parse(fs.readFileSync(this.file, 'utf8'))

      if (
        !parsed ||
        typeof parsed.enabled !== 'boolean' ||
        !Array.isArray(parsed.queue) ||
        parsed.queue.length > 1000 ||
        !parsed.queue.every(validObservation) ||
        (parsed.pending !== null && !validObservation(parsed.pending)) ||
        !Number.isInteger(parsed.dropped) ||
        parsed.dropped < 0 ||
        (parsed.profile !== null && typeof parsed.profile !== 'string')
      ) {
        throw new Error('Invalid optional journal')
      }

      this.state = parsed
    } catch {
      this.storageUnavailable = true
    }
  }
  private save() {
    const tmp = this.file + '.' + randomUUID() + '.tmp'

    try {
      fs.mkdirSync(path.dirname(this.file), { recursive: true })
      const file = fs.openSync(tmp, 'wx', 0o600)

      try {
        fs.writeFileSync(file, JSON.stringify(this.state))
        fs.fsyncSync(file)
      } finally {
        fs.closeSync(file)
      }

      fs.renameSync(tmp, this.file)
      this.storageUnavailable = false
    } catch {
      this.storageUnavailable = true
    } finally {
      try { if (fs.existsSync(tmp)) { fs.unlinkSync(tmp) } } catch { this.storageUnavailable = true }
    }
  }
  consent(enabled: boolean, profile: string) {
    if (!profile || profile.length > 128) {
      throw new Error('Original local profile required')
    }

    this.state.enabled = enabled
    this.state.profile = enabled ? profile : null

    if (!enabled) {
      this.state.queue = []
      this.state.pending = null
    }

    this.save()
  }
  status() {
    return {
      storageUnavailable: this.storageUnavailable,
      enabled: this.state.enabled,
      profile: this.state.profile,
      queued: this.state.queue.length,
      dropped: this.state.dropped
    }
  }
  private enqueue(event: Observation) {
    if (!this.state.enabled) {
      return
    }

    this.state.queue.push(event)
    const floor = Date.now() - 7 * 86400000
    const kept = this.state.queue.filter(e => Date.parse(e.occurred_at) >= floor)
    this.state.dropped += this.state.queue.length - kept.length
    this.state.queue = kept

    while (this.state.queue.length > 1000 || Buffer.byteLength(JSON.stringify(this.state.queue)) > 1024 * 1024) {
      this.state.queue.shift()
      this.state.dropped++
    }

    this.save()
  }
  begin(target: ObservationTarget, fromVersion: string) {
    if (!this.state.enabled) {
      return
    }

    const event: Observation = {
      ...target,
      schema_version: '1.0.0',
      event_id: randomUUID(),
      update_id: randomUUID(),
      installation_ref: 'native-owner-resolves',
      from_version: fromVersion,
      stage: 'handoff',
      outcome: 'pending',
      occurred_at: new Date().toISOString()
    }

    this.state.pending = event
    this.enqueue(event)
  }
  failed(code: string) {
    const pending = this.state.pending

    if (!pending || !this.state.enabled) {
      return
    }

    this.enqueue({
      ...pending,
      event_id: randomUUID(),
      stage: 'verify',
      outcome: 'failed',
      error_code: code,
      occurred_at: new Date().toISOString()
    })
    this.state.pending = null
    this.save()
  }
  reconcile(actual: { version: string; family: string; buildId: string }) {
    const p = this.state.pending

    if (!p || !this.state.enabled) {
      return
    }

    const exact =
      p.target_version === actual.version && p.package_family === actual.family && p.build_id === actual.buildId

    const already = this.state.queue.some(
      e =>
        e.update_id === p.update_id &&
        e.stage === 'restart_confirmation' &&
        e.outcome === (exact ? 'target_confirmed' : 'unknown')
    )

    if (already) {
      return
    }

    this.enqueue({
      ...p,
      event_id: randomUUID(),
      stage: 'restart_confirmation',
      outcome: exact ? 'target_confirmed' : 'unknown',
      occurred_at: new Date().toISOString(),
      observed_version: actual.version,
      observed_family: actual.family,
      ...(exact ? { observed_sha256: p.target_sha256 } : {})
    })

    if (exact) {
      this.state.pending = null
      this.save()
    }
  }
  drain(profile: string) {
    if (!this.state.enabled || this.storageUnavailable || this.state.profile !== profile) {
      return []
    }

    const kept = this.state.queue.filter(e => Date.parse(e.occurred_at) >= Date.now() - 7 * 86400000)

    if (kept.length !== this.state.queue.length) {
      this.state.dropped += this.state.queue.length - kept.length
      this.state.queue = kept
      this.save()
    }

    return this.state.queue.slice(0, 100).map(({ build_id: _, ...event }) => event)
  }
  ack(profile: string, ids: string[]) {
    if (profile !== this.state.profile) {
      return
    }

    this.state.queue = this.state.queue.filter(e => !ids.includes(e.event_id))
    this.save()
  }
}
