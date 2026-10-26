import { randomUUID } from 'node:crypto'

import type { ActiveWork } from '../quit-guard'

export interface UpdateWorkReview {
  token: string
  currentVersion: string
  targetVersion: string
  work: ActiveWork
  stopsWork: boolean
}
interface Snapshot {
  work: ActiveWork
  revision: number
  stopsWork: boolean
}
interface Approval {
  token: string
  target: string
  snapshot: string
  expires: number
}

/** Main-process consent, bound to the selected artifact and reported work.
 * A renderer confirmation cannot authorize a later target or changed work. */
export class UpdateWorkGuard {
  private pending?: Approval
  private allowedSnapshot?: string
  constructor(private readonly read: () => Snapshot, private readonly now = Date.now) {}

  review(target: string, currentVersion: string, targetVersion: string, token?: unknown): UpdateWorkReview | null {
    const state = this.read()
    const snapshot = JSON.stringify(state)

    const valid = this.pending && this.pending.token === token && this.pending.target === target &&
      this.pending.snapshot === snapshot && this.pending.expires > this.now()

    if (!state.work.count || valid) {
      this.allowedSnapshot = snapshot
      this.pending = undefined

      return null
    }

    this.pending = { token: randomUUID(), target, snapshot, expires: this.now() + 120_000 }
    this.allowedSnapshot = undefined

    return { token: this.pending.token, currentVersion, targetVersion, work: state.work, stopsWork: state.stopsWork }
  }

  /** The token expires before acceptance, not during a large download. The
   * accepted operation still rechecks work immediately before stopping it. */
  assertBeforeStop(): void {
    if (this.allowedSnapshot === undefined) { return }
    const state = this.read()

    if (state.work.count && (JSON.stringify(state) !== this.allowedSnapshot)) {
      throw new Error('Active work changed. Review the update again before restarting.')
    }
  }

  finish(): void { this.allowedSnapshot = undefined }
}
