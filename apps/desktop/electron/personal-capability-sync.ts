import { createHash, randomUUID } from 'node:crypto'

import {
  assertWorkspaceScope,
  type createSemantierWorkspaceClient,
  normalizeSemantierWorkspaceOrigin,
  type WorkspaceScope
} from './semantier-workspace-client'

type Json = Record<string, unknown>
export type PersonalPin = { artifact_kind: string; artifact_ref: string; artifact_hash: string }
export type PersonalSnapshot = {
  skill_id: string
  files: Json[]
  contexts: Array<{ source_file: string; content: string; content_hash: string }>
  snapshot_hash: string
}
export type PersonalSource = {
  receipt: PersonalPin
  source_transfer: PersonalPin
  skill_id: string
  source_revision: string
  owner_ref: string
}
type Binding = { incarnation: string; transfer?: Json; acknowledgement?: PersonalPin }
type State = { version: 1; bindings: Record<string, Binding> }
type Client = Pick<
  ReturnType<typeof createSemantierWorkspaceClient>,
  | 'identity'
  | 'personalSources'
  | 'personalRead'
  | 'personalUpload'
  | 'personalPropose'
  | 'personalPackage'
  | 'personalLearningAction'
>
interface Options {
  client: (origin: string) => Client
  native: (path: string, body: Json) => Promise<unknown>
  read: () => unknown
  write: (state: State) => void
}

function pin(value: unknown, kind: string): PersonalPin {
  const p = value as PersonalPin

  if (
    !p ||
    typeof p !== 'object' ||
    Object.keys(p).sort().join(',') !== 'artifact_hash,artifact_kind,artifact_ref' ||
    p.artifact_kind !== kind ||
    typeof p.artifact_ref !== 'string' ||
    !p.artifact_ref ||
    !/^sha256:[a-f0-9]{64}$/.test(p.artifact_hash)
  ) {
    throw new Error('Exact personal source pin required')
  }

  return p
}

function state(value: unknown): State {
  if (value === undefined) {
    return { version: 1, bindings: {} }
  }

  const s = value as State

  if (
    !s ||
    typeof s !== 'object' ||
    s.version !== 1 ||
    Object.keys(s).sort().join(',') !== 'bindings,version' ||
    !s.bindings ||
    typeof s.bindings !== 'object' ||
    Array.isArray(s.bindings)
  ) {
    throw new Error('Personal source recovery data is unavailable')
  }

  for (const [key, b] of Object.entries(s.bindings)) {
    if (
      !/^[a-f0-9]{64}$/.test(key) ||
      !b ||
      typeof b !== 'object' ||
      !/^[a-f0-9]{32}$/.test(b.incarnation) ||
      Object.keys(b).some(k => !['incarnation', 'transfer', 'acknowledgement'].includes(k))
    ) {
      throw new Error('Personal source recovery data is unavailable')
    }

    if (b.transfer && (typeof b.transfer !== 'object' || b.transfer.source_store_incarnation !== b.incarnation)) {
      throw new Error('Personal source recovery data is unavailable')
    }

    if (b.acknowledgement && !b.transfer) {
      throw new Error('Personal source recovery data is unavailable')
    }

    if (b.acknowledgement) {
      pin(b.acknowledgement, 'personal_capability_import')
    }
  }

  return s
}

function scopeOf(value: Json, scope: WorkspaceScope) {
  if (
    value.owner_ref !== scope.userId ||
    value.organization_id !== scope.organizationId ||
    value.workspace_id !== scope.workspaceId
  ) {
    throw new Error('Personal source account changed')
  }
}

/** Authenticated personal-service overlay; no CNG, device quota or agent mode change. */
export class PersonalCapabilitySync {
  private jobs = new Map<string, Promise<unknown>>()
  constructor(private readonly options: Options) {}
  private async current(origin: string, scope: WorkspaceScope) {
    const client = this.options.client(normalizeSemantierWorkspaceOrigin(origin))
    assertWorkspaceScope(await client.identity(), scope)

    return client
  }
  private save(key: string, binding: Binding) {
    const latest = state(this.options.read())
    this.options.write({ ...latest, bindings: { ...latest.bindings, [key]: binding } })
  }
  async preview(origin: string, scope: WorkspaceScope, skillId: string): Promise<PersonalSnapshot> {
    await this.current(origin, scope)

    const value = (await this.options.native('/api/skills/personal/snapshot', {
      skill_id: skillId
    })) as PersonalSnapshot

    await this.current(origin, scope)

    return value
  }
  async sources(origin: string, scope: WorkspaceScope): Promise<{ sources: PersonalSource[]; has_more: boolean }> {
    const client = await this.current(origin, scope)
    const value = await client.personalSources()
    await this.current(origin, scope)

    if (
      !value ||
      !Array.isArray(value.sources) ||
      value.sources.some((row: PersonalSource) => row.owner_ref !== scope.userId)
    ) {
      throw new Error('Personal source scope mismatch')
    }

    return value
  }
  share(
    origin: string,
    scope: WorkspaceScope,
    selected: PersonalSnapshot,
    contextRecord: PersonalSnapshot['contexts'][number]
  ) {
    const key = createHash('sha256')
      .update(
        JSON.stringify([
          normalizeSemantierWorkspaceOrigin(origin),
          scope.userId,
          scope.organizationId,
          scope.workspaceId,
          selected.skill_id
        ])
      )
      .digest('hex')

    const previous = this.jobs.get(key) || Promise.resolve()

    const job = previous
      .catch(() => {})
      .then(async () => {
        const client = await this.current(origin, scope)
        let binding = state(this.options.read()).bindings[key] || { incarnation: randomUUID().replaceAll('-', '') }
        this.save(key, binding)

        const upload = async () => {
          const transfer = binding.transfer!
          scopeOf(transfer, scope)
          await this.current(origin, scope)
          const result = await client.personalUpload(transfer)
          scopeOf(result, scope)

          if (
            result.status !== 'SOURCE_IMPORTED' ||
            result.transfer?.artifact_ref !== transfer.artifact_ref ||
            result.transfer?.artifact_hash !== transfer.artifact_hash
          ) {
            throw new Error('Personal source conflict; retained source was not replaced')
          }

          binding = {
            ...binding,
            acknowledgement: pin(
              {
                artifact_kind: 'personal_capability_import',
                artifact_ref: result.artifact_ref,
                artifact_hash: result.artifact_hash
              },
              'personal_capability_import'
            )
          }
          // Commit exact old-scope ACK before checking a possible account switch.
          this.save(key, binding)
          await this.current(origin, scope)
        }

        if (binding.transfer && !binding.acknowledgement) {
          await upload()
        }

        await this.current(origin, scope)

        const transfer = (await this.options.native('/api/skills/personal/capture', {
          selected,
          context_record: contextRecord,
          scope: { organization_id: scope.organizationId, workspace_id: scope.workspaceId, owner_ref: scope.userId },
          incarnation: binding.incarnation,
          previous: binding.transfer || null,
          acknowledgement: binding.acknowledgement || null
        })) as Json

        scopeOf(transfer, scope)

        if (transfer.source_store_incarnation !== binding.incarnation) {
          throw new Error('Personal source incarnation changed')
        }

        await this.current(origin, scope)

        if (transfer.artifact_hash !== binding.transfer?.artifact_hash) {
          binding = { incarnation: binding.incarnation, transfer }
          this.save(key, binding)
        }

        if (!binding.acknowledgement) {
          await upload()
        }

        return {
          receipt: binding.acknowledgement!,
          source_transfer: { artifact_ref: transfer.artifact_ref, artifact_hash: transfer.artifact_hash },
          source_revision: transfer.source_revision
        }
      })

    this.jobs.set(key, job)
    void job
      .finally(() => {
        if (this.jobs.get(key) === job) {
          this.jobs.delete(key)
        }
      })
      .catch(() => {})

    return job
  }
  async prepare(origin: string, scope: WorkspaceScope, receipt: PersonalPin) {
    pin(receipt, 'personal_capability_import')
    const client = await this.current(origin, scope)
    const source = await client.personalRead(receipt)
    scopeOf(source, scope)
    const result = await client.personalPropose(receipt)
    await this.current(origin, scope)

    const candidate = pin(
      { artifact_kind: 'change_candidate', artifact_ref: result.candidate_ref, artifact_hash: result.candidate_hash },
      'change_candidate'
    )

    return { receipt, candidate, source }
  }
  async admit(origin: string, scope: WorkspaceScope, receipt: PersonalPin, candidate: PersonalPin) {
    pin(receipt, 'personal_capability_import')
    pin(candidate, 'change_candidate')
    const client = await this.current(origin, scope)
    let assessment: Json = {}

    for (const operation of ['assess_candidate', 'approve_candidate', 'activate_candidate', 'publish_procedure']) {
      await this.current(origin, scope)

      const result = await client.personalLearningAction({
        operation,
        request_id: 'personal-sync:' + operation + ':' + candidate.artifact_hash,
        candidate_ref: candidate.artifact_ref,
        candidate_hash: candidate.artifact_hash,
        ...(operation === 'approve_candidate'
          ? { assessment_ref: assessment.assessment_ref, assessment_hash: assessment.assessment_hash }
          : {})
      })

      if (operation === 'assess_candidate') {
        assessment = result.object
      }
    }

    await this.current(origin, scope)

    return { receipt, candidate }
  }
  async install(
    origin: string,
    scope: WorkspaceScope,
    receipt: PersonalPin,
    candidate: PersonalPin,
    replaceExisting = false
  ) {
    pin(receipt, 'personal_capability_import')
    pin(candidate, 'change_candidate')
    const client = await this.current(origin, scope)
    const packageValue = await client.personalPackage(receipt, candidate)

    for (const [actual, expected] of [
      [packageValue.source_import, receipt],
      [packageValue.candidate, candidate]
    ]) {
      if (!actual || Object.keys(expected).some(k => actual[k] !== expected[k as keyof PersonalPin])) {
        throw new Error('Personal package selection changed')
      }
    }

    scopeOf(packageValue.source_transfer, scope)
    scopeOf(packageValue.approved_version, scope)
    await this.current(origin, scope)

    const installed = await this.options.native('/api/skills/personal/install', {
      package: packageValue,
      confirmed: true,
      replace_existing: replaceExisting
    })

    await this.current(origin, scope)

    return installed
  }
}
