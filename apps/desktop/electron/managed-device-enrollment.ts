import { createHash, randomUUID } from 'node:crypto'

import type { createSemantierWorkspaceClient } from './semantier-workspace-client'
import {
  assertWorkspaceScope,
  normalizeSemantierWorkspaceOrigin,
  type WorkspaceScope
} from './semantier-workspace-client'

type Binding = { keyId: string; deviceId?: string; replacesDeviceId?: string }
type State = { version: 1; bindings: Record<string, Binding> }
type Client = Pick<
  ReturnType<typeof createSemantierWorkspaceClient>,
  'identity' | 'managedDevices' | 'managedDeviceChallenge' | 'managedDeviceEnroll'
>
export type KeyRequest = {
  operation: 'create' | 'public' | 'sign_enrollment'
  key_id: string
  context?: Record<string, string | null>
  expected_scope?: Record<string, string>
}
export type PublicProof = { key_id: string; public_key: string; signature?: string; assurance: string }
interface Options {
  client: (origin: string) => Client
  key: (request: KeyRequest) => Promise<PublicProof>
  read: () => unknown
  write: (state: State) => void
  attest?: (context: Record<string, string | null>, keyId: string) => Promise<Record<string, unknown>>
}

/** Key references are local recovery data, never account or enrollment authority. */
function stateFrom(value: unknown): State {
  if (value === undefined) {
    return { version: 1, bindings: {} }
  }

  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Managed device recovery data is unavailable')
  }

  const raw = value as Record<string, unknown>

  if (
    raw.version !== 1 ||
    Object.keys(raw).some(k => k !== 'version' && k !== 'bindings') ||
    !raw.bindings ||
    typeof raw.bindings !== 'object' ||
    Array.isArray(raw.bindings)
  ) {
    throw new Error('Managed device recovery data is unavailable')
  }

  const bindings: Record<string, Binding> = {}

  for (const [id, item] of Object.entries(raw.bindings)) {
    if (!/^[a-f0-9]{64}$/.test(id) || !item || typeof item !== 'object' || Array.isArray(item)) {
      throw new Error('Managed device recovery data is unavailable')
    }

    const b = item as Record<string, unknown>

    if (
      Object.keys(b).some(k => !['keyId', 'deviceId', 'replacesDeviceId'].includes(k)) ||
      typeof b.keyId !== 'string' ||
      !/^[a-f0-9]{32}$/.test(b.keyId) ||
      ['deviceId', 'replacesDeviceId'].some(
        k => b[k] !== undefined && (typeof b[k] !== 'string' || !/^[A-Za-z0-9_-]{1,128}$/.test(b[k] as string))
      )
    ) {
      throw new Error('Managed device recovery data is unavailable')
    }

    bindings[id] = {
      keyId: b.keyId,
      ...(b.deviceId ? { deviceId: b.deviceId as string } : {}),
      ...(b.replacesDeviceId ? { replacesDeviceId: b.replacesDeviceId as string } : {})
    }
  }

  return { version: 1, bindings }
}

export class ManagedDeviceEnrollment {
  private readonly jobs = new Map<string, Promise<unknown>>()
  constructor(private readonly options: Options) {}
  async enroll(origin: string, scope: WorkspaceScope, replacesDeviceId?: string) {
    const canonical = normalizeSemantierWorkspaceOrigin(origin)

    const id = createHash('sha256')
      .update(JSON.stringify([canonical, scope.userId, scope.organizationId, scope.workspaceId]))
      .digest('hex')

    const previous = this.jobs.get(id) || Promise.resolve()
    const job = previous.catch(() => {}).then(() => this.perform(canonical, scope, id, replacesDeviceId))
    this.jobs.set(id, job)

    try {
      return await job
    } finally {
      if (this.jobs.get(id) === job) {
        this.jobs.delete(id)
      }
    }
  }
  private save(id: string, binding: Binding) {
    // Merge the latest owner snapshot after awaits so other accounts are retained.
    const state = stateFrom(this.options.read())
    this.options.write({ version: 1, bindings: { ...state.bindings, [id]: binding } })
  }
  private async perform(origin: string, scope: WorkspaceScope, id: string, replacesDeviceId?: string) {
    const client = this.options.client(origin)
    const current = async () => assertWorkspaceScope(await client.identity(), scope)
    await current()

    const inventory = await client.managedDevices()
    await current()
    const keyProofOnly = inventory.enrollment_policy === 'development_key_proof'

    if (!keyProofOnly && !this.options.attest) {
      throw new Error('Verified hardware attestation is unavailable')
    }

    if (inventory.enrollment_available === false) {
      throw new Error('Server managed enrollment is unavailable')
    }

    if (replacesDeviceId !== undefined && !/^[A-Za-z0-9_-]{1,128}$/.test(replacesDeviceId)) {
      throw new Error('Exact replacement device required')
    }

    const selected = inventory.devices.find(row => row.device_id === replacesDeviceId)
    const serverReplacement = selected?.status === 'revoked' ? undefined : replacesDeviceId
    let binding = stateFrom(this.options.read()).bindings[id]

    if (binding?.deviceId && replacesDeviceId && binding.deviceId !== replacesDeviceId) {
      throw new Error('This PC is already registered; revoke the other PC separately')
    }

    let created = false

    if (!binding || (replacesDeviceId && binding.deviceId)) {
      binding = {
        keyId: randomUUID().replaceAll('-', ''),
        ...(serverReplacement ? { replacesDeviceId: serverReplacement } : {})
      }
      this.save(id, binding)
      created = true
    }

    const publicProof = await this.options.key({ operation: created ? 'create' : 'public', key_id: binding.keyId })

    if (publicProof.key_id !== binding.keyId || typeof publicProof.public_key !== 'string' || !publicProof.public_key) {
      throw new Error('Native key response mismatch')
    }

    await current()

    const attempt = async (intent: 'register' | 'replace' | 'reconnect') => {
      const ch = await client.managedDeviceChallenge({
        public_key: publicProof.public_key,
        intent,
        replaces_device_id: intent === 'replace' ? binding.replacesDeviceId : null
      })

      await current()

      if (
        ch.context.user_id !== scope.userId ||
        ch.context.organization_id !== scope.organizationId ||
        ch.context.workspace_id !== scope.workspaceId
      ) {
        throw new Error('Enrollment challenge scope changed')
      }

      const proof = await this.options.key({
        operation: 'sign_enrollment',
        key_id: binding.keyId,
        context: ch.context,
        expected_scope: {
          user_id: scope.userId,
          organization_id: scope.organizationId,
          workspace_id: scope.workspaceId
        }
      })

      if (
        proof.key_id !== binding.keyId ||
        proof.public_key !== publicProof.public_key ||
        !proof.signature ||
        !ch.context.challenge_id
      ) {
        throw new Error('Native key proof mismatch')
      }

      const attestation = keyProofOnly ? undefined : await this.options.attest!(ch.context, binding.keyId)
      await current()

      return client.managedDeviceEnroll({
        challenge_id: ch.context.challenge_id,
        signature: proof.signature,
        ...(attestation === undefined ? {} : { attestation })
      })
    }

    let receipt

    if (created) {
      receipt = await attempt(binding.replacesDeviceId ? 'replace' : 'register')
    } else {
      try {
        receipt = await attempt('reconnect')
      } catch (error) {
        const e = error as { statusCode?: number; managedDeviceCode?: string }

        if (binding.deviceId || e.statusCode !== 404 || e.managedDeviceCode !== 'device_unavailable') {
          throw error
        }

        // Only an explicit server miss permits completing the original pending intent.
        if (replacesDeviceId !== undefined && binding.replacesDeviceId !== serverReplacement) {
          binding = { keyId: binding.keyId, ...(serverReplacement ? { replacesDeviceId: serverReplacement } : {}) }
          this.save(id, binding)
        }

        receipt = await attempt(binding.replacesDeviceId ? 'replace' : 'register')
      }
    }

    if (
      receipt.status !== 'active' ||
      receipt.assurance !== (keyProofOnly ? 'NOT_VERIFIED' : 'TPM_ATTESTED') ||
      (keyProofOnly && (receipt.hardware_attestation !== 'NOT_VERIFIED' || receipt.hardware_verified !== false)) ||
      !/^[A-Za-z0-9_-]{1,128}$/.test(receipt.device_id)
    ) {
      throw new Error('Qualified enrollment receipt required')
    }

    this.save(id, { keyId: binding.keyId, deviceId: receipt.device_id })
    await current()

    return receipt
  }
}
