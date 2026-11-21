import { describe, expect, it, vi } from 'vitest'

import { PersonalCapabilitySync, type PersonalSnapshot } from './personal-capability-sync'

const scope = { userId: 'u', organizationId: 'org', workspaceId: 'ws' }
const identity = { authenticated: true, user: { user_id: 'u' }, workspace: { id: 'ws' }, organization_id: 'org' }

const receipt = {
  artifact_kind: 'personal_capability_import',
  artifact_ref: 'ack',
  artifact_hash: 'sha256:' + '1'.repeat(64)
}

const candidate = {
  artifact_kind: 'change_candidate',
  artifact_ref: 'candidate',
  artifact_hash: 'sha256:' + '2'.repeat(64)
}

const selected: PersonalSnapshot = {
  skill_id: 'my-report',
  files: [],
  contexts: [{ source_file: 'MEMORY.md', content: 'Selected context', content_hash: 'sha256:' + '3'.repeat(64) }],
  snapshot_hash: 'sha256:' + '4'.repeat(64)
}

function fixture() {
  let saved: unknown

  const client = {
    identity: vi.fn().mockResolvedValue(identity),
    personalSources: vi.fn().mockResolvedValue({ sources: [], has_more: false }),
    personalRead: vi.fn(),
    personalPropose: vi.fn(),
    personalLearningAction: vi.fn(),
    personalPackage: vi.fn(),
    personalUpload: vi.fn().mockImplementation(async value => ({
      ...receipt,
      ...scopeFields,
      transfer: { artifact_ref: value.artifact_ref, artifact_hash: value.artifact_hash },
      status: 'SOURCE_IMPORTED'
    }))
  }

  const native = vi.fn().mockImplementation(async (path, body) => {
    if (path.endsWith('/capture')) {
      return (
        body.previous || {
          ...scopeFields,
          artifact_ref: 'source',
          artifact_hash: 'sha256:' + '5'.repeat(64),
          source_store_incarnation: body.incarnation,
          source_revision: '1:1'
        }
      )
    }

    return selected
  })

  const sync = new PersonalCapabilitySync({
    client: () => client as never,
    native,
    read: () => saved,
    write: value => {
      saved = structuredClone(value)
    }
  })

  return { sync, client, native, state: () => saved }
}

const scopeFields = { owner_ref: 'u', organization_id: 'org', workspace_id: 'ws' }

describe('account-first personal source overlay', () => {
  it('recovers lost upload ACK without a new source identity or device enrollment', async () => {
    const f = fixture()
    f.client.personalUpload.mockRejectedValueOnce(new Error('response lost'))
    await expect(f.sync.share('https://cloud.example', scope, selected, selected.contexts[0])).rejects.toThrow(
      'response lost'
    )
    const first = f.client.personalUpload.mock.calls[0][0]
    const result = await f.sync.share('https://cloud.example', scope, selected, selected.contexts[0])
    expect(result.receipt).toEqual(receipt)
    expect(f.client.personalUpload.mock.calls[1][0]).toEqual(first)
    expect(f.native.mock.calls.filter(([path]) => path.endsWith('/capture'))[1][1].previous).toEqual(first)
    expect(f.state()).toBeTruthy()
  })
  it('rejects account changes before any native source read or cloud write', async () => {
    const f = fixture()
    f.client.identity.mockResolvedValue({ ...identity, user: { user_id: 'other' } })
    await expect(f.sync.preview('https://cloud.example', scope, 'my-report')).rejects.toThrow()
    expect(f.native).not.toHaveBeenCalled()
    expect(f.client.personalUpload).not.toHaveBeenCalled()
  })
  it('rejects a foreign package or changed exact selection before local install', async () => {
    const f = fixture()
    f.client.personalPackage.mockResolvedValue({
      source_import: receipt,
      candidate,
      source_transfer: { ...scopeFields, owner_ref: 'other' },
      approved_version: scopeFields
    })
    await expect(f.sync.install('https://cloud.example', scope, receipt, candidate)).rejects.toThrow('account changed')
    expect(f.native).not.toHaveBeenCalled()
    f.client.personalPackage.mockResolvedValue({
      source_import: { ...receipt, artifact_ref: 'different' },
      candidate,
      source_transfer: scopeFields,
      approved_version: scopeFields
    })
    await expect(f.sync.install('https://cloud.example', scope, receipt, candidate)).rejects.toThrow(
      'selection changed'
    )
    expect(f.native).not.toHaveBeenCalled()
  })
  it('uses only the normal native install path after current cloud authorization', async () => {
    const f = fixture()

    const packageValue = {
      source_import: receipt,
      candidate,
      source_transfer: scopeFields,
      approved_version: scopeFields
    }

    f.client.personalPackage.mockResolvedValue(packageValue)
    f.native.mockResolvedValue({ status: 'installed' })
    await expect(f.sync.install('https://cloud.example', scope, receipt, candidate)).resolves.toEqual({
      status: 'installed'
    })
    expect(f.native).toHaveBeenCalledWith('/api/skills/personal/install', {
      package: packageValue,
      confirmed: true,
      replace_existing: false
    })
  })
  it('does not continue admission after a normal assessment refusal', async () => {
    const f = fixture()
    f.client.personalLearningAction.mockRejectedValue(new Error('ineligible'))
    await expect(f.sync.admit('https://cloud.example', scope, receipt, candidate)).rejects.toThrow('ineligible')
    expect(f.client.personalLearningAction).toHaveBeenCalledTimes(1)
  })
})
