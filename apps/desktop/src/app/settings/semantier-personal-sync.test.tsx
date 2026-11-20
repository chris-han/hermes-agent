import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { expect, it, vi } from 'vitest'

import { SemantierPersonalSync } from './semantier-personal-sync'

vi.mock('@/i18n', () => ({ useI18n: () => ({ locale: 'en' }) }))
const scope = { userId: 'u', workspaceId: 'ws', organizationId: 'org' }

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

const context = { source_file: 'MEMORY.md', content: 'Use a concise table.', content_hash: 'sha256:' + '3'.repeat(64) }

const preview = {
  skill_id: 'my-report',
  files: [{ relative_path: 'SKILL.md', content_hash: 'sha256:' + '4'.repeat(64) }],
  contexts: [context],
  snapshot_hash: 'sha256:' + '5'.repeat(64)
}

const source = {
  receipt,
  source_transfer: { artifact_hash: 'sha256:' + '6'.repeat(64) },
  skill_id: 'my-report',
  source_revision: '1:1',
  owner_ref: 'u'
}

it('requires source/context selection, then keeps share, admission and installation distinct', async () => {
  const api = {
    personalPreview: vi.fn().mockResolvedValue(preview),
    personalShare: vi.fn().mockResolvedValue({ receipt }),
    personalSources: vi.fn().mockResolvedValue({ sources: [source], has_more: false }),
    personalPrepare: vi
      .fn()
      .mockResolvedValue({
        receipt,
        candidate,
        source: {
          artifact_hash: source.source_transfer.artifact_hash,
          skill: {
            files: [{ relative_path: 'SKILL.md', content_base64: btoa('Check the report and present a table.') }]
          },
          decision_context: { content: context.content }
        }
      }),
    personalAdmit: vi.fn().mockResolvedValue({ receipt, candidate }),
    personalInstall: vi.fn().mockResolvedValue({ status: 'installed' })
  }

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: api } })
  render(<SemantierPersonalSync origin="https://cloud.example" scope={scope} />)
  expect(api.personalPreview).not.toHaveBeenCalled()
  fireEvent.change(screen.getByLabelText('Local Skill name'), { target: { value: 'my-report' } })
  fireEvent.click(screen.getByRole('button', { name: 'Preview local source' }))
  await screen.findByText(context.content)
  expect((screen.getByRole('button', { name: 'Share this exact source' }) as HTMLButtonElement).disabled).toBe(true)
  fireEvent.click(screen.getByRole('radio'))
  fireEvent.click(screen.getByRole('button', { name: 'Share this exact source' }))
  await screen.findByRole('button', { name: 'Review this revision' })
  expect(api.personalShare).toHaveBeenCalledWith('https://cloud.example', scope, preview, context)
  expect(api.personalAdmit).not.toHaveBeenCalled()
  expect(api.personalInstall).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Review this revision' }))
  await screen.findByText('Check the report and present a table.')
  fireEvent.click(screen.getByRole('button', { name: 'Enable in Cloud' }))
  await waitFor(() =>
    expect(api.personalAdmit).toHaveBeenCalledWith('https://cloud.example', scope, receipt, candidate)
  )
  await screen.findByText(/Cloud admission completed/)
  expect(api.personalInstall).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Install approved revision on this PC' }))
  await screen.findByText('Installed for the next session.')
  expect(api.personalInstall).toHaveBeenCalledWith('https://cloud.example', scope, receipt, candidate, false)
})

it('keeps failed source upload separate from successful cloud admission', async () => {
  const api = {
    personalSources: vi.fn().mockRejectedValue(new Error('expired session')),
    personalAdmit: vi.fn(),
    personalInstall: vi.fn()
  }

  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { workspace: api } })
  render(<SemantierPersonalSync origin="https://cloud.example" scope={scope} />)
  fireEvent.click(screen.getByRole('button', { name: 'Refresh personal sources' }))
  await screen.findByRole('alert')
  expect(api.personalAdmit).not.toHaveBeenCalled()
  expect(api.personalInstall).not.toHaveBeenCalled()
})
