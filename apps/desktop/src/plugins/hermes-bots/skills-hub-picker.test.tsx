/** Native Bot discovery must bind search and confirmed install to the same owner. */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

const mocks = vi.hoisted(() => ({
  notify: vi.fn(),
  notifyError: vi.fn(),
  request: vi.fn(async (_method: string, _params: Record<string, unknown>) => ({}) as Record<string, unknown>),
  requestProfile: vi.fn(
    async (_route: unknown, _method: string, _params: Record<string, unknown>) => ({}) as Record<string, unknown>
  )
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const original = await importOriginal<typeof HermesSdk>()

  return {
    ...original,
    usePluginI18n: () => translateBots,
    host: {
      ...original.host,
      notify: mocks.notify,
      notifyError: mocks.notifyError,
      request: mocks.request,
      requestProfile: mocks.requestProfile
    }
  }
})

const { HubSkillsSection } = await import('./skills-hub')

const bot = {
  name: 'worker',
  remoteSource: true,
  sourceScoped: true,
  route: { connectionId: 'remote-a', mode: 'remote' as const, profile: 'worker', targetProfile: 'backend-worker' }
}

beforeEach(() => {
  vi.clearAllMocks()
  mocks.request.mockImplementation(async (_method, params) =>
    params.action === 'search' ? { results: [{ name: 'research', description: 'Research' }] } : { installed: true }
  )
  mocks.requestProfile.mockImplementation(async (_route, _method, params) =>
    params.action === 'search' ? { results: [{ name: 'research', description: 'Research' }] } : { installed: true }
  )
})
afterEach(cleanup)

async function search() {
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'research' } })
  fireEvent.click(screen.getByRole('button', { name: 'Search' }))
  await screen.findByText('research', { exact: true })
}

it('routes native discovery and installation to the same Bot owner, without frames or page messages', async () => {
  render(<HubSkillsSection bot={bot} />)
  await search()
  expect(document.querySelector('iframe')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: /research/i }))
  await waitFor(() =>
    expect(mocks.requestProfile).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'remote-a' }),
      'skills.manage',
      { action: 'install', profile: 'backend-worker', query: 'research' }
    )
  )
  expect(mocks.request).not.toHaveBeenCalled()
  await waitFor(() => expect(mocks.notify).toHaveBeenCalledTimes(1))
  window.dispatchEvent(
    new MessageEvent('message', {
      origin: 'https://hermes-agent.nousresearch.com',
      data: { type: 'hermes-skill-pick', name: 'unexpected' }
    })
  )
  expect(mocks.requestProfile.mock.calls.filter(([, , p]) => p.action === 'install')).toHaveLength(1)
})

it('requires confirmed success and permits retry after rejected installation', async () => {
  mocks.request.mockImplementation(async (_method, params) =>
    params.action === 'search' ? { results: [{ name: 'research' }] } : { installed: false }
  )
  render(<HubSkillsSection />)
  await search()
  fireEvent.click(screen.getByRole('button', { name: /research/i }))
  await waitFor(() => expect(mocks.notifyError).toHaveBeenCalled())
  expect(mocks.notify).not.toHaveBeenCalled()
  expect(screen.getByRole('button', { name: /research/i })).toBeTruthy()
})

it('drops late discovery results after changing Bot profile', async () => {
  let resolve!: (value: Record<string, unknown>) => void
  mocks.requestProfile.mockImplementation(
    () =>
      new Promise(r => {
        resolve = r
      })
  )
  const view = render(<HubSkillsSection bot={bot} />)
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'research' } })
  fireEvent.click(screen.getByRole('button', { name: 'Search' }))
  view.rerender(
    <HubSkillsSection
      bot={{ ...bot, name: 'other', route: { ...bot.route, profile: 'other', targetProfile: 'other' } }}
    />
  )
  await act(async () => resolve({ results: [{ name: 'old-profile' }] }))
  expect(screen.queryByText('old-profile')).toBeNull()
})
