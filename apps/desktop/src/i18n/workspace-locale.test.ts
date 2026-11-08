import { beforeEach, expect, it } from 'vitest'
import { workspaceLocaleClient } from './workspace-locale'
const scope = { userId: 'a', workspaceId: 'a', organizationId: 'org' }
beforeEach(() => localStorage.clear())
it('persists Cloud chrome language without Native configuration', async () => {
  const client = workspaceLocaleClient('https://cloud.example', scope)!
  expect(await client.getConfig()).toEqual({})
  await client.saveConfig({ display: { language: 'zh-hant' } })
  expect(await workspaceLocaleClient('https://cloud.example', scope)!.getConfig()).toEqual({ display: { language: 'zh-hant' } })
})
it('isolates account, workspace, organization and origin', async () => {
  await workspaceLocaleClient('https://cloud.example', scope)!.saveConfig({ display: { language: 'zh' } })
  for (const client of [workspaceLocaleClient('https://other.example', scope), workspaceLocaleClient('https://cloud.example', { ...scope, userId: 'b' }), workspaceLocaleClient('https://cloud.example', { ...scope, workspaceId: 'b' }), workspaceLocaleClient('https://cloud.example', { ...scope, organizationId: 'other' })]) {
    expect(await client!.getConfig()).toEqual({})
  }
})
it('has no ambient owner before authentication', () => {
  expect(workspaceLocaleClient('https://cloud.example', undefined)).toBeNull()
})
