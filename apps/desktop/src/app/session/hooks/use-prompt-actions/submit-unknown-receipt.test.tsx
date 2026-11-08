import { WorkspaceRequestStopped } from '@/api/client'
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'
import { readPendingSubmissions } from '@/store/pending-submissions'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { useSubmitPrompt } from './submit'
import type { GatewayRequest } from './utils'

vi.mock('@/store/gateway', async original => ({
  ...(await original<Record<string, unknown>>()),
  retainGatewayForSessionTurn: vi.fn(async () => () => undefined)
}))

beforeEach(() => {
  window.localStorage.clear()
  $sessions.set([])
  $sessionStates.set({})
  $connection.set(null)
  $activeGatewayProfile.set('default')
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

type ClientSessionState = ReturnType<typeof createClientSessionState>

// Adolanium reg-4: the owner answers an exact retry with the EXISTING admission, which may be
// `unknown` (claimed, then the owner died mid-turn). That send was admitted; dropping it as a
// rejection hid the user's message and the retry identity, and invited a duplicate resend.
it('an exact retry answered `unknown` keeps the bubble, retires the retry identity and never replays', async () => {
  let state: ClientSessionState = createClientSessionState()

  const requestGateway = vi.fn(async (_method: string, params?: Record<string, unknown>) => {
    if (requestGateway.mock.calls.length === 1) {
      throw new Error('connection closed')
    }

    return { admission_id: params?.submission_id, session_id: 'runtime-a', status: 'unknown' } as never
  })

  const deps = {
    activeSessionIdRef: { current: 'runtime-a' as string | null },
    selectedStoredSessionIdRef: { current: 'stored-a' as string | null },
    busyRef: { current: false },
    copy: en.desktop,
    createBackendSessionForSend: vi.fn(async () => null),
    getRoutedStoredSessionId: () => null,
    getRuntimeIdForStoredSession: () => 'runtime-a',
    getRouteToken: () => 'same-route',
    requestGateway: requestGateway as GatewayRequest,
    runtimeIdByStoredSessionIdRef: { current: new Map([['stored-a', 'runtime-a']]) },
    resumeStoredSession: vi.fn(),
    syncAttachmentsForSubmit: vi.fn(async (sessionId: string) => ({ sessionId, attachments: [] })),
    updateSessionState: vi.fn((_sid: string, updater: (s: ClientSessionState) => ClientSessionState) => {
      state = updater(state)

      return state
    })
  }

  const { result } = renderHook(() => useSubmitPrompt(deps as never))
  await act(async () => {
    expect(await result.current('lost turn', { submission_id: 'lost-1' })).toBe(false)
  })
  await act(async () => {
    expect(await result.current('lost turn', { submission_id: 'lost-1' })).toBe(true)
  })

  expect(requestGateway.mock.calls.filter(([method]) => method === 'prompt.submit')).toHaveLength(2)
  expect(state.messages.filter(message => message.role === 'user')).toHaveLength(1)
  expect(state.busy).toBe(false)
  expect(readPendingSubmissions('stored-a')['lost-1']).toMatchObject({ status: 'unknown', text: 'lost turn' })
  expect(window.localStorage.getItem('hermes.desktop.preparedSubmissions.v1') ?? '').not.toContain('lost turn')
})


it.each([false,true])('a user-stopped HTTP send retires its retry without a false receipt or failure (admitted row=%s)', async admitted => {
  let state: ClientSessionState=createClientSessionState()
  const accepted=vi.fn()
  const requestGateway=vi.fn(async (_method:string,params?:Record<string,unknown>)=>{
    if(admitted){state={...state,messages:state.messages.map(m=>m.role==='user'?{...m,rowId:263}:m)};$sessionStates.set({'runtime-a':state})}
    throw new WorkspaceRequestStopped(String(params?.submission_id))
  })
  const deps={activeSessionIdRef:{current:'runtime-a'},selectedStoredSessionIdRef:{current:'stored-a'},busyRef:{current:false},copy:en.desktop,
    createBackendSessionForSend:vi.fn(async()=>null),getRoutedStoredSessionId:()=>null,getRuntimeIdForStoredSession:()=>'runtime-a',getRouteToken:()=>'same-route',
    requestGateway:requestGateway as GatewayRequest,runtimeIdByStoredSessionIdRef:{current:new Map([['stored-a','runtime-a']])},resumeStoredSession:vi.fn(),
    syncAttachmentsForSubmit:vi.fn(async(sessionId:string)=>({sessionId,attachments:[]})),
    updateSessionState:vi.fn((_sid:string,updater:(s:ClientSessionState)=>ClientSessionState)=>{state=updater(state);$sessionStates.set({'runtime-a':state});return state})}
  const {result}=renderHook(()=>useSubmitPrompt(deps as never))
  await act(async()=>{await expect(result.current('stopped input',{submission_id:'stopped-1',onAccepted:accepted})).rejects.toMatchObject({name:'WorkspaceRequestStopped',requestId:'stopped-1'})})
  expect(accepted).not.toHaveBeenCalled()
  expect(requestGateway).toHaveBeenCalledTimes(1)
  expect(state.messages.some(m=>m.id.startsWith('assistant-error'))).toBe(false)
  expect(state.messages.filter(m=>m.role==='user')).toHaveLength(admitted?1:0)
  expect(window.localStorage.getItem('hermes.desktop.preparedSubmissions.v1')||'').not.toContain('stopped input')
})
