import { describe, expect, it } from 'vitest'

import { decodeCloudSignInError } from './cloud-sign-in-error'

describe('Cloud sign-in recovery boundary', () => {
  it('decodes the real Electron error envelope without retaining raw prose', () => {
    const raw = new Error(
      'Error invoking remote method \'hermes:api\': Error: 503: {"detail":{"code":"cloud_dns_failed","message":"PRIVATE ENDPOINT","profile":"default"}}'
    )

    expect(decodeCloudSignInError(raw)).toEqual({ code: 'cloud_dns_failed', profile: 'default', httpStatus: 503 })
    expect(JSON.stringify(decodeCloudSignInError(raw))).not.toContain('PRIVATE')
  })

  it('retains native profile authorization refusal without treating it as a server conflict', () => {
    const raw = new Error('403: {"detail":"profile_scope_mismatch"}')
    expect(decodeCloudSignInError(raw)).toEqual({ code: 'cloud_profile_mismatch', profile: null, httpStatus: 403 })
  })

  it('keeps a genuine server conflict distinct from a profile mismatch', () => {
    expect(decodeCloudSignInError({ detail: { code: 'cloud_server_conflict', profile: 'work' } }).code).toBe(
      'cloud_server_conflict'
    )
    expect(
      decodeCloudSignInError({ error_code: 'cloud_profile_mismatch', profile: 'work', http_status: 409 }).code
    ).toBe('cloud_profile_mismatch')
  })

  it('recognizes expired native receipts and rejects unsafe profile labels', () => {
    expect(decodeCloudSignInError({ status: 'expired' }).code).toBe('cloud_session_expired')
    expect(decodeCloudSignInError({ code: 'cloud_profile_mismatch', profile: 'C:\\Users\\private' }).profile).toBeNull()
  })
})

it('does not present a local gateway 503 as a received Cloud response', () => {
  expect(
    decodeCloudSignInError(
      new Error(
        '503: {"detail":{"code":"cloud_http3_unavailable","http_status":null,"error_source":"transport","transport":"http3","transport_code":7,"operation":"start"}}'
      )
    )
  ).toEqual({
    code: 'cloud_http3_unavailable',
    profile: null,
    httpStatus: undefined,
    diagnostics: 'transport · http3 · start · curl 7'
  })
})
