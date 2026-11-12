/** Only typed, public Cloud diagnoses cross into the sign-in UI. Raw transport
 * prose can contain endpoints or authorization material and is never displayed. */
const CODES = new Set([
  'cloud_request_timeout',
  'cloud_metadata_timeout',
  'cloud_metadata_unavailable',
  'cloud_model_access_missing',
  'cloud_provisioning_pending',
  'cloud_entitlement_restricted',
  'cloud_capability_mismatch',

  'cloud_catalog_timeout',
  'cloud_catalog_empty',
  'cloud_catalog_unavailable',
  'cloud_network_unavailable',
  'cloud_dns_failed',
  'cloud_tls_failed',
  'cloud_http3_unavailable',
  'cloud_config_invalid',
  'cloud_response_invalid',
  'cloud_profile_mismatch',
  'cloud_profile_changed',
  'cloud_browser_owner_changed',
  'cloud_profile_unavailable',
  'cloud_storage_invalid',
  'cloud_storage_unavailable',
  'cloud_credential_expired',
  'cloud_authorization_required',
  'cloud_authorization_denied',
  'cloud_server_conflict',
  'cloud_session_unavailable',
  'cloud_session_expired',
  'cloud_session_cancelled',
  'cloud_rate_limited',
  'cloud_server_error',
  'cloud_protocol_unsupported',
  'cloud_origin_redirected',
  'cloud_internal_error'
])

export function decodeCloudSignInError(raw: unknown): {
  code: string
  profile: null | string
  httpStatus?: number
  diagnostics?: string
} {
  let payload: unknown = raw
  let httpStatus: number | undefined

  if (raw instanceof Error || typeof raw === 'string') {
    const message = raw instanceof Error ? raw.message : raw
    const start = message.indexOf('{')
    const match = message.slice(0, start < 0 ? undefined : start).match(/\b([45]\d{2}):/)
    httpStatus = match ? Number(match[1]) : undefined

    try {
      payload = start >= 0 ? JSON.parse(message.slice(start)) : { status: message }
    } catch {
      payload = {}
    }
  }

  const envelope = payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {}

  const detail =
    envelope.detail && typeof envelope.detail === 'object' ? (envelope.detail as Record<string, unknown>) : envelope

  const terminal: Record<string, string> = {
    expired: 'cloud_session_expired',
    denied: 'cloud_authorization_denied',
    revoked: 'cloud_authorization_required',
    cancelled: 'cloud_session_cancelled'
  }

  const nativeScope = envelope.detail === 'profile_scope_mismatch' ? 'cloud_profile_mismatch' : undefined
  const candidate = detail.code ?? detail.error_code ?? nativeScope ?? terminal[String(detail.status ?? '')]

  const profile =
    typeof detail.profile === 'string' && /^[a-zA-Z0-9_-]{1,64}$/.test(detail.profile) ? detail.profile : null

  const status = detail.http_status

  if (typeof status === 'number' && Number.isInteger(status) && status >= 300 && status <= 599) {
    httpStatus = status
  }

  if (detail.http_status === null && ['transport', 'client'].includes(String(detail.error_source))) {
    httpStatus = undefined
  }

  const diagnostics = [
    ['transport', 'server', 'client'].includes(String(detail.error_source)) ? detail.error_source : '',
    ['http3', 'http'].includes(String(detail.transport)) ? detail.transport : '',
    ['start', 'poll', 'claim', 'ack', 'cancel', 'readiness', 'catalog'].includes(String(detail.operation))
      ? detail.operation
      : '',
    typeof detail.transport_code === 'number' &&
    Number.isInteger(detail.transport_code) &&
    detail.transport_code >= 1 &&
    detail.transport_code <= 999
      ? `curl ${detail.transport_code}`
      : ''
  ]
    .filter(Boolean)
    .join(' · ')

  return {
    ...(diagnostics ? { diagnostics } : {}),
    code: typeof candidate === 'string' && CODES.has(candidate) ? candidate : 'cloud_unknown_failure',
    profile,
    httpStatus
  }
}
