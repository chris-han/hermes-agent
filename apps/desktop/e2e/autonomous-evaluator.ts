/** Evaluation only: discovery is access, observable effects decide completion. */
export function unwrapTools(calls: any[]): any[] {
  return calls.flatMap(call => {
    const fn = call.function ?? call
    let args: any = fn.arguments ?? {}

    if (typeof args === 'string') { try { args = JSON.parse(args) } catch { return [{ name: fn.name, arguments: {}, invalid_json: true }] } }

    if (fn.name === 'tool_call') {
      let entries = args.calls

      if (typeof entries === 'string') { try { entries = JSON.parse(entries) } catch { return [{name:fn.name,arguments:args,invalid_json:true}] } }
      entries = Array.isArray(entries) ? entries : [args]

      return unwrapTools(entries).map(item => ({ ...item, wrapper: 'tool_call' }))
    }

    return [{ name: fn.name, arguments: args }]
  })
}

export const browserOperations = (calls: any[]) => unwrapTools(calls).filter(t => ['desktop_preview', 'drive_preview', 'browser_exec'].includes(t.name))

export function accessRoute(names: string[], catalog: string) {
  if (['desktop_preview', 'drive_preview'].every(n => names.includes(n))) { return 'DIRECT' }

  if (['tool_search', 'tool_describe', 'tool_call'].every(n => names.includes(n)) && ['desktop_preview', 'drive_preview'].every(n => catalog.includes(n))) { return 'DEFERRED' }

  return 'UNPROVEN'
}

/** Renderer observations, including native preflight calls and code-owned final replies. */
export function turnEvidence(messages: any[], prompt: string) {
  const text = (message: any) => (message.parts ?? []).filter((part: any) => part.type === 'text').map((part: any) => part.text ?? '').join('\n')
  const start = messages.findLastIndex(message => message.role === 'user' && text(message).trim() === prompt.trim())

  if (start < 0) { return { requestReached: false, calls: [], finalResponse: '', completed: false } }
  const assistant = messages.slice(start + 1).filter(message => message.role === 'assistant')

  const calls = assistant.flatMap(message => (message.parts ?? []).filter((part: any) => part.type === 'tool-call').map((part: any) => ({
    id: part.toolCallId, function: { name: part.toolName, arguments: part.args ?? {} },
    result: part.result, isError: part.isError, completedAt: part.completedAt
  })))

  const last = assistant.at(-1)

  return { requestReached: true, calls, finalResponse: assistant.map(text).filter(Boolean).join('\n'), completed: !!(last?.durableComplete || last?.persistedTurn?.complete) }
}

function workflowReason(x: any, routing: boolean, owner: boolean) {
  return !routing ? 'WRONG_BROWSER_OR_NO_BROWSER_ACTION' : !owner ? 'WRONG_OWNER' : !x.operationSuccess ? 'OPERATION_FAILED' : !x.effect ? 'NO_VERIFIED_EFFECT' : x.falseSuccess ? 'FALSE_SUCCESS' : x.unexpectedSwitch ? 'UNEXPECTED_SWITCH' : 'VERIFIED'
}

export function scoreTrial(x: any) {
  if (!x.requestReached || !x.inferenceCalls) { return { validity: 'INVALID_TEST', reason: 'NO_MODEL_REQUEST', routing: null, workflow: null } }

  if (x.fixtureError || x.consentBlocked || x.authorizedToolUnavailable) { return { validity: 'INVALID_TEST', reason: x.consentBlocked ? 'CONSENT_BLOCKED' : x.authorizedToolUnavailable ? 'NO_AUTHORIZED_TOOL' : 'FIXTURE_ERROR', routing: null, workflow: null } }
  const operations = browserOperations(x.nativeCalls ?? x.calls)
  const first = operations[0]
  const routing = x.target === 'independent' ? first?.name === 'browser_exec' : ['desktop_preview', 'drive_preview'].includes(first?.name)
  const owner = x.ownerCorrect === true
  const workflow = routing && owner && x.operationSuccess === true && x.effect === true && !x.falseSuccess && !x.unexpectedSwitch

  return { validity: 'VALID', reason: workflowReason(x, routing, owner), routing, workflow }
}

/** Use supported buttons; never dismiss an unknown authorization dialog. */
export async function declineProfileConsent(page: any, timeout = 30000) {
  const start = Date.now()
  let stableSince = 0
  let dismissed = 0

  while (Date.now() - start < timeout) {
    const button = page.getByRole('button', { name: /^(Not now|暂不|暫不)$/ }).first()

    if (await button.isVisible().catch(() => false)) {
      await button.click()
      await button.waitFor({ state: 'hidden', timeout: 10000 })
      dismissed++
      stableSince = 0
    }

    const composer = page.locator('[data-slot="composer-root"]').locator('[contenteditable="true"]:visible, textarea:visible').first()
    const ready = await composer.click({ trial: true, timeout: 250 }).then(() => true).catch(() => false)

    if (ready) {
      stableSince ||= Date.now()

      if (Date.now() - stableSince >= 2000) { return { dismissed, composer_interactable: true } }
    } else { stableSince = 0 }

    await new Promise(resolve => setTimeout(resolve, 100))
  }

  throw new Error('Consent/composer readiness did not settle through supported UI')
}


export function trialFailure(state: any, calls: any[]) {
  const modelFailure = state.failures?.find((failure: any) => failure.type === 'forbidden_action_attempt')

  if (modelFailure) { return { kind: 'MODEL_FAILURE', detail: modelFailure } }

  if (state.failures?.length || calls.some(call => call.transport_error || call.http_status >= 400)) {
    return { kind: 'INFRASTRUCTURE', failures: state.failures, calls: calls.map(call => ({ index: call.index, http_status: call.http_status, error: call.transport_error })) }
  }

  return null
}


/** Decode canonical JSON/untrusted envelopes without interpreting retrieved prose. */
export function toolResultPayload(raw: any): any | null {
  let value = raw

  if (typeof value === 'string') {
    let candidate = value.trim()

    if (candidate.startsWith('<untrusted_tool_result')) {
      const start = candidate.indexOf('\n{')
      const end = candidate.lastIndexOf('\n</untrusted_tool_result>')

      if (start < 0 || end < start) { return null }
      candidate = candidate.slice(start + 1, end).trim()
    }

    try { value = JSON.parse(candidate) } catch {
      return null
    }
  }

  if (!value || typeof value !== 'object') { return null }

  return value
}

/** Execution success remains separate from access, schema validity and effects. */
export function toolExecutionStatus(receipt: any): 'PASS' | 'FAIL' | 'UNPROVEN' {
  if (!receipt) { return 'UNPROVEN' }
  if (receipt.isError) { return 'FAIL' }
  const value = toolResultPayload(receipt.result)
  if (!value) { return typeof receipt.result === 'string' && /^Tool '[^']+' does not exist/.test(receipt.result.trim()) ? 'FAIL' : 'UNPROVEN' }
  if (value.success === false || value.error || (typeof value.exit_code === 'number' && value.exit_code !== 0)) { return 'FAIL' }

  return value.success === true || value.exit_code === 0 ? 'PASS' : 'UNPROVEN'
}


/** Offline evaluation of fixture evidence only; never routes tools or supplies facts to the Agent. */
function hasCompleteClaims(claims: any[], metrics: unknown[]) {
  return claims.length === metrics.length && new Set(claims.map(claim => claim.metric)).size === metrics.length
}

function hasSourceQuote(row: any) {
  return typeof row.quote === 'string' && !!row.quote.trim()
}

export function assessComparisonEvidence(reference: any[], observations: any[], claims: any[], unitFactors: Record<string, Record<string, number>>) {
  const verdict = (status: string, reason: string) => ({ status, reason })
  const key = (row: any) => JSON.stringify([row.source, row.metric])

  const normalized = (row: any) => {
    const factor = unitFactors[row.metric]?.[row.unit]

    return typeof row.value === 'number' && Number.isFinite(row.value) && typeof factor === 'number' && Number.isFinite(factor)
      ? row.value * factor : null
  }

  const same = (a: number, b: number) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b))
  const expected = new Map<string, any>()

  if (!reference.length) {return verdict('UNVERIFIED', 'NO_SOURCE_REFERENCE')}

  for (const row of reference) {
    const value = normalized(row)

    if (value === null) {return verdict('UNVERIFIED', 'INCOMPARABLE_UNITS')}
    const previous = expected.get(key(row))

    if (previous && !same(normalized(previous)!, value)) {return verdict('UNVERIFIED', 'CONTRADICTORY_SOURCE')}
    expected.set(key(row), row)
  }

  const seen = new Set<string>()

  for (const row of observations) {
    const original = expected.get(key(row))

    if (!original) {return verdict('UNSUPPORTED', 'WRONG_SOURCE_OR_METRIC')}

    if (row.revision !== original.revision) {return verdict('UNVERIFIED', 'STALE_OBSERVATION')}

    if (!hasSourceQuote(original)) {return verdict('UNVERIFIED', 'NO_SOURCE_QUOTE')}

    if (typeof row.quote !== 'string' || !row.quote.includes(original.quote)) {return verdict('UNSUPPORTED', 'UNSUPPORTED_SOURCE_QUOTE')}
    const value = normalized(row)

    if (value === null) {return verdict('UNVERIFIED', 'INCOMPARABLE_UNITS')}

    if (!same(value, normalized(original)!)) {return verdict('UNSUPPORTED', 'UNSUPPORTED_VALUE')}
    seen.add(key(row))
  }

  if ([...expected.keys()].some(id => !seen.has(id))) {return verdict('UNVERIFIED', 'MISSING_SOURCE_EVIDENCE')}
  const metrics = [...new Set(reference.map(row => row.metric))]

  if (!hasCompleteClaims(claims, metrics)) {return verdict('UNVERIFIED', 'MISSING_COMPARISON_CLAIM')}

  for (const claim of claims) {
    const rows = [...expected.values()].filter(row => row.metric === claim.metric)

    if (!rows.length) {return verdict('UNSUPPORTED', 'UNSUPPORTED_CONCLUSION')}
    const low = Math.min(...rows.map(row => normalized(row)!))
    const high = Math.max(...rows.map(row => normalized(row)!))
    const sources = (value: number) => rows.filter(row => same(normalized(row)!, value)).map(row => row.source).sort()

    if (JSON.stringify(claim.lowest?.slice().sort()) !== JSON.stringify(sources(low)) || JSON.stringify(claim.highest?.slice().sort()) !== JSON.stringify(sources(high))) {return verdict('UNSUPPORTED', 'UNSUPPORTED_CONCLUSION')}
  }

  return verdict('SUPPORTED', 'SOURCE_VALUES_AND_COMPARISONS_VERIFIED')
}
