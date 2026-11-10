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

export function scoreTrial(x: any) {
  if (!x.requestReached || !x.inferenceCalls) { return { validity: 'INVALID_TEST', reason: 'NO_MODEL_REQUEST', routing: null, workflow: null } }

  if (x.fixtureError || x.consentBlocked || x.authorizedToolUnavailable) { return { validity: 'INVALID_TEST', reason: x.consentBlocked ? 'CONSENT_BLOCKED' : x.authorizedToolUnavailable ? 'NO_AUTHORIZED_TOOL' : 'FIXTURE_ERROR', routing: null, workflow: null } }
  const operations = browserOperations(x.nativeCalls ?? x.calls)
  const first = operations[0]
  const routing = x.target === 'independent' ? first?.name === 'browser_exec' : ['desktop_preview', 'drive_preview'].includes(first?.name)
  const owner = x.ownerCorrect === true
  const workflow = routing && owner && x.operationSuccess === true && x.effect === true && !x.falseSuccess && !x.unexpectedSwitch

  return { validity: 'VALID', reason: !routing ? 'WRONG_BROWSER_OR_NO_BROWSER_ACTION' : !owner ? 'WRONG_OWNER' : !x.operationSuccess ? 'OPERATION_FAILED' : !x.effect ? 'NO_VERIFIED_EFFECT' : x.falseSuccess ? 'FALSE_SUCCESS' : x.unexpectedSwitch ? 'UNEXPECTED_SWITCH' : 'VERIFIED', routing, workflow }
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
