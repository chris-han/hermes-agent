// Time-to-first-token — Enter → first assistant token painted. The latency an
// agent app is uniquely judged on, spanning the desktop submit path AND the
// backend/agent-loop first-token time. Backend tier: fires a REAL prompt, needs
// a live backend (and credits). Report-only.
//
//   node scripts/perf/run.mjs first-token --spawn --prompt "hi"

import { SELECTORS, sleep, typeIntoComposer } from '../lib/cdp.mjs'
import { summarize } from '../lib/stats.mjs'

export default {
  name: 'first-token',
  tier: 'backend',
  description: 'Enter → first assistant token painted (real backend).',
  async run(cdp, opts = {}) {
    const rounds = Number(opts.rounds ?? 3)
    const prompt = opts.prompt ?? 'reply with a single short sentence'
    const timeoutMs = Number(opts.timeoutMs ?? 60000)

    await cdp.send('Runtime.enable')

    const firstTokens = []

    for (let i = 0; i < rounds; i++) {
      const baseText = await cdp.eval(`(() => {
        const a = document.querySelectorAll(${JSON.stringify(SELECTORS.assistantMessage)})
        return a.length ? a[a.length - 1].textContent.length : 0
      })()`)
      const baseCount = await cdp.eval(`document.querySelectorAll(${JSON.stringify(SELECTORS.assistantMessage)}).length`)

      await typeIntoComposer(cdp, `${prompt} (${i})`, { cps: 60 })
      const submitAt = Date.now()
      await cdp.eval(`(() => {
        const el = document.querySelector(${JSON.stringify(SELECTORS.composer)})
        el && el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true, cancelable: true }))
      })()`)

      const deadline = Date.now() + timeoutMs
      let firstTokenMs = null

      while (Date.now() < deadline) {
        await sleep(25)
        const grown = await cdp.eval(`(() => {
          const a = document.querySelectorAll(${JSON.stringify(SELECTORS.assistantMessage)})
          if (a.length > ${baseCount}) return true
          return a.length ? a[a.length - 1].textContent.length > ${baseText} : false
        })()`)

        if (grown) {
          firstTokenMs = Date.now() - submitAt
          break
        }
      }

      if (firstTokenMs !== null) {
        firstTokens.push(firstTokenMs)
      }

      // Let the turn finish before the next round.
      const turnDeadline = Date.now() + timeoutMs
      while (Date.now() < turnDeadline) {
        await sleep(250)
        const busy = await cdp.eval(`!!document.querySelector('[data-status="running"], [data-busy="true"]')`)

        if (!busy) {
          break
        }
      }

      await sleep(500)
    }

    if (!firstTokens.length) {
      throw new Error('no first token observed — is a backend with credits connected?')
    }

    const s = summarize(firstTokens)

    return {
      metrics: { first_token_p50_ms: s.p50, first_token_p95_ms: s.p95 },
      detail: { rounds, samples: firstTokens, summary: s }
    }
  }
}

// The scripted Windows lane uses the same renderer CDP client and scenario.
// A frame capture brackets the visible result. The pre-paint observation and
// capture receipt are both retained so the DOM timestamp cannot pass as paint.
export async function runScriptedFirstToken(cdp, provider, {
  nonce, writeSample, timeoutMs = 30000, trace = false
}) {
  await cdp.send('Network.enable')
  await cdp.send('Performance.enable')
  const samples = []
  const frames = []
  let compositorEvents = []
  let resolveTrace
  cdp.on('Tracing.tracingComplete', result => resolveTrace?.(result))
  cdp.on('Tracing.dataCollected', ({ value }) => {
    compositorEvents.push(...value.filter(event => event.name === 'Screenshot'))
  })

  async function chromiumClock() {
    const beforeNs = process.hrtime.bigint().toString()
    const response = await cdp.send('Performance.getMetrics')
    const afterNs = process.hrtime.bigint().toString()
    const metrics = response.metrics.filter(({ name }) => name === 'Timestamp' || name === 'NavigationStart')
    return { beforeNs, afterNs, metrics }
  }

  for (const [direction, event] of [['out', 'Sent'], ['in', 'Received']]) {
    cdp.on('Network.webSocketFrame' + event, ({ requestId, timestamp, response }) => {
      try {
        const message = JSON.parse(response.payloadData)
        if (message.id || message.params?.type?.startsWith('message.')) {
          frames.push({ direction, requestId, timestamp, message })
        }
      } catch {
        // A non-JSON frame carries no turn evidence.
      }
    })
  }

  for (let index = 0; index < 12; index++) {
    const marker = `U${index}-${nonce}`
    const expected = `Windows answer ${index} ${nonce}`
    const sample = { index, marker, expected, warmup: index < 2, startedAt: new Date().toISOString() }
    const frameStart = frames.length
    provider.script(marker, [{
      reasoning: ['Thinking about the request.'],
      text: ['', expected],
      firstTextDelayMs: 200
    }])

    let tracingComplete
    try {
      const configured = await cdp.eval(`(() => {
        const box = document.querySelector(${JSON.stringify(SELECTORS.composer)})
        if (!box) return false
        box.focus()
        if (document.activeElement !== box && !box.contains(document.activeElement)) return false

        const expected = ${JSON.stringify(expected)}
        const state = window.__ttfvcTurn = {
          enter: null, dom: null, bounds: null,
          viewport: { width: innerWidth, height: innerHeight }
        }
        document.addEventListener('keydown', event => {
          if (event.isTrusted && event.key === 'Enter' && box.contains(event.target)) state.enter = performance.now()
        }, { capture: true, once: true })

        const started = performance.now()
        const inspect = () => {
          if (window.__ttfvcTurn !== state || performance.now() - started > ${timeoutMs}) return
          for (const bubble of document.querySelectorAll(${JSON.stringify(SELECTORS.assistantMessage)})) {
            const walker = document.createTreeWalker(bubble, NodeFilter.SHOW_TEXT)
            while (walker.nextNode()) {
              const node = walker.currentNode
              if (node.textContent !== expected || !node.parentElement.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) continue
              const range = document.createRange()
              range.selectNodeContents(node)
              const { x, y, width, height } = range.getBoundingClientRect()
              const hit = document.elementFromPoint(x + width / 2, y + height / 2)
              if (state.enter === null || width <= 0 || height <= 0 || !bubble.contains(hit)) continue
              state.bounds = { x, y, width, height }
              state.dom = performance.now()
              return
            }
          }
          requestAnimationFrame(inspect)
        }
        requestAnimationFrame(inspect)
        return true
      })()`)
      if (!configured) throw new Error('Composer unavailable')
      await cdp.send('Input.insertText', { text: marker + ' Reply with the scripted answer.' })
      if (trace) {
        compositorEvents = []
        tracingComplete = new Promise(resolve => { resolveTrace = resolve })
        try {
          await cdp.send('Tracing.start', {
            categories: 'disabled-by-default-devtools.screenshot',
            transferMode: 'ReportEvents'
          })
        } catch (error) {
          tracingComplete = undefined
          throw error
        }
      }
      sample.chromiumStart = await chromiumClock()
      await cdp.send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13 })
      await cdp.send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13 })
      const deadline = performance.now() + timeoutMs
      let screenshot
      while (performance.now() < deadline) {
        const dom = await cdp.eval('window.__ttfvcTurn')
        if (dom?.dom !== null && dom?.dom !== undefined && !screenshot) {
          const captureRequestedNs = process.hrtime.bigint().toString()
          screenshot = (await cdp.send('Page.captureScreenshot', { format: 'png', fromSurface: true })).data
          sample.surfaceReceipt = { captureRequestedNs, captureReceivedNs: process.hrtime.bigint().toString(), ...dom }
          await writeSample(index, 'png', Buffer.from(screenshot, 'base64'))
        }

        const complete = frames.slice(frameStart).find(frame => frame.direction === 'in' &&
          frame.message.params?.type === 'message.complete' && frame.message.params.payload?.text === expected)
        if (screenshot && complete) {
          sample.sessionId = complete.message.params.session_id
          sample.status = 'CANDIDATE'
          break
        }

        await sleep(20)
      }

      sample.chromiumEnd = await chromiumClock()
      sample.status ??= 'TIMEOUT'
    } catch (error) {
      sample.status = 'FAILED'
      sample.error = error.message
    } finally {
      if (tracingComplete) {
        await cdp.send('Tracing.end')
        const result = await tracingComplete
        await writeSample(index, 'trace.json', JSON.stringify({ result, events: compositorEvents }))
      }
    }

    sample.wire = frames.slice(frameStart).filter(frame => {
      const { message } = frame
      return message.params?.session_id === sample.sessionId && sample.sessionId ||
        message.method === 'prompt.submit' && JSON.stringify(message.params).includes(marker)
    })
    sample.provider = provider.completions.filter(record => record.marker === marker).map(({ body, ...receipt }) => receipt)
    sample.finishedAt = new Date().toISOString()
    samples.push(sample)
    await writeSample(index, 'json', JSON.stringify(sample, null, 2))

    // A failed turn may still be running. Reuse the session only after a matched completion.
    if (sample.status !== 'CANDIDATE') break
  }

  return samples
}
