import { runPython } from '../../../scripts/build/python.mjs'
import fs from 'node:fs'
import path from 'node:path'

/** @param {string} payload */
export function rehashPayloadDigests(payload) {
  // Light builds have no payload. Present payloads require valid tool facts.
  if (!fs.existsSync(path.join(payload, 'manifest.json'))) return
  console.log(`[after-pack] refreshing payload hashes after signing: ${payload}`)
  const started = performance.now()
  let result = 'FAIL'
  try {
    runPython([
      path.resolve(import.meta.dirname, '../../../scripts/bundles/payload.py'), 'rehash', payload],
    { stdio: 'inherit' })
    result = 'PASS'
  } finally {
    const cache = process.env.HERMES_DESKTOP_SERIAL_CACHE
    if (cache && fs.existsSync(path.resolve(cache, '../..') + '.distribution.lock')) {
      const directory = path.join(cache, 'diagnostics')
      fs.mkdirSync(directory, {recursive:true})
      fs.appendFileSync(path.join(directory, 'runtime.jsonl'), JSON.stringify({
        producedAt:new Date().toISOString(), stage:'payload.hash-refresh', result,
        elapsedSeconds:(performance.now()-started)/1000,
      })+'\n')
    }
  }
  console.log('[after-pack] payload hash refresh complete')
}
