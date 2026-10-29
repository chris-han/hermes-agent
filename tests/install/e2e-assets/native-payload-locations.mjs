import fs from 'node:fs'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { parseArgs } from 'node:util'
import { fileURLToPath } from 'node:url'

/** Verify original relative payload bytes at the actual folded installed root.
 * Generated manifests are checked by native metadata/signature gates instead.
 * This proves byte/location conservation; native imports and journeys remain separate.
 */
export function verifyNativePayload(root, plan, graph) {
  if (!Array.isArray(plan.files) || !plan.files.length || plan.files.length > 200000 ||
      !/^[a-f0-9]{64}$/.test(graph.payloadInventorySha256) || plan.payloadInventorySha256 !== graph.payloadInventorySha256) {
    throw new Error('Native inventory binding missing')
  }
  const inventory = plan.files.map(row => ({ path: row.path, sha256: row.sha256, size: row.size }))
  if (createHash('sha256').update(JSON.stringify(inventory).replace(/[\u007f-\uffff]/g, char => '\\u' + char.charCodeAt(0).toString(16).padStart(4, '0'))).digest('hex') !== graph.payloadInventorySha256) {
    throw new Error('Native inventory digest mismatch')
  }
  const physical = fs.realpathSync(root)
  const buffer = Buffer.alloc(1024 * 1024)
  const seen = new Set()
  const payload = new Set()
  let filesChecked = 0
  let bytesChecked = 0
  for (const row of inventory) {
    if (typeof row.path !== 'string' || !row.path || row.path.includes('\\') || row.path.includes(':') ||
        path.posix.normalize(row.path) !== row.path || path.posix.isAbsolute(row.path) ||
        row.path.split('/').some(part => ['.', '..'].includes(part)) || /[\x00\r\n]/.test(row.path) ||
        !Number.isSafeInteger(row.size) || row.size < 0 || !/^[a-f0-9]{64}$/.test(row.sha256) || seen.has(row.path.toLowerCase())) {
      throw new Error('Native payload inventory invalid')
    }
    seen.add(row.path.toLowerCase())
    if (row.path === 'AppxManifest.xml') { continue }
    payload.add(row.path)
    const candidate = path.join(physical, ...row.path.split('/'))
    if (fs.lstatSync(candidate).isSymbolicLink()) { throw new Error('Native payload link refused') }
    const relative = path.relative(physical, fs.realpathSync(candidate))
    if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
      throw new Error('Native payload escaped installed root')
    }
    const fd = fs.openSync(candidate, 'r')
    try {
      const stat = fs.fstatSync(fd)
      if (!stat.isFile() || stat.size !== row.size) { throw new Error('Native payload size mismatch') }
      const hash = createHash('sha256')
      let count
      while ((count = fs.readSync(fd, buffer)) > 0) { hash.update(buffer.subarray(0, count)) }
      if (hash.digest('hex') !== row.sha256) { throw new Error('Native payload content mismatch') }
    } finally { fs.closeSync(fd) }
    filesChecked++
    bytesChecked += row.size
  }
  if (!plan.assignment || Object.keys(plan.assignment).length !== payload.size ||
      Object.keys(plan.assignment).some(name => !payload.has(name)) || !filesChecked) {
    throw new Error('Native payload assignment incomplete')
  }
  return { producedAt: new Date().toISOString(), filesChecked, bytesChecked,
    payloadInventorySha256: graph.payloadInventorySha256,
    scope: 'all original payload bytes at installed relative paths; usability requires native smoke' }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({ options: Object.fromEntries(['root', 'plan', 'graph', 'out'].map(name => [name, { type: 'string' }])) })
  const read = file => JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, ''))
  const result = verifyNativePayload(values.root, read(values.plan), read(values.graph))
  fs.writeFileSync(values.out, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' })
}
