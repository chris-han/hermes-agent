/** Loopback-only engineering origin. Never mints release qualification or admission. */
import fs from 'node:fs'
import fsp from 'node:fs/promises'
import http from 'node:http'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { parseArgs } from 'node:util'
import { fileURLToPath } from 'node:url'

function byteRange(header, size) {
  if (!header) { return [0, size - 1] }
  const match = /^bytes=(\d*)-(\d*)$/.exec(header)
  if (!match || (!match[1] && !match[2])) { return null }
  const start = match[1] ? Number(match[1]) : Math.max(0, size - Number(match[2]))
  const end = match[1] && match[2] ? Math.min(size - 1, Number(match[2])) : size - 1
  return Number.isSafeInteger(start) && Number.isSafeInteger(end) && start >= 0 && start <= end && start < size ? [start, end] : null
}

export async function startNativePackageOrigin(directory, graph, observe = () => {}, port = 0) {
  if (!Number.isInteger(port) || port < 0 || port > 65535) { throw new Error('Invalid loopback origin port') }
  const files = new Map()
  try {
    if (!Array.isArray(graph.objects) || !graph.objects.length || graph.objects.length > 64) { throw new Error('Bounded native object set required') }
    for (const row of graph.objects) {
      if (!/^releases\/channel-builds\/[a-f0-9]{32}\/[A-Za-z0-9.-]+\.(msix|msixbundle)$/.test(row.key) ||
          !Number.isSafeInteger(row.size) || row.size <= 0 || row.size > 480000000 || !/^[a-f0-9]{64}$/.test(row.sha256) || files.has('/' + row.key)) { throw new Error('Invalid native origin object') }
      const file = path.join(directory, path.posix.basename(row.key))
      if ((await fsp.lstat(file)).isSymbolicLink()) { throw new Error('Native origin link refused') }
      const handle = await fsp.open(file, 'r')
      const entry = { handle, row, file }
      files.set('/' + row.key, entry)
      entry.stat = await handle.stat({ bigint: true })
      if (!entry.stat.isFile() || entry.stat.size !== BigInt(row.size)) { throw new Error('Native origin size mismatch') }
      const hash = createHash('sha256')
      for await (const chunk of handle.createReadStream({ autoClose: false })) { hash.update(chunk) }
      if (hash.digest('hex') !== row.sha256) { throw new Error('Native origin digest mismatch') }
    }
    if (!files.has('/' + graph.root)) { throw new Error('Native root absent') }
  } catch (error) { await Promise.all([...files.values()].map(entry => entry.handle.close())); throw error }
  const server = http.createServer(async (req, res) => {
    let bytesRead = 0
    res.on('close', () => observe({ producedAt: new Date().toISOString(), method: req.method, path: req.url,
      range: req.headers.range ?? null, userAgent: req.headers['user-agent'] ?? null, remoteAddress: req.socket.remoteAddress,
      status: res.statusCode, bytesRead, completed: res.writableFinished,
      scope: 'origin stream bytes; not network or differential acceptance' }))
    const entry = files.get(req.url)
    if (!entry || !['GET', 'HEAD'].includes(req.method)) { res.writeHead(404).end(); return }
    try {
      const stat = await entry.handle.stat({ bigint: true })
      if (['dev', 'ino', 'size', 'mtimeNs', 'ctimeNs'].some(key => stat[key] !== entry.stat[key])) { res.writeHead(409).end(); return }
      const etag = '"' + entry.row.sha256 + '"'
      const requestedRange = req.headers['if-range'] && req.headers['if-range'] !== etag ? null : req.headers.range
      const range = byteRange(requestedRange, entry.row.size)
      if (!range) { res.writeHead(416, { 'Content-Range': `bytes */${entry.row.size}` }).end(); return }
      const [start, end] = range
      const partial = start !== 0 || end !== entry.row.size - 1 || Boolean(requestedRange)
      const headers = { 'Content-Length': end - start + 1, 'Content-Type': 'application/octet-stream',
        'Accept-Ranges': 'bytes', 'Cache-Control': 'public, max-age=31536000, immutable', ETag: etag }
      if (partial) { headers['Content-Range'] = `bytes ${start}-${end}/${entry.row.size}` }
      res.writeHead(partial ? 206 : 200, headers)
      if (req.method === 'HEAD') { res.end(); return }
      const requestFile = await fsp.open(entry.file, 'r')
      const requestStat = await requestFile.stat({ bigint: true })
      if (['dev', 'ino', 'size', 'mtimeNs', 'ctimeNs'].some(key => requestStat[key] !== entry.stat[key])) { await requestFile.close(); res.destroy(new Error('Native object changed')); return }
      const stream = requestFile.createReadStream({ autoClose: true, start, end })
      stream.on('data', chunk => { bytesRead += chunk.length })
      stream.on('error', error => res.destroy(error))
      res.on('close', () => stream.destroy())
      stream.pipe(res)
    } catch (error) { res.destroy(error) }
  })
  try { await new Promise((resolve, reject) => { server.once('error', reject); server.listen(port, '127.0.0.1', resolve) }) }
  catch (error) { await Promise.all([...files.values()].map(entry => entry.handle.close())); throw error }
  return { base: `http://127.0.0.1:${server.address().port}`, close: async () => {
    await new Promise(resolve => { server.close(resolve); server.closeAllConnections() })
    await Promise.all([...files.values()].map(entry => entry.handle.close()))
  } }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({ options: Object.fromEntries(['directory', 'graph', 'graph-sha256', 'ready', 'log', 'port'].map(name => [name, { type: 'string' }])) })
  const bytes = await fsp.readFile(values.graph)
  if (createHash('sha256').update(bytes).digest('hex') !== values['graph-sha256']) { throw new Error('Graph file digest mismatch') }
  const log = await fsp.open(values.log, 'wx')
  const origin = await startNativePackageOrigin(values.directory, JSON.parse(bytes.toString('utf8').replace(/^\uFEFF/, '')), row => fs.writeSync(log.fd, JSON.stringify(row) + '\n'), values.port ? Number(values.port) : 0)
  await fsp.writeFile(values.ready, JSON.stringify({ producedAt: new Date().toISOString(), base: origin.base,
    pid: process.pid, graphSha256: values['graph-sha256'], scope: 'loopback-only native engineering origin; no release admission' }) + '\n', { flag: 'wx' })
  for (const signal of ['SIGINT', 'SIGTERM']) { process.once(signal, async () => { await origin.close(); await log.close() }) }
}
