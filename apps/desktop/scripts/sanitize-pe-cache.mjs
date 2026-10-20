/** Content-bound PE admission; mutable files are hashed before any cache hit. */
import fs from 'node:fs'
import path from 'node:path'
import {createHash, randomUUID} from 'node:crypto'
import {fileURLToPath} from 'node:url'
import {walkFiles, readSecurityDirectory, classifySecurityDirectory, clearSecurityDirectory} from './sanitize-pe-signatures.mjs'

const hash = bytes => createHash('sha256').update(bytes).digest('hex')
const rules = hash(Buffer.concat(['sanitize-pe-signatures.mjs', 'sanitize-pe-cache.mjs'].map(name =>
  fs.readFileSync(fileURLToPath(new URL(name, import.meta.url))))))

async function digest(file) {
  const h = createHash('sha256')
  for await (const chunk of fs.createReadStream(file, {highWaterMark: 128 * 1024})) h.update(chunk)
  return h.digest('hex')
}

export async function sanitizeTreeCached(rootDir, receipt) {
  const started = performance.now()
  const directory = path.dirname(receipt)
  fs.mkdirSync(directory, {recursive:true})
  const canonical = fs.realpathSync(directory), requested = path.resolve(directory)
  if ((process.platform === 'win32' ? canonical.toLowerCase() !== requested.toLowerCase() : canonical !== requested)
      || (fs.existsSync(receipt) && fs.lstatSync(receipt).isSymbolicLink())) throw Error('Linked PE cache path')
  let entries = {}, reason = 'missing-receipt'
  try {
    const data = JSON.parse(fs.readFileSync(receipt, 'utf8'))
    if (data.schema === 1 && data.rules === rules && data.entries && !Array.isArray(data.entries)
        && Object.entries(data.entries).every(([k,v])=>/^[a-f0-9]{64}$/.test(k) && typeof v === 'boolean')
        && hash(JSON.stringify(data.entries)) === data.entriesSha256) {
      entries = data.entries; reason = 'verified-content-and-rules'
    } else reason = 'receipt-or-rules-changed'
  } catch(error) {
    if(error.code !== 'ENOENT') reason = 'unreadable-receipt'
  }
  const admitted = {}, repaired = []
  let scanned=0, visited=0, hits=0, misses=0
  const files = walkFiles(rootDir)
  let nextProgress = performance.now()+5000
  // Bounded streams avoid retaining large browser/native binaries in memory.
  await Promise.all(Array.from({length:8}, async()=>{
    for(const file of files) {
      visited++
      const before = await digest(file) // unreadable inputs fail; never falsely admitted
      if(Object.hasOwn(entries,before)) {
        hits++; scanned += Number(entries[before]); admitted[before] = entries[before]
      } else {
        misses++
        const entry = readSecurityDirectory(file)
        const pe = Boolean(entry)
        scanned += Number(pe)
        if(entry && classifySecurityDirectory(entry)==='dangling') {
          clearSecurityDirectory(file,entry.offsetInFile)
          repaired.push(path.relative(rootDir,file))
          admitted[await digest(file)] = pe
        } else admitted[before] = pe
      }
      if(performance.now() >= nextProgress) {
        console.log(`[sanitize-pe-cache] ${visited} files; ${hits} hits, ${misses} rechecks`)
        nextProgress=performance.now()+5000
      }
    }
  }))
  const data = {schema:1,rules,entries:admitted,entriesSha256:hash(JSON.stringify(admitted))}
  const temporary=receipt+'.'+randomUUID()+'.tmp'
  try {fs.writeFileSync(temporary,JSON.stringify(data));fs.renameSync(temporary,receipt)}
  finally {fs.rmSync(temporary,{force:true})}
  const result = {scanned,repaired,visited,hits,misses,reason,elapsedSeconds:(performance.now()-started)/1000}
  fs.appendFileSync(path.join(directory,'diagnostics.jsonl'),JSON.stringify({producedAt:new Date().toISOString(),...result,repaired:repaired.length})+'\n')
  console.log(`[sanitize-pe-cache] ${hits} verified hits; ${misses} content/rule rechecks; ${result.elapsedSeconds.toFixed(2)}s`)
  return result
}
