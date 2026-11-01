// Transaction evidence only: callers must revalidate OS ownership before mutations.
import { open, readFile, rename, unlink } from 'node:fs/promises'
import { randomUUID } from 'node:crypto'

const fields = {
  baseline_registered: ['packageFullName','family','version','installPath','exe'],
  baseline_observed: ['pid','birth','dataDigest'],
  update_started: ['targetSha256','targetVersion'],
  successor_verified: ['packageFullName','family','version','installPath','exe','pid','birth'],
  conservation_verified: ['dataDigest','receiptSha256'],
  cleanup_complete: ['receiptSha256']
}
const order = ['created', ...Object.keys(fields)]
function required(object, names) {
  if (!object || typeof object !== 'object' || names.some(k => !(k in object) || object[k] === '' || object[k] == null)) throw new Error('Missing observed transaction facts')
}
function valid(j) {
  if (!j || j.schema !== 1 || !Array.isArray(j.events) || !j.events.length || j.events[0].phase !== 'created' || j.phase !== j.events.at(-1).phase) throw new Error('Invalid transaction journal')
  for (let i=0;i<j.events.length;i++) {
    if (j.events[i].phase !== order[i]) throw new Error('Invalid transition sequence')
    if (i) required(j.events[i].facts,fields[order[i]])
  }
  required(j.owner,['sid','sessionId','root','baselineSha256'])
  return j
}
export async function readJournal(file) { return valid(JSON.parse(await readFile(file,'utf8'))) }
export async function createJournal(file,owner) {
  required(owner,['sid','sessionId','root','baselineSha256'])
  const j={schema:1,owner,phase:'created',events:[{phase:'created',at:new Date().toISOString()}]}
  const handle=await open(file,'wx',0o600)
  try { await handle.writeFile(JSON.stringify(j,null,2)+'\n'); await handle.sync() } finally { await handle.close() }
  return j
}
export async function advanceJournal(file,phase,facts) {
  // No stale-lock stealing: interruption leaves a hold for explicit inspection.
  const lockPath=file+'.lock'; const lock=await open(lockPath,'wx',0o600)
  try {
    const j=await readJournal(file)
    if (order.indexOf(phase)!==order.indexOf(j.phase)+1 || !fields[phase]) throw new Error('Invalid transition')
    required(facts,fields[phase]); j.events.push({phase,at:new Date().toISOString(),facts}); j.phase=phase
    const temp=file+'.'+randomUUID()+'.tmp'; const handle=await open(temp,'wx',0o600)
    try { await handle.writeFile(JSON.stringify(j,null,2)+'\n'); await handle.sync() } finally { await handle.close() }
    try { await rename(temp,file) } catch (error) { await unlink(temp).catch(()=>{}); throw error }
    return j
  } finally { await lock.close(); await unlink(lockPath) }
}
