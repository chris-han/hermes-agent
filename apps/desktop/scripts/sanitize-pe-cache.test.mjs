import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import {test} from 'vitest'
import {sanitizeTreeCached} from './sanitize-pe-cache.mjs'

function pe(offset=400,size=32) {
 const b=Buffer.alloc(464,0xa5);b.write('MZ');b.writeUInt32LE(128,0x3c);b.writeUInt32LE(0x4550,128)
 b.writeUInt16LE(240,148);b.writeUInt16LE(0x20b,152);b.writeUInt32LE(16,260)
 b.writeUInt32LE(offset,296);b.writeUInt32LE(size,300);return b
}

test('content hits avoid parsing; same-size/mtime replacement and rule corruption recheck',async()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'pe-cache-'));const root=path.join(dir,'app');fs.mkdirSync(root)
 const receipt=path.join(dir,'cache/receipt.json'), file=path.join(root,'no-pe-extension.data')
 try {
  const original=pe();fs.writeFileSync(file,original);fs.writeFileSync(path.join(root,'readme.txt'),'ordinary data')
  const first=await sanitizeTreeCached(root,receipt);assert.equal(first.misses,2);assert.equal(first.scanned,1)
  const warm=await sanitizeTreeCached(root,receipt);assert.equal(warm.hits,2);assert.equal(warm.misses,0)
  assert.deepEqual(fs.readFileSync(file),original)
  const stamp=fs.statSync(file);fs.writeFileSync(file,pe(464,24));fs.utimesSync(file,stamp.atime,stamp.mtime)
  const changed=await sanitizeTreeCached(root,receipt);assert.equal(changed.misses,1);assert.equal(changed.repaired.length,1)
  assert.equal(fs.readFileSync(file).readUInt32LE(296),0)
  const again=await sanitizeTreeCached(root,receipt);assert.equal(again.misses,0)
  const data=JSON.parse(fs.readFileSync(receipt));data.rules='old';fs.writeFileSync(receipt,JSON.stringify(data))
  assert.equal((await sanitizeTreeCached(root,receipt)).misses,2)
  fs.writeFileSync(receipt,'corrupt');assert.equal((await sanitizeTreeCached(root,receipt)).misses,2)
 } finally {fs.rmSync(dir,{recursive:true,force:true})}
})
