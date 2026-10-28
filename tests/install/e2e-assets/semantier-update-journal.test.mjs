import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { createJournal, advanceJournal, readJournal } from './semantier-update-journal.mjs'

test('records ordered observed transitions without accepting skips or rewrites', async () => {
 const dir=await mkdtemp(path.join(os.tmpdir(),'semantier-journal-')); const file=path.join(dir,'journal.json')
 try {
  await createJournal(file,{sid:'S-1-5-21-1-2-3-1001',sessionId:2,root:'C:\\owned',baselineSha256:'a'.repeat(64)})
  await assert.rejects(advanceJournal(file,'successor_verified',{packageFullName:'x'}))
  await advanceJournal(file,'baseline_registered',{packageFullName:'owned_0.0.58.0_x64__abc',family:'owned_abc',version:'0.0.58.0',installPath:'C:\\pkg',exe:'C:\\pkg\\agent.exe'})
  await assert.rejects(advanceJournal(file,'baseline_registered',{}))
  const j=await readJournal(file); assert.equal(j.phase,'baseline_registered'); assert.equal(j.events.length,2)
  await assert.rejects(createJournal(file,{}))
 } finally { await rm(dir,{recursive:true,force:true}) }
})
test('failed or interrupted updates remain recoverable without claiming success', async () => {
 const dir=await mkdtemp(path.join(os.tmpdir(),'semantier-journal-')); const file=path.join(dir,'journal.json')
 try {
  await createJournal(file,{sid:'S-1-5-21-1-2-3-1001',sessionId:2,root:'C:\\owned',baselineSha256:'a'.repeat(64)})
  await advanceJournal(file,'baseline_registered',{packageFullName:'owned',family:'family',version:'0.0.58.0',installPath:'C:\\pkg',exe:'C:\\pkg\\agent.exe'})
  await advanceJournal(file,'baseline_observed',{pid:123,birth:'2026-01-01T00:00:00Z',dataDigest:'b'.repeat(64)})
  await advanceJournal(file,'update_started',{targetSha256:'c'.repeat(64),targetVersion:'0.0.59.0'})
  const j=await readJournal(file); assert.equal(j.phase,'update_started'); assert.equal(j.events.some(e=>e.phase==='successor_verified'),false)
  await assert.rejects(advanceJournal(file,'cleanup_complete',{}))
 } finally { await rm(dir,{recursive:true,force:true}) }
})

test('concurrent advances cannot both claim the same predecessor', async () => {
 const dir=await mkdtemp(path.join(os.tmpdir(),'semantier-journal-')); const file=path.join(dir,'journal.json')
 try {
  await createJournal(file,{sid:'S-1-5-21-1-2-3-1001',sessionId:2,root:'C:\\owned',baselineSha256:'a'.repeat(64)})
  const facts={packageFullName:'owned',family:'family',version:'0.0.58.0',installPath:'C:\\pkg',exe:'C:\\pkg\\agent.exe'}
  const r=await Promise.allSettled([advanceJournal(file,'baseline_registered',facts),advanceJournal(file,'baseline_registered',facts)])
  assert.equal(r.filter(x=>x.status==='fulfilled').length,1)
  assert.equal((await readJournal(file)).events.length,2)
 } finally { await rm(dir,{recursive:true,force:true}) }
})
