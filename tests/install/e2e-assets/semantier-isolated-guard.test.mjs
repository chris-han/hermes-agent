import test from 'node:test'
import assert from 'node:assert/strict'
import { validateSession, validatePackage, validateCleanup } from './semantier-isolated-guard.mjs'

const sid = 'S-1-5-21-1-2-3-1001'
const session = { actualSid:sid, expectedSid:sid, interactive:true, username:'test-release', sessionId:2,
  userProfile:'C:\\Users\\test-release', ownedRoot:'C:\\Users\\test-release\\semantier-acceptance\\run-1',
  markerOwner:sid, preexistingPackage:false }
const base = { name:'Semantier.Agent.Acceptance', publisher:'CN=Semantier Technology, O=Semantier Technology', arch:'x64', version:'0.0.58.0', sha256:'a'.repeat(64), signature:'Valid' }
const next = { ...base, version:'0.0.59.0', sha256:'b'.repeat(64) }
test('accepts a matching disposable interactive owner and increasing trusted packages', () => {
  assert.doesNotThrow(() => validateSession(session)); assert.doesNotThrow(() => validatePackage(base,next,base.sha256,next.sha256))
})
test('refuses wrong or unsafe session, root and preexisting package', () => {
  for (const patch of [{actualSid:'S-1-5-18'}, {interactive:false}, {interactive:'yes'}, {username:'ChrisH'}, {sessionId:0},
    {ownedRoot:'C:\\Users\\ChrisH\\data'}, {ownedRoot:'C:\\Users\\test-release\\semantier-acceptance\\..\\other'},
    {markerOwner:'S-1-5-18'}, {preexistingPackage:true}]) assert.throws(() => validateSession({...session,...patch}))
})
test('refuses package identity, trust, digest, architecture and version mismatches', () => {
  for (const patch of [{name:'Other'}, {publisher:'Other'}, {arch:'arm64'}, {signature:'UnknownError'},
    {sha256:'c'.repeat(64)}, {version:'0.0.58.0'}, {version:'0.0.57.0'}])
    assert.throws(() => validatePackage(base,{...next,...patch},base.sha256,next.sha256))
})
test('cleanup refuses unrelated or interrupted ownership and accepts exact owned identity', () => {
  const owned={ expectedSid:sid, actualSid:sid, expectedPackage:'Semantier.Agent.Acceptance_0.0.59.0_x64__test', actualPackage:'Semantier.Agent.Acceptance_0.0.59.0_x64__test', expectedPid:123, actualPid:123, expectedBirth:'t', actualBirth:'t', installRoot:'C:\\Program Files\\WindowsApps\\owned', executable:'C:\\Program Files\\WindowsApps\\owned\\app\\agent.exe', evidenceCaptured:true }
  assert.doesNotThrow(()=>validateCleanup(owned))
  for (const patch of [{evidenceCaptured:false},{evidenceCaptured:'yes'},{actualSid:'S-1-5-18'},{actualPackage:'other'}, {actualPid:124}, {actualBirth:'u'}, {executable:'C:\\Program Files\\WindowsApps\\other\\agent.exe'}]) assert.throws(()=>validateCleanup({...owned,...patch}))
})
