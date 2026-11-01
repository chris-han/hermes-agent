// Diagnostic policy only. Caller-supplied facts are not mutation authority;
// a future transaction must collect and revalidate OS-backed ownership itself.
import path from 'node:path'

const identity = 'Semantier.Agent.Acceptance'
const publisher = 'CN=Semantier Technology, O=Semantier Technology'
const validSid = s => /^S-1-5-21-(?:\d+-){3}\d+$/i.test(String(s))
const cleanPath = p => typeof p === 'string' && path.win32.isAbsolute(p) && !p.split(/[\\/]/).includes('..')
const inside = (root, child) => {
  if (!cleanPath(root) || !cleanPath(child)) return false
  const relative = path.win32.relative(root, child)
  return relative !== '' && relative !== '..' && !relative.startsWith('..\\') && !path.win32.isAbsolute(relative)
}
const version = v => {
  if (!/^(?:0|[1-9]\d*)(?:\.(?:0|[1-9]\d*)){3}$/.test(String(v))) throw new Error('Invalid package version')
  const parts = v.split('.').map(Number)
  if (parts.some(n => n > 65535)) throw new Error('Invalid package version')
  return parts
}

export function validateSession(s) {
  if (!validSid(s.actualSid) || s.actualSid !== s.expectedSid || s.markerOwner !== s.actualSid ||
      s.interactive !== true || !Number.isInteger(s.sessionId) || s.sessionId <= 0 ||
      String(s.username).toLowerCase() === 'chrish' || s.preexistingPackage !== false ||
      !inside(path.win32.join(s.userProfile || '', 'semantier-acceptance'), s.ownedRoot)) {
    throw new Error('Unqualified isolated session or owned root')
  }
}

export function validatePackage(old, next, oldHash, nextHash) {
  for (const [pkg, hash] of [[old, oldHash], [next, nextHash]]) {
    if (!pkg || pkg.name !== identity || pkg.publisher !== publisher || pkg.arch !== 'x64' ||
        pkg.signature !== 'Valid' || !/^[a-f0-9]{64}$/.test(String(hash)) || pkg.sha256 !== hash) {
      throw new Error('Package identity, signature or digest mismatch')
    }
  }
  const a = version(old.version), b = version(next.version)
  const cmp = b.findIndex((n, i) => n !== a[i])
  if (cmp < 0 || b[cmp] < a[cmp]) throw new Error('Successor version must increase')
}

export function validateCleanup(c) {
  if (!validSid(c.actualSid) || c.actualSid !== c.expectedSid || c.evidenceCaptured !== true ||
      !c.expectedPackage || c.actualPackage !== c.expectedPackage || !Number.isInteger(c.expectedPid) ||
      c.expectedPid <= 0 || c.actualPid !== c.expectedPid || !c.expectedBirth || c.actualBirth !== c.expectedBirth ||
      !inside(c.installRoot, c.executable)) throw new Error('Unproven cleanup ownership')
}
