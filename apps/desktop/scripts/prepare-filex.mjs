// Pinned package/development input preparation; no installer or publication.
import fs from 'node:fs/promises'
import path from 'node:path'
import { createHash } from 'node:crypto'
import { fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'
const { hash } = createRequire(import.meta.url)('./filex-inputs.cjs')
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const platform = process.argv.find(arg => arg.startsWith('--platform='))?.split('=')[1] || process.platform
if (process.arch !== 'x64' || !['linux', 'win32'].includes(platform))
  throw new Error('Phase 1 qualifies x64 WSL and Windows only.')
const name = platform === 'win32' ? 'filex-windows-amd64.exe' : 'filex-linux-amd64'
const expected =
  platform === 'win32'
    ? '190305d0e07c882e1304ebbddccc338a805ad00e3413dd48249f8d01f8729ed5'
    : '331004fe75de55b1a6e08165dc172a8ae357cb1799862e89b2bb59b773df5f11'
const out = path.join(root, 'resources', 'filex')
await fs.mkdir(out, { recursive: true })
const local = process.argv.slice(2).find(arg => !arg.startsWith('--'))
const bytes = local
  ? await fs.readFile(path.resolve(local))
  : Buffer.from(
      await (await fetch('https://github.com/BRF-Tech/filex/releases/download/v0.61.1/' + name)).arrayBuffer()
    )
if (createHash('sha256').update(bytes).digest('hex') !== expected)
  throw new Error('Pinned filex checksum mismatch; nothing installed.')
const destination = path.join(out, platform === 'win32' ? 'filex.exe' : 'filex')
await fs.writeFile(destination, bytes, { mode: 0o700 })
await fs.chmod(destination, 0o700)
const license = await fetch(
  'https://raw.githubusercontent.com/BRF-Tech/filex/2964aba36586b32d142d8fa9edba37d9088541d0/LICENSE'
)
if (!license.ok) throw new Error('Pinned filex license unavailable.')
const licenseBytes = Buffer.from(await license.arrayBuffer())
if (hash(licenseBytes) !== '77c273cd1dbfdb509dd9ca13f81497c9ea8e5b28f7bd41b7bc902399350ede68') throw new Error('Pinned filex license checksum mismatch')
await fs.writeFile(path.join(out, 'LICENSE.filex'), licenseBytes)
await fs.copyFile(path.join(root, 'electron', 'filex-child.mjs'), path.join(out, 'filex-child.mjs'))
await fs.writeFile(path.join(out, 'inputs.json'), JSON.stringify({ version: '0.61.1', commit: '2964aba36586b32d142d8fa9edba37d9088541d0', platform, arch: process.arch, binarySha256: expected, licenseSha256: hash(await fs.readFile(path.join(out, 'LICENSE.filex'))) }, null, 2) + '\n')
console.log('Prepared unmodified filex v0.61.1 for ' + platform + '; SHA-256 ' + expected)
