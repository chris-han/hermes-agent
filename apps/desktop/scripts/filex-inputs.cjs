// Packaged filex is immutable upstream content under the signed package envelope.
const fs = require('node:fs')
const path = require('node:path')
const { createHash } = require('node:crypto')
/** @type {Record<string, string>} */
const pins = {
  'win32-x64': '190305d0e07c882e1304ebbddccc338a805ad00e3413dd48249f8d01f8729ed5',
  'linux-x64': '331004fe75de55b1a6e08165dc172a8ae357cb1799862e89b2bb59b773df5f11'
}
/** @param {import("node:crypto").BinaryLike} bytes */
const hash = bytes => createHash('sha256').update(bytes).digest('hex')
/** @param {string} desktop @param {string} platform @param {string} arch @param {string} resources */
function verifyFilexInputs(desktop, platform = process.platform, arch = process.arch, resources = path.join(desktop, 'resources', 'filex')) {
  const expected = pins[`${platform}-${arch}`]
  if (!expected) throw new Error('Unsupported filex package platform')
  const root = resources
  const binary = path.join(root, platform === 'win32' ? 'filex.exe' : 'filex')
  if (hash(fs.readFileSync(binary)) !== expected) throw new Error('Pinned filex checksum mismatch')
  if (hash(fs.readFileSync(path.join(root, 'filex-child.mjs'))) !== hash(fs.readFileSync(path.join(desktop, 'electron', 'filex-child.mjs'))))
    throw new Error('Stale filex child helper; prepare package inputs again')
  const receipt = JSON.parse(fs.readFileSync(path.join(root, 'inputs.json'), 'utf8'))
  if (receipt.platform !== platform || receipt.arch !== arch || receipt.version !== '0.61.1' || receipt.commit !== '2964aba36586b32d142d8fa9edba37d9088541d0' || receipt.binarySha256 !== expected || receipt.licenseSha256 !== '77c273cd1dbfdb509dd9ca13f81497c9ea8e5b28f7bd41b7bc902399350ede68' || receipt.licenseSha256 !== hash(fs.readFileSync(path.join(root, 'LICENSE.filex'))))
    throw new Error('Stale or wrong-platform filex inputs')
  const noticeHash = hash(fs.readFileSync(path.join(root, 'THIRD-PARTY-NOTICES.txt')))
  if (receipt.noticesSha256 !== noticeHash || noticeHash !== hash(fs.readFileSync(path.join(desktop, 'assets', 'filex-THIRD-PARTY-NOTICES.txt')))) throw new Error('Stale filex redistribution notices')
  return receipt
}
/** @param {string} file */
function isPinnedFilex(file) {
  return /(?:^|[\\/])resources[\\/]filex[\\/]filex\.exe$/i.test(file)
}
module.exports = { pins, hash, verifyFilexInputs, isPinnedFilex }
