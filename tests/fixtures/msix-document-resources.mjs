import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const [repo, matcherModule] = process.argv.slice(2)
const { FileMatcher, copyFiles } = await import(pathToFileURL(matcherModule).href)
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-msix-docx-'))
const source = path.join(temporary, 'source')
const destination = path.join(temporary, 'packed')
const fixture = {
  'venv/Lib/site-packages/docx/templates/default-docx-template/customXml/_rels/item1.xml.rels': 'source relationships',
  'venv/Lib/site-packages/docx/templates/default-docx-template/word/document.xml': 'source document',
  'venv/Lib/site-packages/docx/templates/default.docx': 'runtime default ZIP',
  'venv/Lib/site-packages/docx/templates/default-header.xml': 'runtime header',
  'venv/Lib/site-packages/docx/templates/custom.docx': 'custom document ZIP',
  'venv/Lib/site-packages/docx/opc/phys_pkg.py': 'custom package reader',
  'venv/Lib/site-packages/python_docx-1.2.0.dist-info/licenses/LICENSE': 'license',
  'uv-cache/seed.zip': 'complete offline cache',
}
try {
  for (const [relative, content] of Object.entries(fixture)) {
    const file = path.join(source, relative)
    fs.mkdirSync(path.dirname(file), { recursive: true })
    fs.writeFileSync(file, content)
  }
  for (const platform of ['win32', 'linux']) {
    const script = `Object.defineProperty(process,'platform',{value:process.argv[1]});process.env.HERMES_DESKTOP_VARIANT='bundled';const config=require(process.argv[2]);process.stdout.write(JSON.stringify(config.extraResources.find(r=>r.to==='agent-payload')))`
    const resource = JSON.parse(execFileSync(process.execPath, ['-e', script, platform,
      path.join(repo, 'apps/desktop/electron-builder.config.cjs')], { encoding: 'utf8' }))
    const target = path.join(destination, platform)
    const matcher = new FileMatcher(source, target, value => value, resource.filter)
    await copyFiles([matcher], null, false)
    for (const [relative, content] of Object.entries(fixture)) {
      const omitted = platform === 'win32' && relative.includes('/default-docx-template/')
      const copied = path.join(target, relative)
      assert.equal(fs.existsSync(copied), !omitted, `${platform}: ${relative}`)
      if (!omitted) assert.equal(fs.readFileSync(copied, 'utf8'), content)
      assert.equal(fs.readFileSync(path.join(source, relative), 'utf8'), content)
    }
  }
  console.log('Real electron-builder resource copy: Windows excludes only inert docx template source; runtime/custom templates, reader, license, seed and source remain intact')
} finally {
  fs.rmSync(temporary, { recursive: true, force: true })
}
