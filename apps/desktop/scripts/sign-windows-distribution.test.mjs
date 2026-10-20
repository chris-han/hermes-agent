import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import {execFileSync} from 'node:child_process'
import {test} from 'vitest'

test.skipIf(process.platform!=='win32')('native signing runner checks failures and records credential-free timings',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'distribution-exits-'))
 const quote=s=>"'"+s.replaceAll("'","''")+"'"
 const script=path.join(import.meta.dirname,'sign-windows-distribution.ps1')
 const command=`
 $ErrorActionPreference='Stop'
 $Output=${quote(root)}
 $tokens=$null;$errors=$null
 $ast=[Management.Automation.Language.Parser]::ParseFile(${quote(script)},[ref]$tokens,[ref]$errors)
 if($errors.Count){throw 'Script parse error'}
 $fn=$ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Run-Native'},$true)
 Invoke-Expression $fn.Extent.Text
 Run-Native $env:ComSpec @('/d','/c','exit 0')
 $failed=$false;try{Run-Native $env:ComSpec @('/d','/c','exit 7')}catch{$failed=$true}
 if(!$failed){throw 'Native failure ignored'}
 $failed=$false;try{Run-Native (Join-Path $Output 'missing.exe') @('check')}catch{$failed=$true}
 if(!$failed){throw 'Missing tool ignored'}
 `
 try {
  execFileSync(path.join(process.env.SystemRoot,'System32/WindowsPowerShell/v1.0/powershell.exe'),
   ['-NoProfile','-NonInteractive','-EncodedCommand',Buffer.from(command,'utf16le').toString('base64')],{stdio:'pipe',env:{...process.env,PRIVATE_TEST_CANARY:'never-log-this-canary'}})
  const raw=fs.readFileSync(path.join(root,'distribution-stages.jsonl'),'utf8')
  const rows=raw.replace(/^\uFEFF/,'').trim().split(/\r?\n/).map(line=>JSON.parse(line.replace(/^\uFEFF/,'')))
  assert.deepEqual(rows.map(r=>r.exitCode),[0,7,null])
  assert.ok(rows.every(r=>r.elapsedSeconds>=0 && r.producedAt.endsWith('Z')))
  assert.ok(!raw.includes('never-log-this-canary'))
 } finally {fs.rmSync(root,{recursive:true,force:true})}
})
