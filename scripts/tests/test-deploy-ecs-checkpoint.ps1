$ErrorActionPreference='Stop'
$path=Join-Path $PSScriptRoot '../deploy-ecs.ps1'
$parseErrors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$null,[ref]$parseErrors)
if($parseErrors){throw ($parseErrors | Out-String)}
$fn=$ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Write-RunEvent'},$true)
. ([scriptblock]::Create($fn.Extent.Text))
Add-Type @'
using System;
using System.IO;
using System.Threading;
public static class CheckpointLockFixture {
    public static Timer ReleaseLater(FileStream stream) {
        return new Timer(state => ((FileStream)state).Dispose(), stream, 300, Timeout.Infinite);
    }
}
'@
$utf8=[Text.UTF8Encoding]::new($false)
$script:statusLog=Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString()+'.jsonl')
try {
  [IO.File]::WriteAllText($script:statusLog,'')
  $reader=[IO.File]::Open($script:statusLog,'Open','Read','Read')
  $release=[CheckpointLockFixture]::ReleaseLater($reader)
  Write-RunEvent @{type='checkpoint';stage='test'}
  Write-RunEvent @{type='output';message='next event'}
  $records=@([IO.File]::ReadAllLines($script:statusLog) | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
  if($records.Count -ne 2 -or $records[0].stage -ne 'test' -or $records[1].message -ne 'next event'){throw 'Checkpoint records lost or duplicated'}
  $script:statusLog=Join-Path $script:statusLog 'missing/log.jsonl'
  $failed=$false
  try { Write-RunEvent @{type='checkpoint'} } catch [IO.IOException] { $failed=$true }
  if(!$failed){throw 'Non-contention failure was swallowed'}
  Write-Output 'PASS: transient lock recovery, complete records, and non-contention error propagation'
} finally {
  if($release){$release.Dispose()}
  if($reader){$reader.Dispose()}
  Remove-Item -LiteralPath ($script:statusLog -replace '[\\/]missing[\\/]log.jsonl$','') -ErrorAction SilentlyContinue
}
