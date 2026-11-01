param([string]$ChildRoot='')
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'native-package-deployment.ps1')
if($ChildRoot){
 function Invoke-BoundNativePackageUri([string[]]$Uris,[string[]]$Digests,[string]$Root,[int]$Minutes) {
  $p=@(Get-ChildItem -LiteralPath $ChildRoot -Filter 'ci-native-intent-*.json')
  if($p.Count -ne 1){[Environment]::Exit(78)}
  [Environment]::Exit(77)
 }
 $null=Invoke-CIDurableNativeUri $ChildRoot ('a'*64) ('b'*40) 'positive' @{sid='fixture-sid';session_sid='fixture-sid';session=7;wts_state=0} @('http://127.0.0.1:1/pkg') @('digest') 'http://127.0.0.1:1/pkg' 1
 [Environment]::Exit(79)
}
$root=Join-Path ([IO.Path]::GetTempPath()) ('semantier-ci-crash-test-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $root -ErrorAction Stop|Out-Null
try {
 $child=Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe') -ArgumentList @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',('"'+$PSCommandPath+'"'),'-ChildRoot',('"'+$root+'"')) -PassThru -Wait
 if($child.ExitCode -ne 77){throw 'Fixture did not terminate at the SDK invocation boundary'}
 $intents=@(Get-ChildItem -LiteralPath $root -Filter 'ci-native-intent-*.json');$outcomes=@(Get-ChildItem -LiteralPath $root -Filter 'ci-native-outcome-*.json')
 if($intents.Count -ne 1 -or $outcomes.Count -ne 0){throw 'Terminated invocation was not left unresolved'}
 $i=Get-Content -Raw -LiteralPath $intents[0].FullName|ConvertFrom-Json
 if($i.graphSha256 -cne ('a'*64) -or $i.sourceCommit -cne ('b'*40) -or $i.operation -cne 'positive'){throw 'Unresolved intent binding changed'}
 'PASS: real wrapper records intent before simulated SDK boundary termination; no native deployment'
} finally {Remove-Item -LiteralPath $root -Recurse -Force}
