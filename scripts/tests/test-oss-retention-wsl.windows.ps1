param([Parameter(Mandatory)][string]$UnqualifiedRun,[Parameter(Mandatory)][string]$ResultPath,[string]$Runtime='D:\repo\semantier-runtime')
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
if(!$IsWindows){throw 'Native Windows/WSL boundary required'}
if(Test-Path $ResultPath){throw 'Fresh test output required'}
$Repository=Join-Path $Runtime 'hermes-agent';$script:RuntimeRoot=$Runtime
foreach($path in @((Join-Path $Repository 'scripts/deploy-ecs.ps1'),(Join-Path $Repository 'scripts/deploy-ecs-channel1.ps1'))){
 $ast=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$null,[ref]$null)
 foreach($f in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)){. ([scriptblock]::Create($f.Extent.Text))}
}
function Report-Status($Text){}
$OssRuntimeRoot='/home/chris/repo/semantier-runtime';$OssWslDistro='Ubuntu-22.04';$OssPreviousSetupSha256='';$SupersededBuildOutput=''
$proof=Get-Content (Join-Path $UnqualifiedRun 'result.json') -Raw|ConvertFrom-Json
if($proof.result -cne 'STOP' -or $proof.publication -cne 'NOT_STARTED'){throw 'Test requires an unqualified failed run, never an active release'}
$messages=[Collections.Generic.List[string]]::new();$rejected=$false
try {
 Invoke-OssByteOperation 'retention' (Join-Path $UnqualifiedRun 'SemantierAgent.msix') (Join-Path $UnqualifiedRun 'channel1-metadata.json') $ResultPath 2>&1 | ForEach-Object {$messages.Add([string]$_)}
}catch{$rejected=$true;$messages.Add($_.Exception.Message)}
if(!$rejected -or (Test-Path $ResultPath) -or ($messages -join "`n") -notmatch 'OSS_RETENTION_UNQUALIFIED'){
 throw ('Expected cross-owner qualification denial not observed: '+($messages -join "`n"))
}
Write-Output 'PASS: native/WSL retention mode and pinned source wiring; unqualified run refused before cloud writes or transfers'
