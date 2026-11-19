param([string]$Operator=(Join-Path $PSScriptRoot '../deploy-ecs.ps1'))
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
$utf8=[Text.UTF8Encoding]::new($false)
$ast=[Management.Automation.Language.Parser]::ParseFile($Operator,[ref]$null,[ref]$null)
foreach($fn in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)){. ([scriptblock]::Create($fn.Extent.Text))}
function Require([bool]$Ok,[string]$Message){if(!$Ok){throw $Message}}
$root=Join-Path ([IO.Path]::GetTempPath()) ('preparation-resume-'+[guid]::NewGuid().ToString('N'))
$null=[IO.Directory]::CreateDirectory($root)
try {
 $capture=$ast.Find({param($n)$n -is [Management.Automation.Language.AssignmentStatementAst] -and $n.Left.Extent.Text -ceq '$receipt.previousFailure'},$true)
 $priorOutcome=[pscustomobject]@{result='STOP';error='preparation failed'};$receipt=[ordered]@{}
 . ([scriptblock]::Create($capture.Extent.Text))
 Require ($null -eq $receipt.previousFailure.finishedAt) 'Missing historical timestamp was fabricated'
 $priorOutcome|Add-Member finishedAt '2026-11-19T12:00:00Z'
 . ([scriptblock]::Create($capture.Extent.Text))
 Require ($receipt.previousFailure.finishedAt -ceq $priorOutcome.finishedAt) 'Historical timestamp changed'
 $ToolingQualification='';$script:qualifiedTooling=$null;$Commit='';$SourceCommit='a'*40
 $Output=$root;$NativePublishOnly=$true;$SkipLocalInstall=$true;$OssSetupQualification=$true
 $NativeUpdateFrom='0.0.130.0';$Version='0.0.131.0';$BuildId='b'*32
 $Msix=Join-Path $root 'unsigned.msix';[IO.File]::WriteAllText($Msix,'verified original package')
 $ExpectedInputSha256=Hash $Msix;$script:statusLog=Join-Path $root 'checkpoints.jsonl'
 $request=Join-Path $root 'channel-request.json'
 @{buildId=$BuildId;commit=$SourceCommit;version='0.0.131'}|ConvertTo-Json|Set-Content $request
 $build=[pscustomobject]@{stage='built';commit=$SourceCommit;version='0.0.131';msix=$Msix;sha256=$ExpectedInputSha256;request=$request}
 Write-RunEvent @{type='build';build=$build}
 $previous=[pscustomobject]@{result='STOP';commit=$SourceCommit;signing='NOT_RUN';installation='NOT_RUN';publication='NOT_RUN';error='retained failure'}
 Require (Preserve-ToolingPreparationFailure $previous) 'Exact built-package continuation rejected'
 Require (Preserve-ToolingPreparationFailure $previous) 'Identical failure replay rejected'
 Require (@(Get-Content $statusLog|Where-Object {$_ -match '"type":"build-preparation-failure"'}).Count -eq 1) 'Failure history duplicated'
 $previous.commit='c'*40;Require (!(Preserve-ToolingPreparationFailure $previous)) 'Foreign source accepted';$previous.commit=$SourceCommit
 $previous.signing='PASS';Require (!(Preserve-ToolingPreparationFailure $previous)) 'Signed outcome accepted';$previous.signing='NOT_RUN'
 $previous.publication='PASS';Require (!(Preserve-ToolingPreparationFailure $previous)) 'Published outcome accepted';$previous.publication='NOT_RUN'
 $BuildId='d'*32;Require (!(Preserve-ToolingPreparationFailure $previous)) 'Changed build ID accepted';$BuildId='b'*32
 $Version='0.0.132.0';Require (!(Preserve-ToolingPreparationFailure $previous)) 'Changed version accepted';$Version='0.0.131.0'
 $OssSetupQualification=$false;Require (!(Preserve-ToolingPreparationFailure $previous)) 'Unselected mode accepted';$OssSetupQualification=$true
 $previous.error='different history';$caught=$false;try{Preserve-ToolingPreparationFailure $previous}catch{$caught=$true};Require $caught 'Conflicting historical failure accepted';$previous.error='retained failure'
 $originalHash=$ExpectedInputSha256
 $ExpectedInputSha256='0'*64;Require (!(Preserve-ToolingPreparationFailure $previous)) 'Changed package hash accepted'
 $ExpectedInputSha256=$originalHash;[IO.File]::WriteAllText($Msix,'corrupted package');Require (!(Preserve-ToolingPreparationFailure $previous)) 'Changed package bytes accepted'
 $lease=[IO.File]::Open($script:statusLog,'Open','ReadWrite','None');$lease.Dispose()
 Write-Output 'PASS: missing timestamp, exact built-package resume, replay and source/signing/publication/build/version/hash denials'
} finally {Remove-Item $root -Recurse -Force}
