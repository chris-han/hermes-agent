param([string]$Operator=(Join-Path $PSScriptRoot '../deploy-ecs.ps1'))
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
$ast=[Management.Automation.Language.Parser]::ParseFile($Operator,[ref]$null,[ref]$null)
foreach($f in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)){. ([scriptblock]::Create($f.Extent.Text))}
function Require([bool]$Value,[string]$Message){if(!$Value){throw $Message}}
function Reject([scriptblock]$Action){$failed=$false;try{& $Action|Out-Null}catch{$failed=$true};Require $failed 'Unsafe cache accepted'}
$root=Join-Path ([IO.Path]::GetTempPath()) ('oss-cache-'+[guid]::NewGuid().ToString('N'))
$null=[IO.Directory]::CreateDirectory((Join-Path $root 'setup-download'))
$utf8=[Text.UTF8Encoding]::new($false)
try {
 $Output=$root;$OssCacheOnlyRecovery=$true;$OssSetupQualification=$true;$NativePublishOnly=$true;$SkipLocalInstall=$true;$BuildMode='Prod'
 $SourceCommit='b'*40;$BuildId='c'*32;$Version='0.0.131.0';$ExpectedChannelSha256='a'*64
 $SetupExe=Join-Path $root 'setup.exe';[IO.File]::WriteAllText($SetupExe,'signed setup fixture')
 $Package=Join-Path $root 'SemantierAgent.msix';[IO.File]::WriteAllText($Package,'original package')
 $download=Join-Path $root 'setup-download/SemantierAgent.msix';Copy-Item $Package $download
 $receipt=[ordered]@{sha256=(Hash $Package);ossFullTransferCount=1}
 $metadata=Join-Path $root 'channel1-metadata.json'
 $m=@{request=@{buildId=$BuildId;commit=$SourceCommit;windowsVersion=$Version};packages=@(@{artifact=@{key=('releases/channel-builds/'+$BuildId+'/SemantierAgent.msix');sha256=$receipt.sha256;size=(Get-Item $Package).Length}})}
 @{manifest=$m}|ConvertTo-Json -Depth 8|Set-Content $metadata
 $prior=[pscustomobject]@{result='FAIL';ossFullTransferCount=1;sha256=$receipt.sha256;plan=[pscustomobject]@{sourceCommit=$SourceCommit;buildId=$BuildId;version=$Version};setupTransport=[pscustomobject]@{sha256=(Hash $SetupExe)};error='original signature-child failure';finishedAt='2026-11-19T14:37:12Z'}
 $script:checkpoints=@{'oss-setup-transfer-started'='2026-11-19T14:35:46Z'}
 $script:statusLog=Join-Path $root 'checkpoints.jsonl';[IO.File]::WriteAllText($script:statusLog,'')
 $script:verifications=0
 function Assert-NoReparseAncestor($Path){}
 function Require-Signature($Path){}
 function Get-OssManifestHash($Path){return Hash $Path}
 function Invoke-RetainedSetupVerifiers($Setup,$File){$script:verifications++}
 function Complete-Checkpoint($Name){$script:checkpoints[$Name]='verified'}
 function Run-Setup {throw 'NETWORK MUST NEVER BE CALLED'}
 $OssCacheOnlyRecovery=$false;Reject {Confirm-OssCachedDownload $Package $metadata $prior};$OssCacheOnlyRecovery=$true
 $script:checkpoints.Clear();Reject {Confirm-OssCachedDownload $Package $metadata $prior};$script:checkpoints['oss-setup-transfer-started']='2026-11-19T14:35:46Z'
 $prior.ossFullTransferCount=2;Reject {Confirm-OssCachedDownload $Package $metadata $prior};$prior.ossFullTransferCount=1
 $prior.plan.buildId='d'*32;Reject {Confirm-OssCachedDownload $Package $metadata $prior};$prior.plan.buildId=$BuildId
 $prior.setupTransport.sha256='0'*64;Reject {Confirm-OssCachedDownload $Package $metadata $prior};$prior.setupTransport.sha256=Hash $SetupExe
 [IO.File]::WriteAllText($download,'bad bytes');Reject {Confirm-OssCachedDownload $Package $metadata $prior};Copy-Item $Package $download -Force
 @{buildId=$BuildId;key=$m.packages[0].artifact.key;sha256=$receipt.sha256;size=$m.packages[0].artifact.size;stagingKey='historical'}|ConvertTo-Json|Set-Content (Join-Path $root 'oss-staged.json')
 Confirm-OssCachedDownload $Package $metadata $prior
 Require ((Test-Path (Join-Path $root 'oss-staged-legacy.json')) -and !(Test-Path (Join-Path $root 'oss-staged.json'))) 'Legacy staging receipt overwritten'
 $proof=Get-Content (Join-Path $root 'setup-download/download-receipt.json') -Raw|ConvertFrom-Json
 Require ($proof.qualificationMode -ceq 'CACHE_ONLY_RECOVERY') 'Recovery provenance missing'
 Require ($proof.originalTransferOutcome -ceq 'FAILED_POST_DOWNLOAD') 'Original failure erased'
 Require ($proof.bodyByteEvidence -ceq 'RECONSTRUCTED_APPEND_ONLY_OUTPUT_LENGTH') 'Counter provenance invented'
 Require ($null -eq $proof.protocol -and $null -eq $proof.responses) 'Missing counters fabricated'
 Require ($script:verifications -eq 1 -and $script:checkpoints.ContainsKey('oss-setup-qualified')) 'Verifier/checkpoint missing'
 Require ($receipt.ossFullTransferCount -eq 1) 'Transfer allowance changed'
 Require ((Get-Content $script:statusLog -Raw) -match 'original signature-child failure') 'Original failure not preserved'
 # Proof survives an interruption before qualification checkpoint without downloading.
 $script:checkpoints.Remove('oss-setup-qualified');Confirm-OssCachedDownload $Package $metadata $prior
 Require ($script:verifications -eq 2) 'Retained proof not revalidated'
 Write-Output 'PASS: cache-only qualification, original failure/counters, replay, and scope/attempt/build/setup/hash denials'
}finally{Remove-Item $root -Recurse -Force}
