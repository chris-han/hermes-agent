param([string]$Operator=(Join-Path $PSScriptRoot '../deploy-ecs.ps1'))
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
foreach($path in @($Operator,(Join-Path (Split-Path $Operator) 'deploy-ecs-channel1.ps1'))){
 $ast=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$null,[ref]$null)
 foreach($f in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)){. ([scriptblock]::Create($f.Extent.Text))}
}
function Require([bool]$Value,[string]$Message){if(!$Value){throw $Message}}
$root=Join-Path ([IO.Path]::GetTempPath()) ('oss-retention-'+[guid]::NewGuid().ToString('N'))
$null=New-Item (Join-Path $root 'setup-download') -ItemType Directory
$utf8=[Text.UTF8Encoding]::new($false)
try {
 $Output=$root;$NativePublishOnly=$true;$Commit='b'*40;$SourceCommit=$Commit;$ExpectedVersion='1.2.132.4';$RemoteRoot='/fixture/releases';$SupersededBuildOutput=''
 $package=Join-Path $root 'SemantierAgent.msix';[IO.File]::WriteAllText($package,'signed fixture');Copy-Item $package (Join-Path $root 'setup-download/SemantierAgent.msix')
 $hash=Hash $package;$metadata=Join-Path $root 'channel1-metadata.json';@{manifest=@{request=@{buildId=$ExpectedVersion}}}|ConvertTo-Json -Depth 6|Set-Content $metadata
 $proof=@{status='DOWNLOADED_VERIFIED_NOT_INSTALLED';selection='PINNED_STAGED_CANDIDATE';signature='PINNED_SIGNER_VALID';sha256=$hash;buildId=$ExpectedVersion;manifestSha256=('a'*64)}
 $proof|ConvertTo-Json|Set-Content (Join-Path $root 'setup-download/download-receipt.json')
 $saved=@{result='OSS_PUBLISHED_RETENTION_PENDING';plan=@{sourceCommit=$Commit;version=$ExpectedVersion;buildId=$ExpectedVersion};signedMsix=$package;sha256=$hash;ossFullTransferCount=1}
 $path=Join-Path $root 'result.json';$saved|ConvertTo-Json -Depth 8|Set-Content $path
 function Require-Signature($Path){}
 function Get-OssManifestHash($Path){return 'a'*64}
 function Test-NativePublic($Manifest,$Package,[switch]$Active){return @{status='PASS'}}
 function Report-Status($Text){}
 $script:sshCalls=0;$script:retentionCalls=0;$script:failRetention=$true
 function Ssh($Command){$script:sshCalls++;return 'READ_ONLY_PROOF'}
 function Run-Setup{throw 'Full transfer must never be attempted'}
 function Invoke-OssByteOperation($Mode,$Package,$Metadata,$Result){
  Require ($Mode -ceq 'retention') 'Unexpected byte operation';$script:retentionCalls++
  if($script:failRetention){throw 'fixture deletion interrupted'}
  return @{result='ONE_ACTIVE_MSIX_AND_SETUP'}
 }
 Require (Test-NativeCompletedRelease) 'Pending verified release not handled'
 $failed=Get-Content $path -Raw|ConvertFrom-Json
 Require ($failed.result -ceq 'OSS_PUBLISHED_RETENTION_PENDING' -and $failed.retentionError -ceq 'fixture deletion interrupted') 'Retention failure hidden'
 Require ($script:sshCalls -eq 1 -and $script:retentionCalls -eq 1 -and $failed.ossFullTransferCount -eq 1) 'Failure changed pointer/transfer allowance'
 $script:failRetention=$false
 Require (Test-NativeCompletedRelease) 'Retention replay not handled'
 $done=Get-Content $path -Raw|ConvertFrom-Json
 Require ($done.result -ceq 'OSS_PUBLISHED_SETUP_VERIFIED_NATIVE_NOT_RUN' -and $done.ossRetention.result -ceq 'ONE_ACTIVE_MSIX_AND_SETUP') 'Retention replay not recorded'
 Require ($script:sshCalls -eq 2 -and $script:retentionCalls -eq 2 -and $done.ossFullTransferCount -eq 1) 'Replay attempted rollback/transfer'
 Write-Output 'PASS: retention failure preserves verified active head and one-transfer proof; cleanup-only replay completes'
}finally{Remove-Item $root -Recurse -Force}
