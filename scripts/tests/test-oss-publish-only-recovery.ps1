param([string]$Operator=(Join-Path $PSScriptRoot '../deploy-ecs.ps1'))
$ErrorActionPreference='Stop';Set-StrictMode -Version Latest
foreach($path in @($Operator,(Join-Path (Split-Path $Operator) 'deploy-ecs-channel1.ps1'))){
 $ast=[Management.Automation.Language.Parser]::ParseFile($path,[ref]$null,[ref]$null)
 foreach($f in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)){. ([scriptblock]::Create($f.Extent.Text))}
}
function Require([bool]$Value,[string]$Message){if(!$Value){throw $Message}}
function Reject([scriptblock]$Action){$failed=$false;try{& $Action|Out-Null}catch{$failed=$true};Require $failed 'Required recovery verification skipped'}
$root=Join-Path ([IO.Path]::GetTempPath()) ('oss-publish-recovery-'+[guid]::NewGuid().ToString('N'))
$null=New-Item $root -ItemType Directory
try {
 $OssSetupQualification=$true;$NativePublishOnly=$true;$SkipLocalInstall=$true;$BuildMode='Prod'
 $RecoveryMsix=Join-Path $root 'missing.msix';$package=@{InstallLocation=$root};$script:signatures=0
 function Require-Signature($Path){$script:signatures++;if(!(Test-Path $Path)){throw 'Missing recovery archive'}}
 function Get-PackageSignatureHash($Path){return Hash $Path}
 $result=Get-NativeLocalRecovery $package
 Require ($result.result -ceq 'NOT_RUN_PUBLISH_ONLY_NO_INSTALL' -and $null -eq $result.signature -and $null -eq $result.sha256 -and $script:signatures -eq 0) 'Non-install mode required or invented local recovery'
 foreach($name in @('OssSetupQualification','NativePublishOnly','SkipLocalInstall')){
  Set-Variable $name $false;Reject {Get-NativeLocalRecovery $package};Set-Variable $name $true
 }
 $OssSetupQualification=$false
 $RecoveryMsix=Join-Path $root 'recovery.msix';[IO.File]::WriteAllText($RecoveryMsix,'signature fixture')
 Copy-Item $RecoveryMsix (Join-Path $root 'AppxSignature.p7x')
 $result=Get-NativeLocalRecovery $package
 Require ($result.result -ceq 'VERIFIED_LOCAL_RECOVERY' -and $result.sha256 -ceq (Hash $RecoveryMsix)) 'Legacy recovery verification changed'
 [IO.File]::WriteAllText((Join-Path $root 'AppxSignature.p7x'),'wrong signature')
 Reject {Get-NativeLocalRecovery $package}
 Write-Output 'PASS: explicit OSS no-install mode needs no installed recovery archive; other modes preserve signature/matching-recovery checks'
}finally{Remove-Item $root -Recurse -Force}
